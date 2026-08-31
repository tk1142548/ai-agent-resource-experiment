from __future__ import annotations

import json
import os
import platform
import threading
import time
import tracemalloc
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psutil

from . import SCHEMA_VERSION


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_size(value: Any) -> int:
    return len(json.dumps(value, ensure_ascii=False, default=str, separators=(",", ":")).encode("utf-8"))


def process_totals(process: psutil.Process) -> dict[str, float | int]:
    processes = [process]
    try:
        processes.extend(process.children(recursive=True))
    except (psutil.Error, OSError):
        pass
    cpu = 0.0
    rss = 0
    for item in processes:
        try:
            times = item.cpu_times()
            cpu += float(times.user + times.system)
            rss += int(item.memory_info().rss)
        except (psutil.Error, OSError):
            continue
    return {"cpu_seconds": cpu, "rss_bytes": rss}


class EventLog:
    def __init__(self, path: Path, experiment_id: str):
        self.path = path
        self.experiment_id = experiment_id
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def emit(self, event_type: str, **fields: Any) -> dict[str, Any]:
        event = {
            "schema_version": SCHEMA_VERSION,
            "experiment_id": self.experiment_id,
            "event_type": event_type,
            "timestamp_utc": utc_now(),
            "monotonic_ns": time.monotonic_ns(),
            **fields,
        }
        line = json.dumps(event, ensure_ascii=False, default=str, separators=(",", ":"))
        with self._lock:
            with self.path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(line + "\n")
                handle.flush()
        return event

    def completed_run_ids(self) -> set[str]:
        if not self.path.exists():
            return set()
        completed: set[str] = set()
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                event = json.loads(line)
                if event.get("event_type") == "task_end" and event.get("run_id"):
                    completed.add(str(event["run_id"]))
        return completed


class WindowsGpuCounters:
    """Best-effort process-scoped Windows GPU counters."""

    def __init__(self, pid: int):
        self.pid = pid
        self.available = False
        self.query = None
        self.running: list[Any] = []
        self.dedicated: list[Any] = []
        try:
            import win32pdh

            self.pdh = win32pdh
            self.query = win32pdh.OpenQuery()
            for object_name, counter_name, target in (
                ("GPU Engine", "Running Time", self.running),
                ("GPU Process Memory", "Dedicated Usage", self.dedicated),
            ):
                _, instances = win32pdh.EnumObjectItems(
                    None, None, object_name, win32pdh.PERF_DETAIL_WIZARD
                )
                for instance in instances:
                    if f"pid_{pid}_" not in instance.lower():
                        continue
                    path = win32pdh.MakeCounterPath(
                        (None, object_name, instance, None, 0, counter_name)
                    )
                    target.append(win32pdh.AddCounter(self.query, path))
            win32pdh.CollectQueryData(self.query)
            self.available = True
        except Exception:
            self.close()

    def sample(self) -> dict[str, float | int]:
        if not self.available or self.query is None:
            return {"gpu_running_time_100ns": 0, "gpu_dedicated_bytes": 0}
        try:
            self.pdh.CollectQueryData(self.query)
            running = sum(
                int(self.pdh.GetFormattedCounterValue(c, self.pdh.PDH_FMT_LARGE)[1] or 0)
                for c in self.running
            )
            dedicated = sum(
                int(self.pdh.GetFormattedCounterValue(c, self.pdh.PDH_FMT_LARGE)[1] or 0)
                for c in self.dedicated
            )
            return {
                "gpu_running_time_100ns": running,
                "gpu_dedicated_bytes": dedicated,
            }
        except Exception:
            return {"gpu_running_time_100ns": 0, "gpu_dedicated_bytes": 0}

    def close(self) -> None:
        try:
            if self.query is not None:
                self.pdh.CloseQuery(self.query)
        except Exception:
            pass
        self.query = None


class TaskResourceMonitor:
    def __init__(self, interval_seconds: float = 0.02):
        self.interval_seconds = interval_seconds
        self.process = psutil.Process(os.getpid())
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.start_values: dict[str, float | int] = {}
        self.end_values: dict[str, float | int] = {}
        self.peak_rss_bytes = 0
        self.peak_gpu_dedicated_bytes = 0
        self.gpu = WindowsGpuCounters(os.getpid()) if platform.system() == "Windows" else None

    def start(self) -> None:
        self.start_values = process_totals(self.process)
        gpu = self.gpu.sample() if self.gpu else {"gpu_running_time_100ns": 0, "gpu_dedicated_bytes": 0}
        self.start_values.update(gpu)
        self.peak_rss_bytes = int(self.start_values["rss_bytes"])
        self.peak_gpu_dedicated_bytes = int(gpu["gpu_dedicated_bytes"])
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def _sample_loop(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            values = process_totals(self.process)
            self.peak_rss_bytes = max(self.peak_rss_bytes, int(values["rss_bytes"]))
            if self.gpu:
                gpu = self.gpu.sample()
                self.peak_gpu_dedicated_bytes = max(
                    self.peak_gpu_dedicated_bytes, int(gpu["gpu_dedicated_bytes"])
                )

    def stop(self) -> dict[str, float | int]:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        self.end_values = process_totals(self.process)
        gpu = self.gpu.sample() if self.gpu else {"gpu_running_time_100ns": 0, "gpu_dedicated_bytes": 0}
        self.end_values.update(gpu)
        if self.gpu:
            self.gpu.close()
        return {
            "cpu_seconds": max(
                0.0,
                float(self.end_values["cpu_seconds"]) - float(self.start_values["cpu_seconds"]),
            ),
            "rss_start_bytes": int(self.start_values["rss_bytes"]),
            "rss_end_bytes": int(self.end_values["rss_bytes"]),
            "rss_peak_bytes": int(self.peak_rss_bytes),
            "gpu_running_seconds": max(
                0.0,
                (int(self.end_values["gpu_running_time_100ns"]) - int(self.start_values["gpu_running_time_100ns"]))
                / 10_000_000.0,
            ),
            "gpu_dedicated_peak_bytes": int(self.peak_gpu_dedicated_bytes),
        }


@dataclass
class ToolMeasurement:
    wall_seconds: float
    cpu_seconds: float
    rss_before_bytes: int
    rss_after_bytes: int
    python_peak_allocated_bytes: int


class ToolTimer:
    def __init__(self):
        self.process = psutil.Process(os.getpid())

    def measure(self, function, *args, **kwargs):
        before = process_totals(self.process)
        tracemalloc.start()
        start = time.monotonic()
        try:
            result = function(*args, **kwargs)
            return result, None, self._finish(start, before)
        except Exception as exc:  # caller records and handles the error
            return None, exc, self._finish(start, before)

    def _finish(self, start: float, before: dict[str, float | int]) -> ToolMeasurement:
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        after = process_totals(self.process)
        return ToolMeasurement(
            wall_seconds=time.monotonic() - start,
            cpu_seconds=max(0.0, float(after["cpu_seconds"]) - float(before["cpu_seconds"])),
            rss_before_bytes=int(before["rss_bytes"]),
            rss_after_bytes=int(after["rss_bytes"]),
            python_peak_allocated_bytes=int(peak),
        )

