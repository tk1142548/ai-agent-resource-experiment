from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .run import MODE_ORDER, ROOT, load_config, resolve_api_key, usage_fields


SECRET_PATTERNS = [
    re.compile(rb"Authorization\s*:\s*Bearer\s+\S+", re.I),
    re.compile(rb"MOONSHOT_API_KEY\s*=\s*[^\s\"']+", re.I),
    re.compile(rb"sk-[A-Za-z0-9_-]{20,}"),
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    events = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise AssertionError(f"events.jsonl 第 {line_number} 行无法解析：{exc}") from exc
    return events


def assert_close(left: float, right: float, tolerance: float = 1e-9) -> None:
    if abs(left - right) > tolerance:
        raise AssertionError(f"数值不闭合：{left} != {right}")


def secret_scan(root: Path) -> None:
    key = resolve_api_key().encode("utf-8")
    command = ["git", "ls-files"]
    files = subprocess.check_output(command, cwd=root, text=True, encoding="utf-8").splitlines()
    result_root = root / "resource_experiment" / "results"
    files.extend(str(path.relative_to(root)) for path in result_root.rglob("*") if path.is_file())
    for relative in sorted(set(files)):
        path = root / relative
        if not path.is_file() or path.stat().st_size > 50_000_000:
            continue
        data = path.read_bytes()
        if key and key in data:
            raise AssertionError(f"检测到真实接口密钥：{relative}")
        if any(pattern.search(data) for pattern in SECRET_PATTERNS):
            raise AssertionError(f"检测到敏感授权字段：{relative}")


def validate(results_dir: Path, phase: str, repetitions: int, check_git: bool) -> dict[str, Any]:
    config = load_config()
    events = read_jsonl(results_dir / "events.jsonl")
    task_ends = [event for event in events if event.get("event_type") == "task_end" and event.get("phase") == phase]
    expected = repetitions * len(MODE_ORDER)
    if len(task_ends) != expected:
        raise AssertionError(f"{phase} 任务结束事件应为 {expected} 个，实际为 {len(task_ends)} 个。")
    run_ids = [str(event["run_id"]) for event in task_ends]
    if len(set(run_ids)) != expected:
        raise AssertionError("运行编号不唯一。")
    counts = Counter(event["mode"] for event in task_ends)
    for mode in (item.value for item in MODE_ORDER):
        if counts[mode] != repetitions:
            raise AssertionError(f"{mode} 应有 {repetitions} 个结束事件，实际为 {counts[mode]} 个。")
    if repetitions % 5 == 0:
        positions: dict[str, Counter[int]] = defaultdict(Counter)
        for event in task_ends:
            positions[event["mode"]][int(event["position"])] += 1
        expected_position_count = repetitions // 5
        for mode, position_counts in positions.items():
            if any(position_counts[position] != expected_position_count for position in range(1, 6)):
                raise AssertionError(f"{mode} 的顺序位置不平衡：{dict(position_counts)}")

    model_start = {event["call_id"]: event for event in events if event.get("event_type") == "model_call_start" and event.get("phase") == phase}
    model_end = {event["call_id"]: event for event in events if event.get("event_type") == "model_call_end" and event.get("phase") == phase}
    model_error = {event["call_id"]: event for event in events if event.get("event_type") == "model_call_error" and event.get("phase") == phase}
    if set(model_start) != set(model_end) | set(model_error):
        raise AssertionError("模型调用开始与结束/错误事件不闭合。")
    for call_id, event in model_end.items():
        if event.get("time_to_first_token_seconds") is None:
            raise AssertionError(f"成功模型调用缺少首片段时间：{call_id}")
        if not event.get("response", {}).get("id"):
            raise AssertionError(f"成功模型调用缺少完整响应编号：{call_id}")
        usage = event.get("usage") or {}
        if not usage or usage_fields(usage)["provider_total_tokens"] <= 0:
            raise AssertionError(f"成功模型调用缺少令牌量：{call_id}")

    tool_start = {event["tool_event_id"]: event for event in events if event.get("event_type") == "tool_call_start" and event.get("phase") == phase}
    tool_end = {event["tool_event_id"]: event for event in events if event.get("event_type") == "tool_call_end" and event.get("phase") == phase}
    if set(tool_start) != set(tool_end):
        raise AssertionError("工具调用开始与结束事件不闭合。")
    for tool_id, event in tool_end.items():
        required = ("result", "wall_seconds", "cpu_seconds", "rss_before_bytes", "rss_after_bytes")
        if any(field not in event for field in required):
            raise AssertionError(f"工具调用缺少结果或资源记录：{tool_id}")

    for event in events:
        if event.get("event_type") != "context_built" or event.get("phase") != phase:
            continue
        tokens = event["context_tokens"]
        subtotal = sum(tokens[name] for name in ("system", "tool_definitions", "user", "assistant", "tool_results"))
        if subtotal != tokens["local_total"]:
            raise AssertionError(f"上下文分项不闭合：{event['run_id']} 第 {event['iteration']} 轮")

    run_files = []
    for run_id in run_ids:
        path = results_dir / "runs" / f"{run_id}.json"
        record = json.loads(path.read_text(encoding="utf-8"))
        run_files.append(path)
        if record["run_id"] != run_id or record["mode"] not in counts:
            raise AssertionError(f"任务文件字段错误：{path.name}")
        metrics = record["metrics"]
        prices = config["price_cny_per_million_tokens"]
        recomputed = (
            metrics["cached_prompt_tokens"] * prices["cached_input"]
            + metrics["uncached_prompt_tokens"] * prices["uncached_input"]
            + metrics["completion_tokens"] * prices["output"]
        ) / 1_000_000
        assert_close(float(metrics["cost_cny"]), recomputed)
        time_parts = (
            metrics["model_wall_seconds"]
            + metrics["tool_wall_seconds"]
            + metrics["verification_wall_seconds"]
            + metrics["framework_wall_seconds"]
        )
        assert_close(float(metrics["task_wall_seconds"]), float(time_parts), tolerance=1e-6)
    derived_files = 0
    if phase == "main":
        import pandas as pd

        for stem in ("run_metrics", "config_summary", "paired_differences"):
            csv_frame = pd.read_csv(results_dir / "derived" / f"{stem}.csv")
            parquet_frame = pd.read_parquet(results_dir / "derived" / f"{stem}.parquet")
            if list(csv_frame.columns) != list(parquet_frame.columns) or len(csv_frame) != len(parquet_frame):
                raise AssertionError(f"{stem} 的 CSV 与 Parquet 结构不一致。")
            derived_files += 2
        if len(pd.read_csv(results_dir / "derived" / "run_metrics.csv")) != expected:
            raise AssertionError("任务级派生表行数不是 150。")
        if len(pd.read_csv(results_dir / "derived" / "config_summary.csv")) != len(MODE_ORDER):
            raise AssertionError("配置级派生表行数不是 5。")
        if not (results_dir / "REPORT.md").is_file():
            raise AssertionError("缺少实验报告。")
        for name in ("resource_distributions.png", "success_rate.png", "context_growth.png"):
            if not (results_dir / "figures" / name).is_file():
                raise AssertionError(f"缺少图表：{name}")
    secret_scan(ROOT)
    if check_git:
        branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip()
        if branch != "main":
            raise AssertionError(f"当前分支为 {branch}，应为 main。")
        status = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)
        if status.strip():
            raise AssertionError("工作区存在未提交文件。")
        local = subprocess.check_output(["git", "rev-parse", "main"], cwd=ROOT, text=True).strip()
        remote = subprocess.check_output(["git", "rev-parse", "origin/main"], cwd=ROOT, text=True).strip()
        if local != remote:
            raise AssertionError("main 与 origin/main 未同步。")
    return {
        "phase": phase,
        "task_end_events": len(task_ends),
        "unique_run_ids": len(set(run_ids)),
        "model_calls_completed": len(model_end),
        "model_calls_failed": len(model_error),
        "tool_calls": len(tool_end),
        "run_files": len(run_files),
        "context_events": sum(event.get("event_type") == "context_built" and event.get("phase") == phase for event in events),
        "derived_files_parsed": derived_files,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("pilot", "main"), default="main")
    parser.add_argument("--repetitions", type=int)
    parser.add_argument("--results-dir", type=Path)
    parser.add_argument("--check-git", action="store_true")
    args = parser.parse_args()
    config = load_config()
    repetitions = args.repetitions or (1 if args.phase == "pilot" else int(config["repetitions"]))
    results_dir = args.results_dir or ROOT / "resource_experiment" / "results" / config["experiment_id"]
    result = validate(results_dir, args.phase, repetitions, args.check_git)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
