from __future__ import annotations

import argparse
import json
from pathlib import Path

from .run import ROOT, load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path)
    args = parser.parse_args()
    config = load_config()
    results = (args.results_dir or ROOT / "resource_experiment" / "results" / config["experiment_id"]).resolve()
    expected_root = (ROOT / "resource_experiment" / "results").resolve()
    if expected_root not in results.parents:
        raise RuntimeError("修复目标不在实验结果目录内。")
    event_path = results / "events.jsonl"
    events = [json.loads(line) for line in event_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    starts = {
        event["run_id"]
        for event in events
        if event.get("phase") == "main" and event.get("event_type") == "task_start"
    }
    ends = {
        event["run_id"]
        for event in events
        if event.get("phase") == "main" and event.get("event_type") == "task_end"
    }
    invalid = starts - ends
    for run_id in ends:
        path = results / "runs" / f"{run_id}.json"
        if not path.exists():
            invalid.add(run_id)
            continue
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("error") or not record.get("model_calls"):
            invalid.add(run_id)
    if not invalid:
        print("没有需要补做的运行。")
        return
    removed = [event for event in events if event.get("run_id") in invalid]
    kept = [event for event in events if event.get("run_id") not in invalid]
    archive = results / "recovery_attempts.jsonl"
    with archive.open("a", encoding="utf-8", newline="\n") as handle:
        for event in removed:
            handle.write(json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n")
    event_path.write_text(
        "".join(json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n" for event in kept),
        encoding="utf-8",
    )
    for run_id in invalid:
        path = (results / "runs" / f"{run_id}.json").resolve()
        if path.parent != (results / "runs").resolve():
            raise RuntimeError("任务文件路径越界。")
        if path.exists():
            path.unlink()
    print(json.dumps({"rerun_ids": sorted(invalid), "archived_events": len(removed)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
