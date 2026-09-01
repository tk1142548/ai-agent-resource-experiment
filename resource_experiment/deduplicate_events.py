from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from .run import ROOT, load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path)
    args = parser.parse_args()
    config = load_config()
    results = args.results_dir or ROOT / "resource_experiment" / "results" / config["experiment_id"]
    event_path = results / "events.jsonl"
    events = [json.loads(line) for line in event_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    end_indices: dict[str, list[int]] = defaultdict(list)
    for index, event in enumerate(events):
        if event.get("phase") == "main" and event.get("event_type") == "task_end":
            end_indices[str(event["run_id"])].append(index)
    superseded = 0
    for run_id, indices in end_indices.items():
        for index in indices[:-1]:
            events[index]["event_type"] = "task_attempt_superseded"
            events[index]["superseded_by_later_success"] = True
            events[index]["superseded_run_id"] = run_id
            superseded += 1
    model_starts = {
        str(event["call_id"])
        for event in events
        if event.get("phase") == "main" and event.get("event_type") == "model_call_start"
    }
    orphaned_terminals = 0
    for event in events:
        if (
            event.get("phase") == "main"
            and event.get("event_type") in ("model_call_end", "model_call_error")
            and str(event.get("call_id")) not in model_starts
        ):
            event["original_event_type"] = event["event_type"]
            event["event_type"] = "model_call_terminal_orphaned"
            event["orphan_reason"] = "matching start event was archived during interrupted-run recovery"
            orphaned_terminals += 1
    event_path.write_text(
        "".join(json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n" for event in events),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "unique_run_ids": len(end_indices),
                "superseded_task_end_events": superseded,
                "orphaned_model_terminal_events": orphaned_terminals,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
