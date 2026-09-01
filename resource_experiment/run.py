from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from .streaming_agent import ContextMode, InstrumentedContextAgent
from .telemetry import EventLog, TaskResourceMonitor, utc_now
from .token_counter import KimiTokenCounter


ROOT = Path(__file__).resolve().parents[1]
CONTEXT_DIR = ROOT / "chapter1" / "context"
if str(CONTEXT_DIR) not in sys.path:
    sys.path.insert(0, str(CONTEXT_DIR))

from agent import HIDDEN_RESULT_STYLES  # noqa: E402
from run_experiment_1_1 import CANONICAL_TASK, summarize_arm  # noqa: E402


MODE_ORDER = [
    ContextMode.FULL,
    ContextMode.NO_HISTORY,
    ContextMode.NO_REASONING,
    ContextMode.NO_TOOL_CALLS,
    ContextMode.NO_TOOL_RESULTS,
]


def load_config(path: Path | None = None) -> dict[str, Any]:
    source = path or ROOT / "resource_experiment" / "config.json"
    return json.loads(source.read_text(encoding="utf-8"))


def resolve_api_key() -> str:
    value = os.getenv("MOONSHOT_API_KEY")
    if value:
        return value
    if os.name == "nt":
        import winreg

        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as key:
                value, _ = winreg.QueryValueEx(key, "MOONSHOT_API_KEY")
        except OSError:
            value = None
        if value:
            os.environ["MOONSHOT_API_KEY"] = str(value)
            return str(value)
    raise RuntimeError("Windows 用户级环境变量 MOONSHOT_API_KEY 不存在。")


def latin_order(block_index: int) -> list[ContextMode]:
    offset = block_index % len(MODE_ORDER)
    return MODE_ORDER[offset:] + MODE_ORDER[:offset]


def usage_fields(usage: dict[str, Any]) -> dict[str, int]:
    prompt = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
    completion = int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
    prompt_details = usage.get("prompt_tokens_details") or usage.get("input_tokens_details") or {}
    completion_details = usage.get("completion_tokens_details") or usage.get("output_tokens_details") or {}
    cached = int(prompt_details.get("cached_tokens") or usage.get("cached_tokens") or 0)
    reasoning = int(completion_details.get("reasoning_tokens") or usage.get("reasoning_tokens") or 0)
    return {
        "prompt_tokens": prompt,
        "cached_prompt_tokens": min(cached, prompt),
        "completion_tokens": completion,
        "reasoning_tokens": min(reasoning, completion),
        "provider_total_tokens": int(usage.get("total_tokens") or prompt + completion),
    }


def aggregate_metrics(
    agent: InstrumentedContextAgent,
    arm: dict[str, Any],
    resources: dict[str, Any],
    total_wall: float,
    verification_wall: float,
    prices: dict[str, float],
) -> dict[str, Any]:
    totals = {
        "prompt_tokens": 0,
        "cached_prompt_tokens": 0,
        "completion_tokens": 0,
        "reasoning_tokens": 0,
        "provider_total_tokens": 0,
    }
    for event in agent.model_events:
        fields = usage_fields(event.get("usage") or {})
        for key, value in fields.items():
            totals[key] += value
    uncached = max(0, totals["prompt_tokens"] - totals["cached_prompt_tokens"])
    cost = (
        totals["cached_prompt_tokens"] * prices["cached_input"]
        + uncached * prices["uncached_input"]
        + totals["completion_tokens"] * prices["output"]
    ) / 1_000_000
    api_turns = arm["api_turns"]
    context_by_part = {
        key: sum(int(turn.get("context_tokens", {}).get(key, 0)) for turn in api_turns)
        for key in ("system", "tool_definitions", "user", "assistant", "tool_results", "local_total")
    }
    model_wall = sum(float(event["wall_seconds"]) for event in agent.model_events)
    tool_wall = sum(float(event["wall_seconds"]) for event in agent.tool_events)
    repeated_tool_wall = sum(
        float(event["wall_seconds"]) for event in agent.tool_events if event.get("repeated")
    )
    first_tokens = [
        float(event["time_to_first_token_seconds"])
        for event in agent.model_events
        if event.get("time_to_first_token_seconds") is not None
    ]
    return {
        **totals,
        "uncached_prompt_tokens": uncached,
        "local_context_tokens": context_by_part,
        "local_provider_token_difference": totals["prompt_tokens"] - context_by_part["local_total"],
        "model_call_count": len(agent.model_events),
        "model_wall_seconds": model_wall,
        "first_token_seconds_mean": sum(first_tokens) / len(first_tokens) if first_tokens else None,
        "tool_wall_seconds": tool_wall,
        "repeated_tool_wall_seconds": repeated_tool_wall,
        "verification_wall_seconds": verification_wall,
        "framework_wall_seconds": max(0.0, total_wall - model_wall - tool_wall - verification_wall),
        "task_wall_seconds": total_wall,
        "retry_count": sum(int(event.get("retry_count", 0)) for event in agent.model_events),
        "request_bytes": sum(int(event.get("request_bytes", 0)) for event in agent.model_events),
        "response_bytes": sum(int(event.get("response_bytes", 0)) for event in agent.model_events),
        "tool_action_count": len(agent.tool_events),
        "repeated_tool_action_count": sum(bool(event.get("repeated")) for event in agent.tool_events),
        "cost_cny": cost,
        "success_cost_cny": cost if arm["task_success"] else None,
        **resources,
    }


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def run_one(
    config: dict[str, Any],
    event_log: EventLog,
    token_counter: KimiTokenCounter,
    api_key: str,
    results_dir: Path,
    phase: str,
    block: int,
    position: int,
    mode: ContextMode,
) -> dict[str, Any]:
    run_id = f"{phase}-b{block + 1:02d}-p{position + 1}-{mode.value}"
    context = {"run_id": run_id, "phase": phase, "block": block + 1, "position": position + 1}
    event_log.emit("task_start", **context, mode=mode.value, task=CANONICAL_TASK)
    monitor = TaskResourceMonitor(float(config["resource_sample_interval_seconds"]))
    monitor.start()
    started_at = utc_now()
    started = time.monotonic()
    agent = InstrumentedContextAgent(
        api_key=api_key,
        context_mode=mode,
        event_log=event_log,
        token_counter=token_counter,
        run_context=context,
        model=config["model"],
        hidden_result_content=HIDDEN_RESULT_STYLES[config["hidden_result_style"]],
    )
    try:
        result = agent.execute_task(CANONICAL_TASK, max_iterations=int(config["max_iterations"]))
        verification_started = time.monotonic()
        event_log.emit("verification_start", **context, mode=mode.value)
        arm = summarize_arm(
            mode,
            result,
            time.monotonic() - started,
            CANONICAL_TASK,
            HIDDEN_RESULT_STYLES[config["hidden_result_style"]],
        )
        verification_wall = time.monotonic() - verification_started
        event_log.emit(
            "verification_end",
            **context,
            mode=mode.value,
            wall_seconds=verification_wall,
            task_success=arm["task_success"],
            groundedness=arm["groundedness"],
            context_contract=arm["context_contract"],
        )
    finally:
        total_wall = time.monotonic() - started
        resources = monitor.stop()
        agent.close()
    metrics = aggregate_metrics(
        agent,
        arm,
        resources,
        total_wall,
        verification_wall,
        config["price_cny_per_million_tokens"],
    )
    record = {
        "schema_version": config["schema_version"],
        "experiment_id": config["experiment_id"],
        **context,
        "mode": mode.value,
        "started_at": started_at,
        "ended_at": utc_now(),
        "task": CANONICAL_TASK,
        "settings": {
            "provider": config["provider"],
            "model": config["model"],
            "max_iterations": config["max_iterations"],
            "hidden_result_style": config["hidden_result_style"],
            "serial_execution": True,
        },
        "quality": {
            "completed": arm["completed"],
            "task_success": arm["task_success"],
            "groundedness": arm["groundedness"],
            "outcome": arm["outcome"],
            "termination_reason": "terminal_response" if arm["completed"] else ("error" if arm["error"] else "iteration_limit"),
            "context_contract": arm["context_contract"],
            "final_answer": arm["final_answer"],
        },
        "metrics": metrics,
        "model_calls": agent.model_events,
        "tool_calls": agent.tool_events,
        "api_turns": arm["api_turns"],
        "reasoning_steps": arm["reasoning_steps"],
        "error": arm["error"],
    }
    write_json(results_dir / "runs" / f"{run_id}.json", record)
    valid_response = not arm["error"] and bool(agent.model_events)
    event_log.emit(
        "task_end" if valid_response else "task_attempt_failed",
        **context,
        mode=mode.value,
        completed=arm["completed"],
        task_success=arm["task_success"],
        outcome=arm["outcome"],
        termination_reason=record["quality"]["termination_reason"],
        metrics=metrics,
        error=arm["error"],
    )
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("pilot", "main"), required=True)
    parser.add_argument("--repetitions", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--results-dir", type=Path)
    args = parser.parse_args()
    config = load_config()
    repetitions = args.repetitions or (1 if args.phase == "pilot" else int(config["repetitions"]))
    results_dir = args.results_dir or ROOT / "resource_experiment" / "results" / config["experiment_id"]
    results_dir.mkdir(parents=True, exist_ok=True)
    event_log = EventLog(results_dir / "events.jsonl", config["experiment_id"])
    completed = event_log.completed_run_ids() if args.resume else set()
    api_key = resolve_api_key()
    token_counter = KimiTokenCounter(ROOT / "resource_experiment" / ".cache" / "tokenizer")
    scheduled = [
        (block, position, mode)
        for block in range(repetitions)
        for position, mode in enumerate(latin_order(block))
    ]
    event_log.emit(
        "experiment_start",
        phase=args.phase,
        repetitions=repetitions,
        scheduled_runs=len(scheduled),
        model=config["model"],
    )
    for index, (block, position, mode) in enumerate(scheduled, 1):
        run_id = f"{args.phase}-b{block + 1:02d}-p{position + 1}-{mode.value}"
        if run_id in completed:
            print(f"[{index}/{len(scheduled)}] resume skip {run_id}", flush=True)
            continue
        attempt = 0
        while True:
            print(f"[{index}/{len(scheduled)}] start {run_id} attempt={attempt + 1}", flush=True)
            record = run_one(config, event_log, token_counter, api_key, results_dir, args.phase, block, position, mode)
            valid_response = not record["error"] and bool(record["model_calls"])
            if valid_response:
                print(
                    f"[{index}/{len(scheduled)}] end {run_id} success={record['quality']['task_success']} "
                    f"wall={record['metrics']['task_wall_seconds']:.2f}s cost={record['metrics']['cost_cny']:.6f} CNY",
                    flush=True,
                )
                break
            backoffs = config["failed_task_backoff_seconds"]
            delay = int(backoffs[min(attempt, len(backoffs) - 1)])
            print(
                f"[{index}/{len(scheduled)}] retry {run_id} after={delay}s error={record['error']}",
                flush=True,
            )
            attempt += 1
            time.sleep(delay)
    event_log.emit("experiment_end", phase=args.phase, repetitions=repetitions)


if __name__ == "__main__":
    main()
