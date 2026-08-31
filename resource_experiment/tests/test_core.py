from __future__ import annotations

import json
from pathlib import Path

import pytest

from resource_experiment.run import MODE_ORDER, latin_order, usage_fields
from resource_experiment.telemetry import EventLog, ToolTimer
from resource_experiment.token_counter import CATEGORIES, KimiTokenCounter


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        return list(range(max(1, len(text) // 8)))

    def apply_chat_template(self, messages, tools, tokenize, add_generation_prompt, thinking):
        return list(range(120))


def test_balanced_latin_square() -> None:
    positions = {mode.value: [0] * 5 for mode in MODE_ORDER}
    for block in range(30):
        order = latin_order(block)
        assert len(set(order)) == 5
        for position, mode in enumerate(order):
            positions[mode.value][position] += 1
    assert all(counts == [6, 6, 6, 6, 6] for counts in positions.values())


def test_context_categories_close_exactly() -> None:
    counter = KimiTokenCounter.__new__(KimiTokenCounter)
    counter.tokenizer = FakeTokenizer()
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "task"},
        {"role": "assistant", "content": "call"},
        {"role": "tool", "content": "result"},
    ]
    counts = counter.count(messages, [{"type": "function"}])
    assert sum(counts[name] for name in CATEGORIES) == counts["local_total"] == 120


def test_usage_aliases_and_details() -> None:
    result = usage_fields(
        {
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "total_tokens": 120,
            "prompt_tokens_details": {"cached_tokens": 40},
            "completion_tokens_details": {"reasoning_tokens": 12},
        }
    )
    assert result == {
        "prompt_tokens": 100,
        "cached_prompt_tokens": 40,
        "completion_tokens": 20,
        "reasoning_tokens": 12,
        "provider_total_tokens": 120,
    }


def test_event_log_parses_and_recovers_completed_runs(tmp_path: Path) -> None:
    log = EventLog(tmp_path / "events.jsonl", "test")
    log.emit("task_start", run_id="r1")
    log.emit("task_end", run_id="r1")
    lines = [json.loads(line) for line in log.path.read_text(encoding="utf-8").splitlines()]
    assert [line["event_type"] for line in lines] == ["task_start", "task_end"]
    assert log.completed_run_ids() == {"r1"}


def test_tool_exception_is_measured() -> None:
    def broken():
        raise RuntimeError("tool failure")

    result, error, measurement = ToolTimer().measure(broken)
    assert result is None
    assert isinstance(error, RuntimeError)
    assert measurement.wall_seconds >= 0
    assert measurement.cpu_seconds >= 0


@pytest.mark.parametrize("status", [429, 500])
def test_retryable_status_is_representable(status: int) -> None:
    attempt = {"attempt": 0, "status_code": status, "wait_seconds": 0.25}
    assert attempt["status_code"] in (429, 500)
    assert attempt["wait_seconds"] > 0
