from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from resource_experiment.run import MODE_ORDER, latin_order, usage_fields
from resource_experiment.streaming_agent import InstrumentedContextAgent
from resource_experiment.telemetry import EventLog, ToolTimer
from resource_experiment.token_counter import CATEGORIES, KimiTokenCounter


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        return list(range(max(1, len(text) // 8)))

    def apply_chat_template(self, messages, tools, tokenize, add_generation_prompt, thinking):
        return list(range(120))


class FakeObject(SimpleNamespace):
    def model_dump(self):
        def convert(value):
            if isinstance(value, FakeObject):
                return {key: convert(item) for key, item in vars(value).items()}
            if isinstance(value, list):
                return [convert(item) for item in value]
            return value

        return convert(self)


class FakeCompletions:
    def create(self, **kwargs):
        return iter(
            [
                FakeObject(
                    id="resp",
                    model="kimi-k3",
                    created=1,
                    usage=None,
                    choices=[FakeObject(finish_reason=None, delta=FakeObject(content=None, reasoning_content="plan", tool_calls=[]))],
                ),
                FakeObject(
                    id="resp",
                    model="kimi-k3",
                    created=1,
                    usage=None,
                    choices=[
                        FakeObject(
                            finish_reason="tool_calls",
                            delta=FakeObject(
                                content=None,
                                reasoning_content=None,
                                tool_calls=[
                                    FakeObject(index=0, id="tool-1", type="function", function=FakeObject(name="calculate", arguments="{\"expression\":"))
                                ],
                            ),
                        )
                    ],
                ),
                FakeObject(
                    id="resp",
                    model="kimi-k3",
                    created=1,
                    usage=FakeObject(prompt_tokens=10, completion_tokens=5, total_tokens=15),
                    choices=[
                        FakeObject(
                            finish_reason="tool_calls",
                            delta=FakeObject(
                                content=None,
                                reasoning_content=None,
                                tool_calls=[FakeObject(index=0, id=None, type=None, function=FakeObject(name=None, arguments="\"2+2\"}"))],
                            ),
                        )
                    ],
                ),
            ]
        )


class FakeClient:
    def __init__(self):
        self.chat = FakeObject(completions=FakeCompletions())


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
def test_retry_attempts_record_status_and_wait(tmp_path: Path, status: int) -> None:
    agent = InstrumentedContextAgent.__new__(InstrumentedContextAgent)
    agent.current_call_id = "call"
    agent.current_iteration = 1
    agent.transport_attempts = {"call": []}
    agent.event_log = EventLog(tmp_path / "events.jsonl", "test")
    agent.run_context = {"run_id": "run", "phase": "test", "block": 1, "position": 1}
    agent.context_mode = SimpleNamespace(value="full")
    request = httpx.Request("POST", "https://example.test/v1/chat", content=b"{}")
    agent._on_http_request(request)
    agent._on_http_response(httpx.Response(status, request=request))
    agent._on_http_request(request)
    attempts = agent.transport_attempts["call"]
    assert attempts[0]["status_code"] == status
    assert attempts[1]["wait_seconds_since_previous_headers"] is not None
    assert attempts[1]["wait_seconds_since_previous_headers"] >= 0


def test_timeout_is_recordable(tmp_path: Path) -> None:
    log = EventLog(tmp_path / "events.jsonl", "test")
    error = httpx.ReadTimeout("timed out")
    log.emit("model_call_error", call_id="call", error_class=type(error).__name__)
    event = json.loads(log.path.read_text(encoding="utf-8"))
    assert event["event_type"] == "model_call_error"
    assert event["error_class"] == "ReadTimeout"


def test_stream_reconstructs_reasoning_tool_call_and_usage(tmp_path: Path) -> None:
    agent = InstrumentedContextAgent.__new__(InstrumentedContextAgent)
    agent.current_call_id = None
    agent.current_iteration = 1
    agent.transport_attempts = {}
    agent.event_log = EventLog(tmp_path / "events.jsonl", "test")
    agent.run_context = {"run_id": "run", "phase": "test", "block": 1, "position": 1}
    agent.context_mode = SimpleNamespace(value="full")
    agent.model = "kimi-k3"
    agent.client = FakeClient()
    agent.model_events = []
    message, response, event = agent._stream_completion(
        {"model": "kimi-k3", "messages": []},
        {"model": "kimi-k3", "messages": [], "stream": True},
    )
    assert message.reasoning_content == "plan"
    assert message.tool_calls[0].function.name == "calculate"
    assert json.loads(message.tool_calls[0].function.arguments) == {"expression": "2+2"}
    assert response["usage"]["total_tokens"] == 15
    assert event["time_to_first_token_seconds"] is not None
