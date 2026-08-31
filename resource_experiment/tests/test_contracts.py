from __future__ import annotations

from run_experiment_1_1 import evaluate_context_contract


def response(reasoning="thought"):
    return {"id": "resp", "choices": [{"message": {"reasoning_content": reasoning}}]}


def test_no_history_contract() -> None:
    turns = [
        {
            "request": {
                "messages": [{"role": "system"}, {"role": "user"}],
                "tools": [{"type": "function"}],
            },
            "response": response(),
        },
        {
            "request": {
                "messages": [{"role": "system"}, {"role": "user"}],
                "tools": [{"type": "function"}],
            },
            "response": response(),
        },
    ]
    assert evaluate_context_contract("no_history", turns)["passed"]


def test_no_reasoning_contract() -> None:
    turns = [
        {"request": {"messages": [{"role": "system"}, {"role": "user"}], "tools": [{}]}, "response": response()},
        {
            "request": {
                "messages": [
                    {"role": "system"},
                    {"role": "user"},
                    {"role": "assistant", "tool_calls": [{}]},
                    {"role": "tool", "content": "1"},
                ],
                "tools": [{}],
            },
            "response": response(),
        },
    ]
    assert evaluate_context_contract("no_reasoning", turns)["passed"]


def test_no_tool_results_contract() -> None:
    turns = [
        {"request": {"messages": [{"role": "system"}, {"role": "user"}], "tools": [{}]}, "response": response()},
        {
            "request": {
                "messages": [
                    {"role": "system"},
                    {"role": "user"},
                    {"role": "assistant", "tool_calls": [{}], "reasoning_content": "r"},
                    {"role": "tool", "content": ""},
                ],
                "tools": [{}],
            },
            "response": response(),
        },
    ]
    assert evaluate_context_contract("no_tool_results", turns, "")["passed"]
