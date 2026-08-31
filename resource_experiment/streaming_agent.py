from __future__ import annotations

import json
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
from openai import OpenAI

ROOT = Path(__file__).resolve().parents[1]
CONTEXT_DIR = ROOT / "chapter1" / "context"
if str(CONTEXT_DIR) not in sys.path:
    sys.path.insert(0, str(CONTEXT_DIR))

from agent import (  # noqa: E402
    ContextAwareAgent,
    ContextMode,
    ToolCall,
    _reasoning_safe_temperature,
)

from .telemetry import EventLog, ToolTimer, json_size
from .token_counter import KimiTokenCounter


@dataclass
class FunctionCall:
    name: str
    arguments: str

    def model_dump(self) -> dict[str, str]:
        return {"name": self.name, "arguments": self.arguments}


@dataclass
class MessageToolCall:
    id: str
    function: FunctionCall
    type: str = "function"

    def model_dump(self) -> dict[str, Any]:
        return {"id": self.id, "type": self.type, "function": self.function.model_dump()}


@dataclass
class AssembledMessage:
    role: str
    content: str | None
    reasoning_content: str | None
    tool_calls: list[MessageToolCall]

    @property
    def model_extra(self) -> dict[str, Any]:
        return {"reasoning_content": self.reasoning_content} if self.reasoning_content else {}

    def model_dump(self) -> dict[str, Any]:
        result: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.reasoning_content:
            result["reasoning_content"] = self.reasoning_content
        if self.tool_calls:
            result["tool_calls"] = [call.model_dump() for call in self.tool_calls]
        return result

    def dict(self) -> dict[str, Any]:
        return self.model_dump()


class InstrumentedContextAgent(ContextAwareAgent):
    def __init__(
        self,
        api_key: str,
        context_mode: ContextMode,
        event_log: EventLog,
        token_counter: KimiTokenCounter,
        run_context: dict[str, Any],
        model: str = "kimi-k3",
        hidden_result_content: str = "",
    ):
        super().__init__(
            api_key,
            context_mode=context_mode,
            provider="kimi",
            model=model,
            verbose=False,
            hidden_result_content=hidden_result_content,
        )
        self.event_log = event_log
        self.token_counter = token_counter
        self.run_context = dict(run_context)
        self.current_iteration = 0
        self.current_call_id: str | None = None
        self.transport_attempts: dict[str, list[dict[str, Any]]] = {}
        self.tool_events: list[dict[str, Any]] = []
        self.model_events: list[dict[str, Any]] = []
        self._seen_tool_signatures: set[str] = set()
        self._tool_index = 0
        self._http_client = httpx.Client(
            timeout=httpx.Timeout(180.0),
            event_hooks={"request": [self._on_http_request], "response": [self._on_http_response]},
        )
        self.client = OpenAI(
            api_key=api_key,
            base_url=self.base_url,
            http_client=self._http_client,
            max_retries=2,
        )

    def close(self) -> None:
        self._http_client.close()

    def _base(self) -> dict[str, Any]:
        return {**self.run_context, "mode": self.context_mode.value}

    def _on_http_request(self, request: httpx.Request) -> None:
        call_id = self.current_call_id or "unscoped"
        previous = self.transport_attempts.setdefault(call_id, [])[-1:] 
        now_ns = time.monotonic_ns()
        wait_seconds = None
        if previous and previous[0].get("headers_ns") is not None:
            wait_seconds = (now_ns - int(previous[0]["headers_ns"])) / 1e9
        attempt = {
            "attempt": len(self.transport_attempts[call_id]),
            "start_ns": now_ns,
            "method": request.method,
            "url": str(request.url),
            "request_body_bytes": len(request.content) if request.content else 0,
            "status_code": None,
            "headers_ns": None,
            "wait_seconds_since_previous_headers": wait_seconds,
        }
        self.transport_attempts[call_id].append(attempt)
        self.event_log.emit(
            "model_transport_attempt_start",
            **self._base(),
            call_id=call_id,
            iteration=self.current_iteration,
            attempt=attempt["attempt"],
            request_body_bytes=attempt["request_body_bytes"],
            wait_seconds_since_previous_headers=wait_seconds,
        )

    def _on_http_response(self, response: httpx.Response) -> None:
        call_id = self.current_call_id or "unscoped"
        attempts = self.transport_attempts.get(call_id, [])
        pending = next((item for item in reversed(attempts) if item["status_code"] is None), None)
        if pending is None:
            return
        pending["status_code"] = response.status_code
        pending["headers_ns"] = time.monotonic_ns()
        self.event_log.emit(
            "model_transport_attempt_headers",
            **self._base(),
            call_id=call_id,
            iteration=self.current_iteration,
            attempt=pending["attempt"],
            status_code=response.status_code,
            headers_seconds=(pending["headers_ns"] - pending["start_ns"]) / 1e9,
        )

    @staticmethod
    def _delta_reasoning(delta: Any) -> str:
        value = getattr(delta, "reasoning_content", None)
        if value:
            return str(value)
        extra = getattr(delta, "model_extra", None) or {}
        return str(extra.get("reasoning_content") or extra.get("reasoning") or "")

    @staticmethod
    def _meaningful_delta(delta: Any) -> bool:
        if getattr(delta, "content", None) or InstrumentedContextAgent._delta_reasoning(delta):
            return True
        for call in getattr(delta, "tool_calls", None) or []:
            function = getattr(call, "function", None)
            if getattr(call, "id", None) or getattr(function, "name", None) or getattr(function, "arguments", None):
                return True
        return False

    def _stream_completion(self, create_kwargs: dict[str, Any], request_data: dict[str, Any]) -> tuple[AssembledMessage, dict[str, Any], dict[str, Any]]:
        call_id = uuid.uuid4().hex
        self.current_call_id = call_id
        self.transport_attempts[call_id] = []
        start_ns = time.monotonic_ns()
        self.event_log.emit(
            "model_call_start",
            **self._base(),
            call_id=call_id,
            iteration=self.current_iteration,
            request=request_data,
            request_bytes=json_size(request_data),
        )
        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        tool_parts: dict[int, dict[str, str]] = {}
        response_id = ""
        response_model = self.model
        created = None
        finish_reason = None
        usage: dict[str, Any] = {}
        response_bytes = 0
        chunk_count = 0
        first_token_ns: int | None = None
        try:
            stream = self.client.chat.completions.create(
                **create_kwargs,
                stream=True,
                stream_options={"include_usage": True},
            )
            for chunk in stream:
                chunk_count += 1
                chunk_dict = chunk.model_dump() if hasattr(chunk, "model_dump") else chunk.dict()
                response_bytes += json_size(chunk_dict)
                response_id = getattr(chunk, "id", None) or response_id
                response_model = getattr(chunk, "model", None) or response_model
                created = getattr(chunk, "created", None) or created
                chunk_usage = getattr(chunk, "usage", None)
                if chunk_usage:
                    usage = chunk_usage.model_dump() if hasattr(chunk_usage, "model_dump") else chunk_usage.dict()
                choices = getattr(chunk, "choices", None) or []
                if not choices:
                    continue
                choice = choices[0]
                finish_reason = getattr(choice, "finish_reason", None) or finish_reason
                delta = choice.delta
                if first_token_ns is None and self._meaningful_delta(delta):
                    first_token_ns = time.monotonic_ns()
                    self.event_log.emit(
                        "model_first_token",
                        **self._base(),
                        call_id=call_id,
                        iteration=self.current_iteration,
                        time_to_first_token_seconds=(first_token_ns - start_ns) / 1e9,
                    )
                if getattr(delta, "content", None):
                    content_parts.append(str(delta.content))
                reasoning = self._delta_reasoning(delta)
                if reasoning:
                    reasoning_parts.append(reasoning)
                for call in getattr(delta, "tool_calls", None) or []:
                    index = int(getattr(call, "index", 0) or 0)
                    entry = tool_parts.setdefault(index, {"id": "", "type": "function", "name": "", "arguments": ""})
                    if getattr(call, "id", None):
                        entry["id"] += str(call.id)
                    if getattr(call, "type", None):
                        entry["type"] = str(call.type)
                    function = getattr(call, "function", None)
                    if function and getattr(function, "name", None):
                        entry["name"] += str(function.name)
                    if function and getattr(function, "arguments", None):
                        entry["arguments"] += str(function.arguments)
        except Exception as exc:
            end_ns = time.monotonic_ns()
            self.event_log.emit(
                "model_call_error",
                **self._base(),
                call_id=call_id,
                iteration=self.current_iteration,
                wall_seconds=(end_ns - start_ns) / 1e9,
                error_class=type(exc).__name__,
                error_message=str(exc),
                transport_attempts=self.transport_attempts.get(call_id, []),
            )
            self.current_call_id = None
            raise
        end_ns = time.monotonic_ns()
        tool_calls = [
            MessageToolCall(
                id=value["id"],
                type=value["type"],
                function=FunctionCall(value["name"], value["arguments"]),
            )
            for _, value in sorted(tool_parts.items())
        ]
        content = "".join(content_parts) or None
        reasoning_content = "".join(reasoning_parts) or None
        message = AssembledMessage("assistant", content, reasoning_content, tool_calls)
        response = {
            "id": response_id,
            "object": "chat.completion",
            "created": created,
            "model": response_model,
            "choices": [{"index": 0, "message": message.model_dump(), "finish_reason": finish_reason}],
            "usage": usage,
        }
        attempts = self.transport_attempts.get(call_id, [])
        event = {
            "call_id": call_id,
            "iteration": self.current_iteration,
            "wall_seconds": (end_ns - start_ns) / 1e9,
            "time_to_first_token_seconds": ((first_token_ns - start_ns) / 1e9) if first_token_ns else None,
            "request_bytes": json_size(request_data),
            "response_bytes": response_bytes,
            "chunk_count": chunk_count,
            "retry_count": max(0, len(attempts) - 1),
            "transport_attempts": attempts,
            "usage": usage,
        }
        self.model_events.append(event)
        self.event_log.emit(
            "model_call_end",
            **self._base(),
            **event,
            response=response,
        )
        self.current_call_id = None
        return message, response, event

    def _execute_measured_tool(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        self._tool_index += 1
        signature = json.dumps(
            [tool_name, arguments], ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
        )
        repeated = signature in self._seen_tool_signatures
        self._seen_tool_signatures.add(signature)
        tool_event_id = uuid.uuid4().hex
        self.event_log.emit(
            "tool_call_start",
            **self._base(),
            tool_event_id=tool_event_id,
            iteration=self.current_iteration,
            tool_index=self._tool_index,
            tool_name=tool_name,
            arguments=arguments,
            signature=signature,
            repeated=repeated,
            input_bytes=json_size(arguments),
        )
        timer = ToolTimer()
        result, error, measurement = timer.measure(super()._execute_tool, tool_name, arguments)
        if error is not None:
            result = {"error": str(error)}
        event = {
            "tool_event_id": tool_event_id,
            "iteration": self.current_iteration,
            "tool_index": self._tool_index,
            "tool_name": tool_name,
            "arguments": arguments,
            "signature": signature,
            "repeated": repeated,
            "success": error is None and not (isinstance(result, dict) and result.get("error")),
            "error_class": type(error).__name__ if error else None,
            "wall_seconds": measurement.wall_seconds,
            "cpu_seconds": measurement.cpu_seconds,
            "rss_before_bytes": measurement.rss_before_bytes,
            "rss_after_bytes": measurement.rss_after_bytes,
            "python_peak_allocated_bytes": measurement.python_peak_allocated_bytes,
            "input_bytes": json_size(arguments),
            "output_bytes": json_size(result),
            "result": result,
        }
        self.tool_events.append(event)
        self.event_log.emit("tool_call_end", **self._base(), **event)
        return result

    def execute_task(self, task: str, max_iterations: int = 5) -> dict[str, Any]:
        self.conversation_history.append({"role": "user", "content": task})
        messages = self.conversation_history
        iteration = 0
        final_answer = None
        error = None
        while iteration < max_iterations:
            iteration += 1
            self.current_iteration = iteration
            api_messages = self._prepare_messages_for_api()
            tools = self._get_tools_description() if self.context_mode != ContextMode.NO_TOOL_CALLS else None
            context_tokens = self.token_counter.count(api_messages, tools)
            request_data: dict[str, Any] = {
                "model": self.model,
                "messages": self._json_snapshot(api_messages),
                "temperature": _reasoning_safe_temperature(self.model, 0.3),
                "max_tokens": 8192,
                "stream": True,
                "stream_options": {"include_usage": True},
            }
            if tools is not None:
                request_data["tools"] = tools
                request_data["tool_choice"] = "auto"
            self.event_log.emit(
                "context_built",
                **self._base(),
                iteration=iteration,
                context_tokens=context_tokens,
                message_roles=[message.get("role") for message in api_messages],
            )
            create_kwargs = {
                "model": self.model,
                "messages": api_messages,
                "tools": tools,
                "tool_choice": "auto" if tools is not None else None,
                "temperature": _reasoning_safe_temperature(self.model, 0.3),
                "max_tokens": 8192,
                "timeout": 180,
            }
            try:
                message, response_dict, _ = self._stream_completion(create_kwargs, request_data)
                self.trajectory.api_turns.append(
                    {
                        "iteration": iteration,
                        "provider": self.provider,
                        "resolved_model": self.model,
                        "base_url": self.base_url,
                        "using_openrouter": False,
                        "request": self._json_snapshot(request_data),
                        "response": self._json_snapshot(response_dict),
                        "context_tokens": context_tokens,
                    }
                )
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                self.trajectory.api_turns.append(
                    {
                        "iteration": iteration,
                        "provider": self.provider,
                        "resolved_model": self.model,
                        "base_url": self.base_url,
                        "using_openrouter": False,
                        "error": {"class": type(exc).__name__, "message": str(exc)},
                        "request": self._json_snapshot(request_data),
                        "context_tokens": context_tokens,
                    }
                )
                break
            has_tool_calls = bool(message.tool_calls)
            if message.reasoning_content:
                self.trajectory.reasoning_steps.append(message.reasoning_content)
            if not has_tool_calls:
                messages.append(self._prepare_assistant_message(message))
                content = (message.content or "").strip()
                if content:
                    marked = self._extract_final_answer(content)
                    final_answer = marked if marked is not None else content
                break
            messages.append(self._prepare_assistant_message(message))
            for tool_call in message.tool_calls:
                function_name = tool_call.function.name
                raw_args = tool_call.function.arguments or "{}"
                try:
                    function_args = json.loads(raw_args)
                except json.JSONDecodeError as exc:
                    result = {"error": f"Invalid tool arguments: {exc}", "raw_arguments": raw_args[:500]}
                    function_args = {}
                else:
                    result = self._execute_measured_tool(function_name, function_args)
                self.trajectory.tool_calls.append(
                    ToolCall(tool_name=function_name, arguments=function_args, result=result)
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(result, ensure_ascii=False, default=str)
                        if self.context_mode != ContextMode.NO_TOOL_RESULTS
                        else self.hidden_result_content,
                    }
                )
            if message.content and "FINAL ANSWER:" in message.content:
                final_answer = self._extract_final_answer(message.content)
                break
        completed = bool(final_answer and str(final_answer).strip())
        return {
            "final_answer": final_answer,
            "trajectory": self.trajectory,
            "iterations": iteration,
            "completed": completed,
            "task_success": None,
            "success": completed,
            "error": error,
            **self._backend_identity(),
        }
