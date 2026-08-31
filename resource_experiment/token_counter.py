from __future__ import annotations

import json
from pathlib import Path
from typing import Any


CATEGORIES = ("system", "tool_definitions", "user", "assistant", "tool_results")
TOKENIZER_REPOSITORY = "moonshotai/Kimi-K3"
TOKENIZER_REVISION = "a590ce090cb049c93a33dfe8c208ec652aa20503"


class KimiTokenCounter:
    def __init__(self, cache_dir: Path):
        from transformers import AutoTokenizer

        cache_dir.mkdir(parents=True, exist_ok=True)
        self.tokenizer = AutoTokenizer.from_pretrained(
            TOKENIZER_REPOSITORY,
            revision=TOKENIZER_REVISION,
            trust_remote_code=True,
            cache_dir=str(cache_dir),
        )

    def _encode_len(self, value: Any) -> int:
        text = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def count(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None) -> dict[str, int]:
        full_ids = self.tokenizer.apply_chat_template(
            messages,
            tools=tools or None,
            tokenize=True,
            add_generation_prompt=True,
            thinking=True,
        )
        full = len(full_ids)
        raw = {
            "system": self._encode_len([m for m in messages if m.get("role") == "system"]),
            "tool_definitions": self._encode_len(tools or []),
            "user": self._encode_len([m for m in messages if m.get("role") == "user"]),
            "assistant": self._encode_len([m for m in messages if m.get("role") == "assistant"]),
            "tool_results": self._encode_len([m for m in messages if m.get("role") == "tool"]),
        }
        raw_total = sum(raw.values())
        if raw_total <= full:
            counts = dict(raw)
            counts["system"] += full - raw_total
        elif raw_total:
            scaled = {name: int(full * value / raw_total) for name, value in raw.items()}
            scaled["system"] += full - sum(scaled.values())
            counts = scaled
        else:
            counts = {name: 0 for name in CATEGORIES}
            counts["system"] = full
        counts["local_total"] = full
        return counts
