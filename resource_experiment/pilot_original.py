from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from .run import MODE_ORDER, ROOT, load_config, resolve_api_key


CONTEXT_DIR = ROOT / "chapter1" / "context"
if str(CONTEXT_DIR) not in sys.path:
    sys.path.insert(0, str(CONTEXT_DIR))

from agent import HIDDEN_RESULT_STYLES, ContextAwareAgent  # noqa: E402
from run_experiment_1_1 import CANONICAL_TASK, summarize_arm  # noqa: E402


def main() -> None:
    config = load_config()
    api_key = resolve_api_key()
    output = ROOT / "resource_experiment" / "results" / config["experiment_id"] / "original_pilot"
    output.mkdir(parents=True, exist_ok=True)
    arms = []
    for index, mode in enumerate(MODE_ORDER, 1):
        print(f"[{index}/5] 原程序开始 {mode.value}", flush=True)
        agent = ContextAwareAgent(
            api_key,
            context_mode=mode,
            provider="kimi",
            model=config["model"],
            verbose=False,
            hidden_result_content=HIDDEN_RESULT_STYLES[config["hidden_result_style"]],
        )
        started = time.monotonic()
        result = agent.execute_task(CANONICAL_TASK, max_iterations=int(config["max_iterations"]))
        arm = summarize_arm(
            mode,
            result,
            time.monotonic() - started,
            CANONICAL_TASK,
            HIDDEN_RESULT_STYLES[config["hidden_result_style"]],
        )
        arms.append(arm)
        print(f"[{index}/5] 原程序完成 {mode.value} success={arm['task_success']}", flush=True)
    evidence = {
        "experiment_id": config["experiment_id"],
        "phase": "original_pilot",
        "provider": "kimi",
        "model": config["model"],
        "arms": arms,
    }
    (output / "evidence.json").write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
