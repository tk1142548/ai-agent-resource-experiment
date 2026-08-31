from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CONTEXT_DIR = ROOT / "chapter1" / "context"
for path in (ROOT, CONTEXT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
