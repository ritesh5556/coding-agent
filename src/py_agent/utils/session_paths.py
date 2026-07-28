from __future__ import annotations

import hashlib
from datetime import datetime
from pathlib import Path
from typing import Optional


def sessions_dir_for_cwd(cwd: str) -> Path:
    resolved = str(Path(cwd).resolve())
    digest = hashlib.sha256(resolved.encode("utf-8")).hexdigest()[:16]
    directory = Path.home() / ".py-agent" / "sessions" / digest
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def new_session_path(cwd: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    return sessions_dir_for_cwd(cwd) / f"{timestamp}.jsonl"


def most_recent_session_path(cwd: str) -> Optional[Path]:
    directory = sessions_dir_for_cwd(cwd)
    candidates = list(directory.glob("*.jsonl"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)
