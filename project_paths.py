from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable


def find_workspace_root(marker_dir: str = 'data', start_path: Path | None = None) -> Path:
    """Locate the workspace root by searching upward for a marker directory."""
    start_path = (Path(start_path) if start_path is not None else Path.cwd()).resolve()
    for parent in [start_path] + list(start_path.parents):
        if (parent / marker_dir).exists():
            return parent
    return start_path


WORKSPACE_ROOT: Path = find_workspace_root()
DATA_DIR: Path = WORKSPACE_ROOT / 'data'


def project_path(*parts: str) -> Path:
    """Return a path relative to the workspace root."""
    return WORKSPACE_ROOT.joinpath(*parts)


def data_path(*parts: str) -> Path:
    """Return a path relative to the workspace data folder."""
    return DATA_DIR.joinpath(*parts)


# Ensure imports from notebooks in subfolders can find this helper.
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))
