"""
Locating trained RL checkpoints on disk.

Checkpoints are written by :class:`training.ppo_trainer.PPOTrainer` as ``.pt``
files under ``checkpoints/``. They are deliberately not tracked in git, so every
entry point has to degrade gracefully when none are present.
"""
from __future__ import annotations

import pathlib
import re

CHECKPOINTS_DIR = pathlib.Path(__file__).resolve().parent.parent / "checkpoints"

# risk_ppo_204800.pt / risk_ppo_final_1000000.pt -> the trailing step count
_STEP_RE = re.compile(r"_(\d+)\.pt$")


def _step_count(path: pathlib.Path) -> int:
    match = _STEP_RE.search(path.name)
    return int(match.group(1)) if match else -1


def list_checkpoints(directory: pathlib.Path | str | None = None) -> list[pathlib.Path]:
    """All PyTorch checkpoints in `directory`, ordered from fewest to most steps."""
    root = pathlib.Path(directory) if directory is not None else CHECKPOINTS_DIR
    if not root.is_dir():
        return []
    return sorted(root.glob("*.pt"), key=lambda p: (_step_count(p), p.name))


def find_latest_checkpoint(
    directory: pathlib.Path | str | None = None,
) -> pathlib.Path | None:
    """
    Best available checkpoint, or None if the directory holds none.

    A ``best_model.pt`` written by an evaluation callback always wins; otherwise
    the checkpoint with the highest training step count is used.
    """
    checkpoints = list_checkpoints(directory)
    if not checkpoints:
        return None
    for path in checkpoints:
        if path.name == "best_model.pt":
            return path
    return checkpoints[-1]
