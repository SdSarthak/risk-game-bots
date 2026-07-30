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


def is_compatible(path: pathlib.Path) -> bool:
    """
    Whether a checkpoint's policy head matches the current action layout.

    Checkpoints from an older encoding load fine but index a different action
    space, so they have to be skipped rather than silently mis-played. An
    unreadable file is treated as incompatible.
    """
    try:
        import torch

        from agents.action_space import ACTION_SPACE_SIZE
    except ImportError:
        return True  # torch is absent, so nothing will load this anyway

    try:
        header = torch.load(path, map_location="cpu", weights_only=True)
        return int(header["action_size"]) == ACTION_SPACE_SIZE
    except Exception:  # noqa: BLE001 - torch raises many types for a bad file
        return False


def find_latest_checkpoint(
    directory: pathlib.Path | str | None = None,
    require_compatible: bool = True,
) -> pathlib.Path | None:
    """
    Best available checkpoint, or None if the directory holds no usable one.

    A ``best_model.pt`` written by an evaluation callback always wins; otherwise
    the checkpoint with the highest training step count is used. Checkpoints
    built for a different action space are skipped unless
    ``require_compatible`` is False.
    """
    checkpoints = list_checkpoints(directory)
    if require_compatible:
        checkpoints = [p for p in checkpoints if is_compatible(p)]
    if not checkpoints:
        return None
    for path in checkpoints:
        if path.name == "best_model.pt":
            return path
    return checkpoints[-1]
