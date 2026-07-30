"""Shared fixtures. Every test here is deterministic and needs no downloads."""
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
CONFIGS_DIR = ROOT / "configs"

# Allow `pytest tests/...` from anywhere without installing the project
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.board import BoardConfig          # noqa: E402
from engine.rules import RulesEngine          # noqa: E402
from engine.state import GameState            # noqa: E402

ALL_CONFIG_NAMES = sorted(p.stem for p in CONFIGS_DIR.glob("*.json"))


@pytest.fixture(scope="session")
def configs_dir() -> pathlib.Path:
    return CONFIGS_DIR


@pytest.fixture
def small_board() -> BoardConfig:
    return BoardConfig.load(str(CONFIGS_DIR / "small_20.json"))


@pytest.fixture
def classic_board() -> BoardConfig:
    return BoardConfig.load(str(CONFIGS_DIR / "classic_42.json"))


@pytest.fixture(params=ALL_CONFIG_NAMES)
def any_board(request) -> BoardConfig:
    """Parametrised over every board config in configs/."""
    return BoardConfig.load(str(CONFIGS_DIR / f"{request.param}.json"))


@pytest.fixture
def small_game(small_board) -> tuple[GameState, RulesEngine]:
    state = GameState.new_game(small_board, num_players=2, seed=42)
    engine = RulesEngine(small_board, num_players=2, seed=42)
    return state, engine
