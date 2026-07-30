"""Pydantic request/response models for the Risk API."""
from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, Field

MIN_PLAYERS = 2
MAX_PLAYERS = 6


class PlayerConfig(BaseModel):
    type: Literal["human", "random", "rule_based", "mcts", "rl"]
    checkpoint: str | None = None  # Optional override for type="rl"


class GameCreateRequest(BaseModel):
    board_config: Literal["small_20", "classic_42", "grid_6x6"] = "grid_6x6"
    players: list[PlayerConfig] = Field(min_length=MIN_PLAYERS, max_length=MAX_PLAYERS)
    seed: int | None = None  # Fixes the deal and the dice, for reproducible games


class TerritoryState(BaseModel):
    id: int
    name: str
    continent: str
    owner: int          # player index (-1 = unclaimed)
    troops: int
    adjacent: list[int]
    row: int | None = None   # grid row (None for non-grid boards)
    col: int | None = None   # grid col (None for non-grid boards)


class GridInfo(BaseModel):
    rows: int
    cols: int


class PlayerState(BaseModel):
    id: int
    type: str
    is_human: bool
    eliminated: bool
    territory_count: int
    troop_count: int
    card_count: int


class ActionRequest(BaseModel):
    phase: str          # "DRAFT", "ATTACK", "FORTIFY"
    src: int = -1
    dst: int = -1
    troops: int = 0     # -1 ends the current phase


class ActionOption(BaseModel):
    """One legal action, in the same shape the client posts back."""
    phase: str
    src: int
    dst: int
    troops: int
    end_phase: bool


class LegalActionsResponse(BaseModel):
    game_id: str
    current_player: int
    phase: str
    actions: list[ActionOption]


class GameStateResponse(BaseModel):
    game_id: str
    status: Literal["active", "finished"]
    winner: int | None
    current_player: int
    phase: str
    troops_to_place: int
    territories: list[TerritoryState]
    players: list[PlayerState]
    turn_number: int
    grid: GridInfo | None = None  # present when board is grid-based


class GameCreateResponse(BaseModel):
    game_id: str
    board_name: str
    num_territories: int
    players: list[PlayerState]
