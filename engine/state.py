from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np

from engine.board import BoardConfig
from engine.constants import CardType, Phase

MAX_TERRITORIES = 42  # Pad observations to this length for fixed-size arrays
MAX_PLAYERS = 6
MAX_CARDS_PER_PLAYER = 5


@dataclass
class GameState:
    """Full mutable game state. Clone with copy() before speculative steps."""

    board: BoardConfig
    num_players: int

    # Core state arrays (index = territory id)
    owners: list[int]        # owner player index (-1 = unclaimed, only during setup)
    troops: list[int]        # troop count per territory

    # Turn tracking
    current_player: int = 0
    phase: Phase = Phase.DRAFT

    # Draft state
    troops_to_place: int = 0

    # Attack state
    attack_src: int = -1     # -1 means no attack in progress

    # Cards
    cards: list[list[CardType]] = field(default_factory=list)  # cards[player] = list of CardType
    card_trade_count: int = 0  # how many times cards have been traded globally

    # Eliminated players
    eliminated: list[bool] = field(default_factory=list)

    # Track territory just conquered this attack phase (for card draw)
    conquered_this_turn: bool = False

    @classmethod
    def new_game(cls, board: BoardConfig, num_players: int, seed: int | None = None) -> "GameState":
        """Initialize a new game with territories randomly distributed."""
        rng = np.random.default_rng(seed)
        n = board.num_territories
        owners = [-1] * n
        troops = [0] * n

        # Distribute territories randomly
        territory_ids = list(range(n))
        rng.shuffle(territory_ids)
        for i, tid in enumerate(territory_ids):
            owners[tid] = i % num_players
            troops[tid] = 1  # Start with 1 troop each

        state = cls(
            board=board,
            num_players=num_players,
            owners=owners,
            troops=troops,
            current_player=0,
            phase=Phase.DRAFT,
            troops_to_place=0,  # computed below after state exists
            cards=[[] for _ in range(num_players)],
            eliminated=[False] * num_players,
        )
        # Use same reinforcement formula as every other turn
        state.troops_to_place = max(3, len(state.territories_of(0)) // 3) + sum(
            bonus for continent, bonus in board.continent_bonuses.items()
            if state.controls_continent(0, continent)
        )
        return state

    @staticmethod
    def _initial_troops(num_players: int) -> int:
        """Standard Risk starting armies minus 1 per territory (territories already have 1)."""
        starting = {2: 40, 3: 35, 4: 30, 5: 25, 6: 20}
        base = starting.get(num_players, 20)
        return base  # Full draft allotment; rules engine handles subtracting placed troops

    def copy(self) -> "GameState":
        s = copy.copy(self)
        s.owners = self.owners[:]
        s.troops = self.troops[:]
        s.cards = [c[:] for c in self.cards]
        s.eliminated = self.eliminated[:]
        return s

    def to_observation(self) -> np.ndarray:
        """
        Flat float32 observation vector for RL.
        Layout (fixed MAX_TERRITORIES padding):
          [owner_norm_0..N, troops_norm_0..N, current_player_norm,
           phase_onehot(3), cards_onehot(MAX_PLAYERS * MAX_CARDS_PER_PLAYER)]
        """
        n = MAX_TERRITORIES
        obs = np.zeros(n * 2 + 1 + 3 + MAX_PLAYERS * MAX_CARDS_PER_PLAYER, dtype=np.float32)

        # Ownership normalized to [-1, 1] per player perspective (current_player = 1.0)
        for i in range(self.board.num_territories):
            if self.owners[i] == self.current_player:
                obs[i] = 1.0
            elif self.owners[i] == -1:
                obs[i] = 0.0
            else:
                obs[i] = -1.0 / max(1, self.num_players - 1)

        # Troops normalized by total troops on board
        total_troops = max(1, sum(self.troops))
        for i in range(self.board.num_territories):
            obs[n + i] = self.troops[i] / total_troops

        # Current player normalized
        obs[n * 2] = self.current_player / max(1, self.num_players - 1)

        # Phase one-hot
        phase_map = {Phase.DRAFT: 0, Phase.ATTACK: 1, Phase.FORTIFY: 2}
        obs[n * 2 + 1 + phase_map[self.phase]] = 1.0

        # Cards for current player (one-hot per slot)
        card_offset = n * 2 + 1 + 3
        card_type_map = {CardType.INFANTRY: 0, CardType.CAVALRY: 1,
                         CardType.ARTILLERY: 2, CardType.WILD: 3}
        for slot, card in enumerate(self.cards[self.current_player][:MAX_CARDS_PER_PLAYER]):
            obs[card_offset + slot] = card_type_map[card] / 3.0

        return obs

    @property
    def obs_size(self) -> int:
        return MAX_TERRITORIES * 2 + 1 + 3 + MAX_PLAYERS * MAX_CARDS_PER_PLAYER

    def territories_of(self, player: int) -> list[int]:
        return [i for i, o in enumerate(self.owners) if o == player]

    def troop_count_of(self, player: int) -> int:
        return sum(t for i, t in enumerate(self.troops) if self.owners[i] == player)

    def controls_continent(self, player: int, continent: str) -> bool:
        members = self.board.territories_in_continent(continent)
        return all(self.owners[t] == player for t in members)

    def active_players(self) -> list[int]:
        return [p for p in range(self.num_players) if not self.eliminated[p]]

    def __repr__(self) -> str:
        lines = [f"Player {self.current_player} | Phase: {self.phase.name}"]
        for pid in range(self.num_players):
            if not self.eliminated[pid]:
                terrs = len(self.territories_of(pid))
                troops = self.troop_count_of(pid)
                lines.append(f"  P{pid}: {terrs} territories, {troops} troops")
        return "\n".join(lines)
