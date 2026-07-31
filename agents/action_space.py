"""
Fixed action encoding for the learned agent.

The engine hands out a *list* of legal actions whose length and ordering change
every step, so "action index 7" means something different from one state to the
next. A policy cannot learn that: index 7 is a draft on one turn and an attack
from an unrelated territory on the next.

This module maps the policy's output onto a layout that never moves:

    [0,  T)                     draft: commit the whole allotment to territory t
    [T,  T + T*A)               attack: from t against its k-th neighbour
    [T + T*A, T + 2*T*A)        fortify: from t to its k-th neighbour
    [T + 2*T*A]                 end the current phase

with T = MAX_TERRITORIES and A = MAX_ADJACENT, padded so the same head works
for every board. Index 61 is *always* "attack from territory 1 against its
second neighbour", whatever the board looks like right now.

The encoding covers a deliberate subset of what the rules allow:

* drafts place every remaining army on one territory rather than splitting them,
* attacks always roll the maximum dice available,
* fortifies move the whole movable garrison to an *adjacent* friendly territory,
  where the rules also permit any connected one.

Every action it emits is legal; it just cannot express every legal action. That
trade keeps the head small enough to learn from.
"""
from __future__ import annotations

import numpy as np

from engine.board import BoardConfig
from engine.constants import MAX_ATTACK_DICE, MIN_TROOPS, MIN_TROOPS_TO_ATTACK, Phase
from engine.rules import Action
from engine.state import MAX_TERRITORIES, GameState

# Widest adjacency list on any shipped board (Kamchatka and friends top out here)
MAX_ADJACENT = 10

DRAFT_OFFSET = 0
ATTACK_OFFSET = MAX_TERRITORIES
FORTIFY_OFFSET = ATTACK_OFFSET + MAX_TERRITORIES * MAX_ADJACENT
END_PHASE_INDEX = FORTIFY_OFFSET + MAX_TERRITORIES * MAX_ADJACENT
ACTION_SPACE_SIZE = END_PHASE_INDEX + 1


class RiskActionSpace:
    """Encodes and decodes actions for one board."""

    size = ACTION_SPACE_SIZE

    def __init__(self, board: BoardConfig) -> None:
        if board.num_territories > MAX_TERRITORIES:
            raise ValueError(
                f"Board '{board.name}' has {board.num_territories} territories; "
                f"the action space is padded to {MAX_TERRITORIES}"
            )
        widest = max(len(t.adjacent) for t in board.territories.values())
        if widest > MAX_ADJACENT:
            raise ValueError(
                f"Board '{board.name}' has a territory with {widest} neighbours; "
                f"the action space allows {MAX_ADJACENT}"
            )
        self.board = board

    # ------------------------------------------------------------------
    # Index arithmetic
    # ------------------------------------------------------------------

    @staticmethod
    def draft_index(territory: int) -> int:
        return DRAFT_OFFSET + territory

    @staticmethod
    def attack_index(src: int, neighbour_slot: int) -> int:
        return ATTACK_OFFSET + src * MAX_ADJACENT + neighbour_slot

    @staticmethod
    def fortify_index(src: int, neighbour_slot: int) -> int:
        return FORTIFY_OFFSET + src * MAX_ADJACENT + neighbour_slot

    # ------------------------------------------------------------------
    # Decoding
    # ------------------------------------------------------------------

    def decode(self, index: int, state: GameState) -> Action | None:
        """
        Turn a policy output into an engine action.

        Returns None when the index does not name a legal move in this state,
        which lets callers fall back rather than crash on an unmasked policy.
        """
        if not 0 <= index < ACTION_SPACE_SIZE:
            return None
        if index == END_PHASE_INDEX:
            if (state.phase is Phase.DRAFT and state.troops_to_place > 0
                    and state.territories_of(state.current_player)):
                return None  # the draft cannot be skipped while armies are placeable
            return Action(phase=state.phase, troops=-1)

        player = state.current_player

        if index < ATTACK_OFFSET:
            if state.phase is not Phase.DRAFT or state.troops_to_place <= 0:
                return None
            territory = index - DRAFT_OFFSET
            if territory >= self.board.num_territories or state.owners[territory] != player:
                return None
            return Action(phase=Phase.DRAFT, dst=territory,
                          troops=state.troops_to_place)

        if index < FORTIFY_OFFSET:
            if state.phase is not Phase.ATTACK:
                return None
            src, dst = self._resolve(index - ATTACK_OFFSET)
            if src is None or dst is None:
                return None
            if state.owners[src] != player or state.owners[dst] == player:
                return None
            if state.troops[src] < MIN_TROOPS_TO_ATTACK:
                return None
            dice = min(state.troops[src] - 1, MAX_ATTACK_DICE)
            return Action(phase=Phase.ATTACK, src=src, dst=dst, troops=dice)

        if state.phase is not Phase.FORTIFY:
            return None
        src, dst = self._resolve(index - FORTIFY_OFFSET)
        if src is None or dst is None:
            return None
        if state.owners[src] != player or state.owners[dst] != player:
            return None
        movable = state.troops[src] - MIN_TROOPS
        if movable <= 0:
            return None
        return Action(phase=Phase.FORTIFY, src=src, dst=dst, troops=movable)

    def _resolve(self, offset: int) -> tuple[int | None, int | None]:
        """Split a (src, neighbour slot) offset into concrete territory ids."""
        src, slot = divmod(offset, MAX_ADJACENT)
        if src >= self.board.num_territories:
            return None, None
        neighbours = self.board.adjacent_to(src)
        if slot >= len(neighbours):
            return None, None
        return src, neighbours[slot]

    # ------------------------------------------------------------------
    # Masking
    # ------------------------------------------------------------------

    def legal_mask(self, state: GameState) -> np.ndarray:
        """Binary mask over the fixed action space; 1 where `decode` succeeds."""
        mask = np.zeros(ACTION_SPACE_SIZE, dtype=np.int8)
        player = state.current_player

        if state.phase is Phase.DRAFT:
            owned = state.territories_of(player)
            if state.troops_to_place > 0 and owned:
                for territory in owned:
                    mask[self.draft_index(territory)] = 1
            else:
                # No allotment left, or nowhere to put it: ending is the only
                # move. Leaving the mask empty here hands the policy a row with
                # nothing legal in it.
                mask[END_PHASE_INDEX] = 1
            return mask

        mask[END_PHASE_INDEX] = 1  # attack and fortify may always be ended

        if state.phase is Phase.ATTACK:
            for src in state.territories_of(player):
                if state.troops[src] < MIN_TROOPS_TO_ATTACK:
                    continue
                for slot, dst in enumerate(self.board.adjacent_to(src)):
                    if state.owners[dst] != player:
                        mask[self.attack_index(src, slot)] = 1
        elif state.phase is Phase.FORTIFY:
            for src in state.territories_of(player):
                if state.troops[src] - MIN_TROOPS <= 0:
                    continue
                for slot, dst in enumerate(self.board.adjacent_to(src)):
                    if state.owners[dst] == player:
                        mask[self.fortify_index(src, slot)] = 1

        return mask
