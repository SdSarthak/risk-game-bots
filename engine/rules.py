from __future__ import annotations

import random
from dataclasses import dataclass

from engine.board import BoardConfig
from engine.cards import CardDeck
from engine.constants import (
    MAX_ATTACK_DICE,
    MAX_DEFEND_DICE,
    MIN_TROOPS,
    MIN_TROOPS_TO_ATTACK,
    Phase,
)
from engine.state import GameState


@dataclass(frozen=True)
class Action:
    """Unified action dataclass for all game phases."""
    phase: Phase
    src: int = -1       # source territory (-1 if unused)
    dst: int = -1       # destination territory (-1 if unused)
    troops: int = 0     # troop count involved
    # For DRAFT: dst=territory to reinforce, troops=number to place
    # For ATTACK: src=attacker, dst=defender, troops=number of attacking dice (1-3)
    # For FORTIFY: src=from, dst=to, troops=number to move
    # Special: troops=-1 means "end phase" (pass)

    def is_end_phase(self) -> bool:
        return self.troops == -1

    def __repr__(self) -> str:
        if self.is_end_phase():
            return f"Action(END_{self.phase.name})"
        return f"Action({self.phase.name}, src={self.src}, dst={self.dst}, troops={self.troops})"


class RulesEngine:
    def __init__(self, board: BoardConfig, num_players: int, seed: int | None = None):
        self.board = board
        self.num_players = num_players
        self._rng = random.Random(seed)
        self.deck = CardDeck(board.num_territories, seed=seed)

    # ------------------------------------------------------------------
    # Legal action generation
    # ------------------------------------------------------------------

    def legal_actions(self, state: GameState) -> list[Action]:
        if state.phase == Phase.DRAFT:
            return self._legal_draft_actions(state)
        elif state.phase == Phase.ATTACK:
            return self._legal_attack_actions(state)
        elif state.phase == Phase.FORTIFY:
            return self._legal_fortify_actions(state)
        return []

    def _legal_draft_actions(self, state: GameState) -> list[Action]:
        actions = []
        player = state.current_player
        my_territories = state.territories_of(player)
        remaining = state.troops_to_place
        if remaining <= 0:
            # Must move to attack phase
            return [Action(phase=Phase.DRAFT, troops=-1)]
        # Place 1..remaining troops on any owned territory
        for t in my_territories:
            for n in range(1, remaining + 1):
                actions.append(Action(phase=Phase.DRAFT, dst=t, troops=n))
        return actions

    def _legal_attack_actions(self, state: GameState) -> list[Action]:
        actions = [Action(phase=Phase.ATTACK, troops=-1)]  # always can end attack
        player = state.current_player
        for src in state.territories_of(player):
            if state.troops[src] < MIN_TROOPS_TO_ATTACK:
                continue
            for dst in self.board.adjacent_to(src):
                if state.owners[dst] != player:
                    max_dice = min(state.troops[src] - 1, MAX_ATTACK_DICE)
                    for dice in range(1, max_dice + 1):
                        actions.append(Action(phase=Phase.ATTACK, src=src, dst=dst, troops=dice))
        return actions

    def _legal_fortify_actions(self, state: GameState) -> list[Action]:
        actions = [Action(phase=Phase.FORTIFY, troops=-1)]  # always can end/skip
        player = state.current_player
        my_territories = state.territories_of(player)
        my_set = set(my_territories)
        for src in my_territories:
            movable = state.troops[src] - MIN_TROOPS
            if movable <= 0:
                continue
            # Can fortify to any reachable friendly territory
            reachable = self._reachable_friendly(state, src, player, my_set)
            for dst in reachable:
                if dst == src:
                    continue
                for n in range(1, movable + 1):
                    actions.append(Action(phase=Phase.FORTIFY, src=src, dst=dst, troops=n))
        return actions

    def _reachable_friendly(self, state: GameState, start: int, player: int,
                             my_set: set[int]) -> set[int]:
        """BFS to find all friendly territories reachable from start."""
        visited = {start}
        queue = [start]
        while queue:
            cur = queue.pop()
            for nb in self.board.adjacent_to(cur):
                if nb not in visited and nb in my_set:
                    visited.add(nb)
                    queue.append(nb)
        return visited

    # ------------------------------------------------------------------
    # Apply action
    # ------------------------------------------------------------------

    def apply_action(self, state: GameState, action: Action) -> GameState:
        """Returns a new GameState after applying action. Does NOT mutate state."""
        s = state.copy()
        if action.phase == Phase.DRAFT:
            self._apply_draft(s, action)
        elif action.phase == Phase.ATTACK:
            self._apply_attack(s, action)
        elif action.phase == Phase.FORTIFY:
            self._apply_fortify(s, action)
        return s

    def _apply_draft(self, s: GameState, action: Action) -> None:
        if action.is_end_phase():
            s.phase = Phase.ATTACK
            return
        s.troops[action.dst] += action.troops
        s.troops_to_place -= action.troops

    def _apply_attack(self, s: GameState, action: Action) -> None:
        if action.is_end_phase():
            if s.conquered_this_turn:
                card = self.deck.draw()
                s.cards[s.current_player].append(card)
                s.conquered_this_turn = False
            s.phase = Phase.FORTIFY
            return

        src, dst, atk_dice = action.src, action.dst, action.troops
        atk_losses, def_losses = self.simulate_attack(atk_dice,
                                                       min(s.troops[dst], MAX_DEFEND_DICE))
        s.troops[src] -= atk_losses
        s.troops[dst] -= def_losses

        if s.troops[dst] <= 0:
            # Attacker captures territory
            old_owner = s.owners[dst]
            s.owners[dst] = s.current_player
            s.troops[dst] = atk_dice  # move attacking dice count in
            s.troops[src] -= atk_dice
            s.conquered_this_turn = True

            # Check if defender is eliminated
            if not s.territories_of(old_owner):
                s.eliminated[old_owner] = True
                # Attacker gets all defender's cards
                s.cards[s.current_player].extend(s.cards[old_owner])
                s.cards[old_owner] = []

    def _apply_fortify(self, s: GameState, action: Action) -> None:
        if action.is_end_phase():
            self._advance_turn(s)
            return
        s.troops[action.src] -= action.troops
        s.troops[action.dst] += action.troops
        self._advance_turn(s)  # Only one fortify move allowed per turn

    def _advance_turn(self, s: GameState) -> None:
        active = s.active_players()
        if len(active) <= 1:
            return  # game over, don't cycle
        idx = active.index(s.current_player) if s.current_player in active else 0
        next_idx = (idx + 1) % len(active)
        s.current_player = active[next_idx]
        s.phase = Phase.DRAFT
        s.conquered_this_turn = False
        s.troops_to_place = self._calculate_reinforcements(s, s.current_player)

    def _calculate_reinforcements(self, s: GameState, player: int) -> int:
        territory_bonus = max(3, len(s.territories_of(player)) // 3)
        continent_bonus = sum(
            bonus
            for continent, bonus in self.board.continent_bonuses.items()
            if s.controls_continent(player, continent)
        )
        return territory_bonus + continent_bonus

    # ------------------------------------------------------------------
    # Dice simulation
    # ------------------------------------------------------------------

    def simulate_attack(self, num_attackers: int, num_defenders: int) -> tuple[int, int]:
        """
        Roll dice and return (attacker_losses, defender_losses).
        Attacker rolls num_attackers dice (max 3).
        Defender rolls num_defenders dice (max 2).
        Compare highest dice pairs.
        """
        atk_rolls = sorted([self._rng.randint(1, 6) for _ in range(num_attackers)], reverse=True)
        def_rolls = sorted([self._rng.randint(1, 6) for _ in range(num_defenders)], reverse=True)

        atk_losses = 0
        def_losses = 0
        for a, d in zip(atk_rolls, def_rolls):
            if a > d:
                def_losses += 1
            else:
                atk_losses += 1
        return atk_losses, def_losses

    # ------------------------------------------------------------------
    # Terminal checks
    # ------------------------------------------------------------------

    def is_terminal(self, state: GameState) -> bool:
        return len(state.active_players()) <= 1

    def winner(self, state: GameState) -> int | None:
        active = state.active_players()
        if len(active) == 1:
            return active[0]
        return None

    # ------------------------------------------------------------------
    # Card trade (call before calculating draft reinforcements)
    # ------------------------------------------------------------------

    def trade_cards(self, state: GameState, card_indices: list[int]) -> GameState:
        """Trade 3 cards for reinforcement bonus. Returns new state."""
        s = state.copy()
        player = s.current_player
        traded = [s.cards[player][i] for i in sorted(card_indices, reverse=True)]
        for i in sorted(card_indices, reverse=True):
            s.cards[player].pop(i)
        self.deck.discard(traded)
        bonus = CardDeck.bonus_for_trade(s.card_trade_count)
        s.card_trade_count += 1
        s.troops_to_place += bonus
        return s
