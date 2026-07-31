from __future__ import annotations

import random
from dataclasses import dataclass

from engine.board import BoardConfig
from engine.cards import CardDeck
from engine.constants import (
    CARDS_PER_SET,
    MAX_ATTACK_DICE,
    MAX_CARDS_IN_HAND,
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
    """
    Applies Risk rules to a :class:`GameState`.

    Card sets are traded automatically at the start of each draft phase: the
    engine cashes in every set the player holds when ``eager_card_trades`` is
    True (the default), and otherwise only cashes in the minimum needed to get
    back under the five-card hand limit. Agents therefore never have to emit
    trade actions, and the escalating trade bonus still drives the late game.
    """

    def __init__(self, board: BoardConfig, num_players: int, seed: int | None = None,
                 eager_card_trades: bool = True):
        self.board = board
        self.num_players = num_players
        self._rng = random.Random(seed)
        self.deck = CardDeck(board.num_territories, seed=seed)
        self.eager_card_trades = eager_card_trades

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
        if remaining <= 0 or not my_territories:
            # Nothing left to place, or nowhere to place it: move to attack.
            # Without the second case a player holding no ground but a pending
            # allotment gets an empty legal-action list, and every driver
            # (play_game, benchmark, the API's bot loop) reads that as a wedged
            # game and abandons it.
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
        s.turn_number += 1
        s.conquered_this_turn = False
        s.troops_to_place = self.calculate_reinforcements(s, s.current_player)
        self.auto_trade_cards(s, s.current_player)

    def calculate_reinforcements(self, s: GameState, player: int) -> int:
        """Troops a player receives at the start of their draft phase."""
        return s.reinforcements_for(player)

    # ------------------------------------------------------------------
    # Cards
    # ------------------------------------------------------------------

    def auto_trade_cards(self, s: GameState, player: int) -> int:
        """
        Cash in card sets for the player, mutating ``s`` in place.

        Trades every available set when ``eager_card_trades`` is set, otherwise
        only enough sets to bring the hand back under the five-card limit.
        Returns the total bonus troops added to ``s.troops_to_place``.
        """
        total_bonus = 0
        while True:
            hand = s.cards[player]
            forced = len(hand) > MAX_CARDS_IN_HAND
            if not forced and not self.eager_card_trades:
                break
            if len(hand) < CARDS_PER_SET:
                break
            indices = CardDeck.find_valid_set(hand)
            if indices is None:
                break
            total_bonus += self._trade_set(s, player, indices)
        return total_bonus

    def _trade_set(self, s: GameState, player: int, card_indices: list[int]) -> int:
        """Remove a set from the player's hand and grant the trade bonus."""
        hand = s.cards[player]
        if len(card_indices) != CARDS_PER_SET:
            raise ValueError(f"A trade needs exactly {CARDS_PER_SET} cards")
        if len(set(card_indices)) != CARDS_PER_SET:
            raise ValueError("Card indices must be distinct")
        if any(not 0 <= i < len(hand) for i in card_indices):
            raise ValueError(f"Card index out of range for a hand of {len(hand)}")

        traded = [hand[i] for i in card_indices]
        for i in sorted(card_indices, reverse=True):
            hand.pop(i)
        self.deck.discard(traded)

        bonus = CardDeck.bonus_for_trade(s.card_trade_count)
        s.card_trade_count += 1
        s.troops_to_place += bonus
        return bonus

    def trade_cards(self, state: GameState, card_indices: list[int]) -> GameState:
        """Trade one set of 3 cards for the current escalating bonus. Returns a new state."""
        s = state.copy()
        self._trade_set(s, s.current_player, card_indices)
        return s

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
