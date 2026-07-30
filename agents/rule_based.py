"""
Rule-based heuristic agent for Risk.

Strategy:
  DRAFT   — Reinforce the border territory with the lowest troops-to-enemy ratio.
  ATTACK  — Attack if own troops / enemy troops >= attack_threshold.
             Prefer targets that complete a continent.
             End attack when no favorable attack exists.
  FORTIFY — Move troops from interior (no enemy neighbors) to the weakest border.
"""
from agents.base import BaseAgent
from engine.constants import Phase
from engine.rules import Action
from engine.state import GameState


class RuleBasedAgent(BaseAgent):
    def __init__(self, player_id: int, attack_threshold: float = 1.5):
        super().__init__(player_id)
        self.attack_threshold = attack_threshold

    def choose_action(self, state: GameState, legal_actions: list[Action]) -> Action:
        if state.phase == Phase.DRAFT:
            return self._draft(state, legal_actions)
        elif state.phase == Phase.ATTACK:
            return self._attack(state, legal_actions)
        elif state.phase == Phase.FORTIFY:
            return self._fortify(state, legal_actions)
        return legal_actions[0]

    # ------------------------------------------------------------------
    # Draft
    # ------------------------------------------------------------------

    def _draft(self, state: GameState, legal_actions: list[Action]) -> Action:
        end_actions = [a for a in legal_actions if a.is_end_phase()]
        place_actions = [a for a in legal_actions if not a.is_end_phase()]
        if not place_actions:
            return end_actions[0]

        # Prefer reinforcing a border territory (one adjacent to an enemy)
        player = state.current_player
        border_territories = self._border_territories(state, player)

        # Score each target: lower troop count on border = higher priority
        def score(dst: int) -> float:
            if dst in border_territories:
                # Ratio of enemy adjacent troops to own troops (higher = more threatened)
                enemy_adj = sum(
                    state.troops[nb]
                    for nb in state.board.adjacent_to(dst)
                    if state.owners[nb] != player
                )
                return enemy_adj / max(1, state.troops[dst])
            return 0.0  # interior territory, low priority

        # Find the max-troops action targeting the best border territory
        scored_targets = {}
        for a in place_actions:
            s = score(a.dst)
            if a.dst not in scored_targets or s > scored_targets[a.dst]:
                scored_targets[a.dst] = s

        if scored_targets:
            best_dst = max(scored_targets, key=lambda d: scored_targets[d])
            # Place all remaining troops there at once
            max_action = max(
                (a for a in place_actions if a.dst == best_dst),
                key=lambda a: a.troops,
            )
            return max_action

        return max(place_actions, key=lambda a: a.troops)

    # ------------------------------------------------------------------
    # Attack
    # ------------------------------------------------------------------

    def _attack(self, state: GameState, legal_actions: list[Action]) -> Action:
        attack_actions = [a for a in legal_actions if not a.is_end_phase()]
        end_action = next(a for a in legal_actions if a.is_end_phase())

        if not attack_actions:
            return end_action

        player = state.current_player
        best = None
        best_score = -1.0

        for a in attack_actions:
            ratio = state.troops[a.src] / max(1, state.troops[a.dst])
            if ratio < self.attack_threshold:
                continue

            score = ratio
            # Bonus: does capturing this territory complete a continent?
            continent = state.board.continent_of(a.dst)
            members = state.board.territories_in_continent(continent)
            owned = sum(1 for t in members if state.owners[t] == player)
            if owned == len(members) - 1:
                score += 10.0  # Almost have the continent

            if score > best_score:
                best_score = score
                best = a

        if best is None:
            return end_action

        # Always send max dice (min of available-1 and 3)
        max_dice = min(state.troops[best.src] - 1, 3)
        return Action(phase=Phase.ATTACK, src=best.src, dst=best.dst, troops=max_dice)

    # ------------------------------------------------------------------
    # Fortify
    # ------------------------------------------------------------------

    def _fortify(self, state: GameState, legal_actions: list[Action]) -> Action:
        move_actions = [a for a in legal_actions if not a.is_end_phase()]
        end_action = next(a for a in legal_actions if a.is_end_phase())

        if not move_actions:
            return end_action

        player = state.current_player
        border_territories = self._border_territories(state, player)

        # Find interior territories (no enemy neighbors) with excess troops
        # and the weakest border territory as destination
        if not border_territories:
            return end_action

        weakest_border = min(border_territories, key=lambda t: state.troops[t])

        # Find move actions that go to the weakest border from an interior
        interior_to_border = [
            a for a in move_actions
            if a.dst == weakest_border
            and a.src not in border_territories
        ]

        if interior_to_border:
            # Move as many as possible
            best = max(interior_to_border, key=lambda a: a.troops)
            return best

        # Fall back: any move toward the weakest border
        to_border = [a for a in move_actions if a.dst == weakest_border]
        if to_border:
            return max(to_border, key=lambda a: a.troops)

        return end_action

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _border_territories(self, state: GameState, player: int) -> set[int]:
        """Territories owned by player that are adjacent to at least one enemy."""
        return {
            t for t in state.territories_of(player)
            if any(state.owners[nb] != player for nb in state.board.adjacent_to(t))
        }
