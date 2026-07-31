import random
from agents.base import BaseAgent
from engine.rules import Action
from engine.state import GameState


class RandomAgent(BaseAgent):
    """Selects uniformly at random from legal actions."""

    def __init__(self, player_id: int, seed: int | None = None):
        super().__init__(player_id)
        self._seed = seed
        self._rng = random.Random(seed)

    def reset(self, seed: int | None = None) -> None:
        """Re-seed for a new game so a driver's --seed reproduces the match."""
        if seed is not None:
            # Offset by the seat so two random agents in one game do not play
            # the identical action stream
            self._seed = seed + self.player_id
        self._rng = random.Random(self._seed)

    def choose_action(self, state: GameState, legal_actions: list[Action]) -> Action:
        if not legal_actions:
            raise ValueError("RandomAgent was asked to move with no legal actions")
        return self._rng.choice(legal_actions)
