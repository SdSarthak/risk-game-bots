import random
from agents.base import BaseAgent
from engine.rules import Action
from engine.state import GameState


class RandomAgent(BaseAgent):
    """Selects uniformly at random from legal actions."""

    def __init__(self, player_id: int, seed: int | None = None):
        super().__init__(player_id)
        self._rng = random.Random(seed)

    def choose_action(self, state: GameState, legal_actions: list[Action]) -> Action:
        return self._rng.choice(legal_actions)
