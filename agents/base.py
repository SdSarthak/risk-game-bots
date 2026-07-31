from abc import ABC, abstractmethod
from engine.rules import Action
from engine.state import GameState


class BaseAgent(ABC):
    """Abstract base for all Risk agents."""

    def __init__(self, player_id: int):
        self.player_id = player_id

    @abstractmethod
    def choose_action(self, state: GameState, legal_actions: list[Action]) -> Action:
        """Select one action from the list of legal actions."""
        ...

    def reset(self, seed: int | None = None) -> None:
        """
        Called at the start of a new game.

        Agents that make their own random choices must re-seed from `seed` when
        one is given; otherwise a driver's `--seed` fixes only the deal and the
        dice, and the same command produces a different result every run.
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(player={self.player_id})"
