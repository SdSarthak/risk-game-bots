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

    def reset(self) -> None:
        """Called at the start of a new game. Override to reset internal state."""
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(player={self.player_id})"
