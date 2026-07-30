"""
Agent backed by a trained PyTorch actor-critic.

Inference mirrors training exactly: the same observation vector, the same fixed
action layout and the same legality mask, so a policy behaves in a real game the
way it did in the environment it learned in.
"""
from __future__ import annotations

import numpy as np
import torch

from agents.action_space import ACTION_SPACE_SIZE, RiskActionSpace
from agents.base import BaseAgent
from agents.neural_net import DEVICE, RiskActorCritic
from engine.board import BoardConfig
from engine.rules import Action
from engine.state import GameState


class RLAgent(BaseAgent):
    """Wraps a trained RiskActorCritic model for inference during game play."""

    def __init__(self, player_id: int, model: RiskActorCritic,
                 deterministic: bool = True) -> None:
        super().__init__(player_id)
        if model.action_size != ACTION_SPACE_SIZE:
            raise ValueError(
                f"Checkpoint has a {model.action_size}-way policy head but the "
                f"action space is {ACTION_SPACE_SIZE} wide. It was trained "
                "against a different encoding — retrain with "
                "training/run_training.py."
            )
        self._model = model
        self._model.eval()
        self._deterministic = deterministic
        self._encoder: RiskActionSpace | None = None
        self._encoder_board: BoardConfig | None = None

    @classmethod
    def load(cls, player_id: int, checkpoint_path: str,
             deterministic: bool = True) -> "RLAgent":
        """Load from a .pt checkpoint saved by PPOTrainer."""
        from training.ppo_trainer import PPOTrainer
        model = PPOTrainer.load_model(checkpoint_path)
        return cls(player_id, model, deterministic=deterministic)

    def _encoder_for(self, board: BoardConfig) -> RiskActionSpace:
        # Games are played on one board at a time; rebuild only when it changes
        if self._encoder is None or self._encoder_board is not board:
            self._encoder = RiskActionSpace(board)
            self._encoder_board = board
        return self._encoder

    def choose_action(self, state: GameState, legal_actions: list[Action]) -> Action:
        encoder = self._encoder_for(state.board)
        mask = encoder.legal_mask(state)

        if not mask.any():
            # Nothing the encoder can express: defer to the engine's own list
            return legal_actions[0]

        obs_t = torch.as_tensor(state.to_observation(),
                                dtype=torch.float32).unsqueeze(0).to(DEVICE)
        mask_t = torch.as_tensor(mask, dtype=torch.float32).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            action_idx_t, _, _ = self._model.act(obs_t, mask_t,
                                                 deterministic=self._deterministic)

        action = encoder.decode(int(action_idx_t.item()), state)
        if action is not None:
            return action

        # The mask should make this unreachable; take the first legal slot anyway
        for index in np.flatnonzero(mask):
            fallback = encoder.decode(int(index), state)
            if fallback is not None:
                return fallback
        return legal_actions[0]
