"""
RL Agent using the native PyTorch actor-critic model.

Replaces the old Stable Baselines3 PPO wrapper.
"""
from __future__ import annotations

import numpy as np
import torch

from agents.base import BaseAgent
from agents.neural_net import DEVICE, MAX_LEGAL_ACTIONS, RiskActorCritic
from engine.rules import Action
from engine.state import GameState


class RLAgent(BaseAgent):
    """Wraps a trained RiskActorCritic model for inference during game play."""

    def __init__(self, player_id: int, model: RiskActorCritic) -> None:
        super().__init__(player_id)
        self._model = model
        self._model.eval()

    @classmethod
    def load(cls, player_id: int, checkpoint_path: str) -> "RLAgent":
        """Load from a .pt checkpoint saved by PPOTrainer."""
        from training.ppo_trainer import PPOTrainer
        model = PPOTrainer.load_model(checkpoint_path)
        return cls(player_id, model)

    def choose_action(self, state: GameState, legal_actions: list[Action]) -> Action:
        obs = state.to_observation()                              # np.ndarray
        mask = np.zeros(MAX_LEGAL_ACTIONS, dtype=np.float32)
        for i in range(min(len(legal_actions), MAX_LEGAL_ACTIONS)):
            mask[i] = 1.0

        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        mask_t = torch.tensor(mask, dtype=torch.float32).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            action_idx_t, _, _ = self._model.act(obs_t, mask_t, deterministic=True)

        action_idx = int(action_idx_t.item())
        action_idx = min(action_idx, len(legal_actions) - 1)
        return legal_actions[action_idx]
