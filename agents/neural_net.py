"""
Actor-Critic neural network for Risk RL agent.

Architecture:
  - Shared MLP encoder (obs → hidden)
  - Actor head:  hidden → logits over MAX_LEGAL_ACTIONS
  - Critic head: hidden → scalar value

Action masking: illegal actions are set to -inf before softmax so the
policy never samples them during training or inference.

CUDA: moves to GPU automatically if available.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from agents.action_space import ACTION_SPACE_SIZE

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# One logit per slot in the fixed action layout (see agents/action_space.py)
MAX_LEGAL_ACTIONS = ACTION_SPACE_SIZE


class RiskActorCritic(nn.Module):
    """
    Shared-encoder actor-critic for Risk.

    obs_size   : flat observation vector length
    action_size: MAX_LEGAL_ACTIONS (fixed-size action space with masking)
    hidden_size: width of hidden layers
    num_layers : depth of shared MLP (not counting heads)
    """

    def __init__(
        self,
        obs_size: int,
        action_size: int = MAX_LEGAL_ACTIONS,
        hidden_size: int = 256,
        num_layers: int = 3,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1")
        # Kept as attributes so a checkpoint can rebuild the same architecture
        self.obs_size = obs_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Shared trunk
        layers: list[nn.Module] = [nn.Linear(obs_size, hidden_size), nn.ReLU()]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden_size, hidden_size), nn.ReLU()]
        self.trunk = nn.Sequential(*layers)

        # Actor head
        self.actor = nn.Linear(hidden_size, action_size)

        # Critic head
        self.critic = nn.Linear(hidden_size, 1)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.zeros_(m.bias)
        # Actor output: smaller init so early policy is near-uniform
        nn.init.orthogonal_(self.actor.weight, gain=0.01)

    def forward(
        self,
        obs: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> tuple[torch.distributions.Categorical, torch.Tensor]:
        """
        obs         : (B, obs_size)   float32
        action_mask : (B, action_size) binary int8/float  1=legal 0=illegal
        Returns:
            dist  : masked Categorical distribution over actions
            value : (B,) scalar value estimates
        """
        h = self.trunk(obs)
        logits = self.actor(h)                          # (B, action_size)
        mask_bool = action_mask.bool()
        # A row with nothing legal would give Categorical a row of -inf and make
        # it emit NaNs; fall back to uniform so a bad mask cannot poison a batch.
        empty_rows = ~mask_bool.any(dim=-1, keepdim=True)
        mask_bool = mask_bool | empty_rows
        logits = logits.masked_fill(~mask_bool, torch.finfo(logits.dtype).min)
        dist = torch.distributions.Categorical(logits=logits)
        value = self.critic(h).squeeze(-1)              # (B,)
        return dist, value

    @torch.no_grad()
    def act(
        self,
        obs: torch.Tensor,
        action_mask: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Single-step action selection (no grad).
        Returns: action (scalar), log_prob, value
        """
        dist, value = self(obs, action_mask)
        if deterministic:
            action = dist.probs.argmax(dim=-1)
        else:
            action = dist.sample()
        return action, dist.log_prob(action), value

    def evaluate(
        self,
        obs: torch.Tensor,
        action_mask: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Used during PPO update to re-evaluate stored transitions.
        Returns: log_probs, values, entropy
        """
        dist, value = self(obs, action_mask)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, value, entropy


def make_model(obs_size: int, **kwargs) -> RiskActorCritic:
    """Create and move model to best available device."""
    model = RiskActorCritic(obs_size=obs_size, **kwargs)
    return model.to(DEVICE)
