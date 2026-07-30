"""
Pure-PyTorch PPO trainer for Risk.

Runs N parallel (sequential) environment episodes to collect rollouts,
then performs K epochs of mini-batch gradient updates.

CUDA: tensors are moved to GPU when available (see neural_net.DEVICE).

Usage:
    python training/ppo_trainer.py --config grid_6x6 --timesteps 1000000
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Ensure project root on path when run as script
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from agents.neural_net import DEVICE, MAX_LEGAL_ACTIONS, RiskActorCritic, make_model
from training.gym_env import RiskEnv

CHECKPOINTS_DIR = pathlib.Path(__file__).parent.parent / "checkpoints"


# ─────────────────────────────────────────────────────────────────────────────
# Rollout buffer
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RolloutBuffer:
    obs: list = field(default_factory=list)
    actions: list = field(default_factory=list)
    log_probs: list = field(default_factory=list)
    values: list = field(default_factory=list)
    rewards: list = field(default_factory=list)
    dones: list = field(default_factory=list)
    masks: list = field(default_factory=list)

    def clear(self) -> None:
        self.obs.clear(); self.actions.clear(); self.log_probs.clear()
        self.values.clear(); self.rewards.clear(); self.dones.clear()
        self.masks.clear()

    def add(self, obs, action, log_prob, value, reward, done, mask) -> None:
        self.obs.append(obs)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(done)
        self.masks.append(mask)

    def __len__(self) -> int:
        return len(self.rewards)

    def compute_returns_and_advantages(
        self,
        last_value: float,
        gamma: float,
        gae_lambda: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """GAE returns and advantages."""
        n = len(self.rewards)
        advantages = np.zeros(n, dtype=np.float32)
        last_gae = 0.0
        values_np = np.array([v.item() if isinstance(v, torch.Tensor) else v
                               for v in self.values], dtype=np.float32)
        rewards_np = np.array(self.rewards, dtype=np.float32)
        dones_np = np.array(self.dones, dtype=np.float32)

        for t in reversed(range(n)):
            next_val = last_value if t == n - 1 else values_np[t + 1]
            next_done = 0.0 if t == n - 1 else dones_np[t + 1]
            delta = rewards_np[t] + gamma * next_val * (1 - next_done) - values_np[t]
            last_gae = delta + gamma * gae_lambda * (1 - next_done) * last_gae
            advantages[t] = last_gae

        returns = advantages + values_np
        return (
            torch.tensor(returns, dtype=torch.float32),
            torch.tensor(advantages, dtype=torch.float32),
        )


# ─────────────────────────────────────────────────────────────────────────────
# PPO Trainer
# ─────────────────────────────────────────────────────────────────────────────

class PPOTrainer:
    def __init__(
        self,
        env: RiskEnv,
        model: RiskActorCritic,
        # PPO hyperparameters
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.05,   # higher to prevent premature collapse on 500-action space
        max_grad_norm: float = 0.5,
        n_steps: int = 2048,          # steps per rollout
        n_epochs: int = 10,           # PPO update epochs per rollout
        batch_size: int = 64,
        total_timesteps: int = 0,     # set in train() for LR schedule
    ) -> None:
        self.env = env
        self.model = model
        self._base_lr = lr
        self._total_timesteps = total_timesteps
        self.optimizer = optim.Adam(model.parameters(), lr=lr, eps=1e-5)

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.n_steps = n_steps
        self.n_epochs = n_epochs
        self.batch_size = batch_size

        self.buffer = RolloutBuffer()
        self._obs: np.ndarray | None = None
        self._mask: np.ndarray | None = None
        self._episode_rewards: deque = deque(maxlen=100)
        self._episode_lengths: deque = deque(maxlen=100)
        self._wins: deque = deque(maxlen=100)
        self._cur_ep_reward = 0.0
        self._cur_ep_len = 0
        self.total_steps = 0

    def _reset_env(self) -> None:
        obs_dict, _ = self.env.reset()
        self._obs = obs_dict["obs"]
        self._mask = obs_dict["action_mask"]
        self._cur_ep_reward = 0.0
        self._cur_ep_len = 0

    def collect_rollout(self) -> None:
        """Collect n_steps transitions into self.buffer."""
        if self._obs is None:
            self._reset_env()

        self.buffer.clear()
        self.model.eval()

        for _ in range(self.n_steps):
            obs_t = torch.tensor(self._obs, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            mask_t = torch.tensor(self._mask, dtype=torch.float32).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                action_t, log_prob_t, value_t = self.model.act(obs_t, mask_t)

            action = int(action_t.item())
            obs_dict, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated

            self.buffer.add(
                obs=self._obs.copy(),
                action=action,
                log_prob=log_prob_t.item(),
                value=value_t.item(),
                reward=float(reward),
                done=float(done),
                mask=self._mask.copy(),
            )

            self._cur_ep_reward += float(reward)
            self._cur_ep_len += 1
            self.total_steps += 1

            if done:
                self._episode_rewards.append(self._cur_ep_reward)
                self._episode_lengths.append(self._cur_ep_len)
                # The env reports the outcome: a truncated episode can pay out
                # positively without anyone having won.
                self._wins.append(1 if info.get("is_win") else 0)
                obs_dict, _ = self.env.reset()
                self._cur_ep_reward = 0.0
                self._cur_ep_len = 0

            self._obs = obs_dict["obs"]
            self._mask = obs_dict["action_mask"]

        # Bootstrap last value for GAE
        with torch.no_grad():
            obs_t = torch.tensor(self._obs, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            mask_t = torch.tensor(self._mask, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            _, _, last_value = self.model.act(obs_t, mask_t)
        self._last_value = last_value.item()

    def update(self) -> dict[str, float]:
        """One PPO update pass over the collected rollout."""
        self.model.train()

        returns, advantages = self.buffer.compute_returns_and_advantages(
            self._last_value, self.gamma, self.gae_lambda
        )
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Stack rollout into tensors
        obs_np = np.array(self.buffer.obs, dtype=np.float32)
        masks_np = np.array(self.buffer.masks, dtype=np.float32)
        actions_np = np.array(self.buffer.actions, dtype=np.int64)
        old_log_probs_np = np.array(self.buffer.log_probs, dtype=np.float32)

        obs_all = torch.tensor(obs_np).to(DEVICE)
        masks_all = torch.tensor(masks_np).to(DEVICE)
        actions_all = torch.tensor(actions_np).to(DEVICE)
        old_log_probs_all = torch.tensor(old_log_probs_np).to(DEVICE)
        returns_all = returns.to(DEVICE)
        advantages_all = advantages.to(DEVICE)

        n = len(obs_all)
        losses = {"policy": 0.0, "value": 0.0, "entropy": 0.0, "total": 0.0}
        n_updates = 0

        for _ in range(self.n_epochs):
            indices = torch.randperm(n)
            for start in range(0, n, self.batch_size):
                idx = indices[start: start + self.batch_size]
                obs_b = obs_all[idx]
                masks_b = masks_all[idx]
                actions_b = actions_all[idx]
                old_log_b = old_log_probs_all[idx]
                ret_b = returns_all[idx]
                adv_b = advantages_all[idx]

                log_probs, values, entropy = self.model.evaluate(obs_b, masks_b, actions_b)

                # Ratio for PPO clipping
                ratio = torch.exp(log_probs - old_log_b)
                surr1 = ratio * adv_b
                surr2 = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range) * adv_b
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = F.mse_loss(values, ret_b)
                entropy_loss = -entropy.mean()

                loss = (
                    policy_loss
                    + self.value_coef * value_loss
                    + self.entropy_coef * entropy_loss
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()

                losses["policy"] += policy_loss.item()
                losses["value"] += value_loss.item()
                losses["entropy"] += entropy_loss.item()
                losses["total"] += loss.item()
                n_updates += 1

        for k in losses:
            losses[k] /= max(1, n_updates)
        return losses

    def _update_lr(self, total_timesteps: int) -> None:
        """Linear LR decay from base_lr to 0 over training."""
        if total_timesteps <= 0:
            return
        frac = max(0.0, 1.0 - self.total_steps / total_timesteps)
        new_lr = self._base_lr * frac
        for pg in self.optimizer.param_groups:
            pg["lr"] = new_lr

    def train(
        self,
        total_timesteps: int,
        save_interval: int = 100_000,
        checkpoint_prefix: str = "risk_ppo",
        log_interval: int = 10,
    ) -> None:
        self._total_timesteps = total_timesteps
        CHECKPOINTS_DIR.mkdir(exist_ok=True)
        rollout_count = 0
        t_start = time.time()

        print(f"Training on device: {DEVICE}")
        print(f"Total timesteps: {total_timesteps:,} | Steps/rollout: {self.n_steps}")
        print(f"Saving every {save_interval:,} steps to {CHECKPOINTS_DIR}/")
        print()

        while self.total_steps < total_timesteps:
            self._update_lr(total_timesteps)
            self.collect_rollout()
            losses = self.update()
            rollout_count += 1

            # Periodic logging
            if rollout_count % log_interval == 0:
                elapsed = time.time() - t_start
                fps = self.total_steps / max(1, elapsed)
                mean_rew = np.mean(self._episode_rewards) if self._episode_rewards else 0.0
                mean_len = np.mean(self._episode_lengths) if self._episode_lengths else 0.0
                win_rate = np.mean(self._wins) if self._wins else 0.0
                print(
                    f"[{self.total_steps:>9,}] "
                    f"fps={fps:>6.0f}  "
                    f"ep_rew={mean_rew:>6.3f}  "
                    f"ep_len={mean_len:>6.0f}  "
                    f"win%={win_rate*100:>5.1f}  "
                    f"loss={losses['total']:>7.4f}  "
                    f"pi={losses['policy']:>7.4f}  "
                    f"v={losses['value']:>7.4f}  "
                    f"ent={losses['entropy']:>6.4f}"
                )

            # Periodic checkpoint
            if self.total_steps % save_interval < self.n_steps:
                path = CHECKPOINTS_DIR / f"{checkpoint_prefix}_{self.total_steps}.pt"
                self.save(str(path))
                print(f"  >> checkpoint saved: {path.name}")

        final_path = CHECKPOINTS_DIR / f"{checkpoint_prefix}_final_{total_timesteps}.pt"
        self.save(str(final_path))
        print(f"\nTraining complete. Final model: {final_path}")

    def save(self, path: str) -> None:
        torch.save({
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "total_steps": self.total_steps,
            "obs_size": self.model.obs_size,
            "action_size": self.model.action_size,
            # Without the shape, --hidden/--layers runs cannot be reloaded
            "hidden_size": self.model.hidden_size,
            "num_layers": self.model.num_layers,
        }, path)

    @classmethod
    def load_model(cls, path: str, device: torch.device | None = None) -> RiskActorCritic:
        """Load a saved model checkpoint. Returns a RiskActorCritic on DEVICE."""
        dev = device or DEVICE
        # Checkpoints hold only tensors and ints, so the safe loader is enough
        ckpt = torch.load(path, map_location=dev, weights_only=True)
        model = RiskActorCritic(
            obs_size=ckpt["obs_size"],
            action_size=ckpt["action_size"],
            # Checkpoints written before the shape was recorded used the defaults
            hidden_size=ckpt.get("hidden_size", 256),
            num_layers=ckpt.get("num_layers", 3),
        ).to(dev)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        return model


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Train Risk RL agent with PyTorch PPO.")
    parser.add_argument("--config", default="grid_6x6",
                        help="Board config (grid_6x6, small_20, classic_42)")
    parser.add_argument("--num-players", type=int, default=2)
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--checkpoint", default=None,
                        help="Resume from .pt checkpoint file")
    parser.add_argument("--save-interval", type=int, default=100_000)
    parser.add_argument("--n-steps", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--entropy-coef", type=float, default=0.05,
                        help="Entropy regularisation coefficient (default 0.05)")
    parser.add_argument("--hidden", type=int, default=256,
                        help="Hidden layer width")
    parser.add_argument("--layers", type=int, default=3,
                        help="Number of hidden layers in trunk")
    parser.add_argument("--opponent", default="rule_based",
                        choices=["random", "rule_based", "mcts"],
                        help="Opponent agent type for training")
    args = parser.parse_args()

    from agents.rule_based import RuleBasedAgent
    from agents.random_agent import RandomAgent
    from agents.mcts import MCTSAgent

    opponent_map = {
        "random": RandomAgent,
        "rule_based": RuleBasedAgent,
        "mcts": MCTSAgent,
    }

    env = RiskEnv(
        config_name=args.config,
        num_players=args.num_players,
        opponent_agent_cls=opponent_map[args.opponent],
    )

    obs_dict, _ = env.reset()
    obs_size = obs_dict["obs"].shape[0]
    print(f"Board: {args.config} | Players: {args.num_players} | Obs size: {obs_size}")
    print(f"Device: {DEVICE} | Hidden: {args.hidden}x{args.layers} | Actions: {MAX_LEGAL_ACTIONS}")

    if args.checkpoint:
        print(f"Resuming from: {args.checkpoint}")
        model = PPOTrainer.load_model(args.checkpoint)
    else:
        model = make_model(obs_size, hidden_size=args.hidden, num_layers=args.layers)

    trainer = PPOTrainer(
        env=env,
        model=model,
        lr=args.lr,
        entropy_coef=args.entropy_coef,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
    )

    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=DEVICE, weights_only=True)
        trainer.total_steps = ckpt.get("total_steps", 0)
        trainer.optimizer.load_state_dict(ckpt["optimizer_state"])

    trainer.train(
        total_timesteps=args.timesteps,
        save_interval=args.save_interval,
    )


if __name__ == "__main__":
    main()
