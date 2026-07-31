"""
GAE and rollout-buffer arithmetic, on synthetic transitions.

These tests need no environment, no network and no checkpoint: the buffer is
filled by hand so every expected number can be written out in closed form.
"""
import pytest

torch = pytest.importorskip("torch", reason="torch is needed for the PPO trainer")

from training.ppo_trainer import RolloutBuffer  # noqa: E402

GAMMA = 0.9
LAM = 0.8


def fill(buffer, rewards, values, dones):
    for r, v, d in zip(rewards, values, dones):
        buffer.add(obs=None, action=0, log_prob=0.0, value=v,
                   reward=r, done=float(d), mask=None)


class TestGAE:
    def test_single_step_with_no_done_bootstraps(self):
        buffer = RolloutBuffer()
        fill(buffer, rewards=[1.0], values=[0.5], dones=[0])
        returns, adv = buffer.compute_returns_and_advantages(2.0, GAMMA, LAM)
        expected = 1.0 + GAMMA * 2.0 - 0.5
        assert adv[0].item() == pytest.approx(expected)
        assert returns[0].item() == pytest.approx(expected + 0.5)

    def test_terminal_step_does_not_bootstrap(self):
        """A finished episode has no next state; the advantage is r - V(s)."""
        buffer = RolloutBuffer()
        fill(buffer, rewards=[1.0], values=[0.5], dones=[1])
        _, adv = buffer.compute_returns_and_advantages(2.0, GAMMA, LAM)
        assert adv[0].item() == pytest.approx(1.0 - 0.5)

    def test_value_of_the_next_episode_never_leaks_backwards(self):
        """
        Step 0 ends the episode, so step 1 belongs to a fresh game. Making the
        next episode's value huge must not move step 0's advantage at all.
        """
        def advantage_of_step_zero(next_value):
            buffer = RolloutBuffer()
            fill(buffer, rewards=[1.0, 0.0], values=[0.5, next_value], dones=[1, 0])
            _, adv = buffer.compute_returns_and_advantages(0.0, GAMMA, LAM)
            return adv[0].item()

        assert advantage_of_step_zero(0.0) == pytest.approx(advantage_of_step_zero(1000.0))
        assert advantage_of_step_zero(0.0) == pytest.approx(1.0 - 0.5)

    def test_trace_is_cut_at_the_episode_boundary(self):
        """
        Rewards earned in the next episode must not flow back into this one's
        advantages, however large they are.
        """
        def advantage_of_step_zero(next_reward):
            buffer = RolloutBuffer()
            fill(buffer, rewards=[0.0, next_reward], values=[0.0, 0.0], dones=[1, 0])
            _, adv = buffer.compute_returns_and_advantages(0.0, GAMMA, LAM)
            return adv[0].item()

        assert advantage_of_step_zero(0.0) == pytest.approx(0.0)
        assert advantage_of_step_zero(50.0) == pytest.approx(0.0)

    def test_within_an_episode_the_trace_does_propagate(self):
        buffer = RolloutBuffer()
        fill(buffer, rewards=[0.0, 1.0], values=[0.0, 0.0], dones=[0, 1])
        _, adv = buffer.compute_returns_and_advantages(0.0, GAMMA, LAM)
        # delta1 = 1.0; delta0 = 0; adv0 = gamma * lambda * adv1
        assert adv[1].item() == pytest.approx(1.0)
        assert adv[0].item() == pytest.approx(GAMMA * LAM * 1.0)

    def test_final_step_marked_done_ignores_the_bootstrap_value(self):
        buffer = RolloutBuffer()
        fill(buffer, rewards=[0.0, -1.0], values=[0.0, 0.25], dones=[0, 1])
        _, adv = buffer.compute_returns_and_advantages(99.0, GAMMA, LAM)
        assert adv[1].item() == pytest.approx(-1.0 - 0.25)

    def test_returns_equal_advantages_plus_values(self):
        buffer = RolloutBuffer()
        fill(buffer, rewards=[0.1, 0.2, -1.0, 0.3],
             values=[0.5, 0.4, 0.3, 0.2], dones=[0, 0, 1, 0])
        returns, adv = buffer.compute_returns_and_advantages(0.1, GAMMA, LAM)
        for i, v in enumerate([0.5, 0.4, 0.3, 0.2]):
            assert returns[i].item() == pytest.approx(adv[i].item() + v, abs=1e-5)

    def test_all_finite_on_a_long_mixed_rollout(self):
        buffer = RolloutBuffer()
        n = 200
        fill(buffer,
             rewards=[0.01 * (i % 7) for i in range(n)],
             values=[0.5] * n,
             dones=[1 if i % 33 == 0 else 0 for i in range(n)])
        returns, adv = buffer.compute_returns_and_advantages(0.0, 0.99, 0.95)
        assert torch.isfinite(returns).all()
        assert torch.isfinite(adv).all()

    def test_accepts_tensor_values(self):
        buffer = RolloutBuffer()
        fill(buffer, rewards=[1.0], values=[torch.tensor(0.5)], dones=[1])
        _, adv = buffer.compute_returns_and_advantages(0.0, GAMMA, LAM)
        assert adv[0].item() == pytest.approx(0.5)


class TestSeeding:
    def _rollout(self, seed):
        import numpy as np

        from agents.neural_net import make_model
        from training.gym_env import RiskEnv
        from training.ppo_trainer import PPOTrainer

        torch.manual_seed(seed)
        np.random.seed(seed)
        env = RiskEnv(config_name="small_20", num_players=2)
        model = make_model(env.observation_space["obs"].shape[0], hidden_size=16,
                           num_layers=1)
        trainer = PPOTrainer(env=env, model=model, n_steps=48, seed=seed)
        trainer.collect_rollout()
        return list(trainer.buffer.actions), [round(r, 8) for r in trainer.buffer.rewards]

    def test_a_seeded_run_replays_exactly(self):
        """Without a seed stream on env.reset(), no training run is reproducible."""
        pytest.importorskip("gymnasium", reason="gymnasium is needed for RiskEnv")
        assert self._rollout(7) == self._rollout(7)

    def test_episode_seeds_are_distinct_and_bounded(self):
        from agents.neural_net import RiskActorCritic
        from training.ppo_trainer import PPOTrainer

        trainer = PPOTrainer.__new__(PPOTrainer)
        trainer.seed, trainer._episode_index = 11, 0
        seeds = [trainer._next_episode_seed() for _ in range(5)]
        assert seeds == [11, 12, 13, 14, 15]
        assert all(0 <= s < 2 ** 31 - 1 for s in seeds)
        assert RiskActorCritic  # imported models still construct

    def test_an_unseeded_trainer_leaves_the_env_to_seed_itself(self):
        from training.ppo_trainer import PPOTrainer

        trainer = PPOTrainer.__new__(PPOTrainer)
        trainer.seed, trainer._episode_index = None, 0
        assert trainer._next_episode_seed() is None


@pytest.mark.parametrize("kwargs", [
    {"n_steps": 0}, {"batch_size": 0}, {"n_epochs": 0},
])
def test_rejects_degenerate_hyperparameters(kwargs):
    from agents.neural_net import RiskActorCritic
    from training.ppo_trainer import PPOTrainer

    with pytest.raises(ValueError):
        PPOTrainer(env=None, model=RiskActorCritic(obs_size=8, hidden_size=4,
                                                   num_layers=1), **kwargs)


class TestBuffer:
    def test_clear_empties_every_field(self):
        buffer = RolloutBuffer()
        fill(buffer, rewards=[1.0, 2.0], values=[0.0, 0.0], dones=[0, 1])
        assert len(buffer) == 2
        buffer.clear()
        assert len(buffer) == 0
        assert buffer.obs == [] and buffer.masks == [] and buffer.dones == []
