"""Tests for reward shaping, the Gymnasium environment and self-play plumbing."""
import numpy as np
import pytest

from agents.random_agent import RandomAgent
from engine.rules import RulesEngine
from engine.state import GameState
from training.reward import RewardShaper, potential
from training.self_play import SelfPlayTrainer

gym = pytest.importorskip("gymnasium", reason="gymnasium is needed for RiskEnv")

from training.gym_env import MAX_LEGAL_ACTIONS, RiskEnv  # noqa: E402


class TestPotential:
    def test_bounded_between_zero_and_one(self, any_board):
        state = GameState.new_game(any_board, num_players=2, seed=1)
        for player in range(2):
            assert 0.0 <= potential(state, player) <= 1.0

    def test_owning_everything_maximises_potential(self, small_board):
        state = GameState.new_game(small_board, num_players=2, seed=1)
        state.owners = [0] * small_board.num_territories
        assert potential(state, 0) == pytest.approx(1.0)
        assert potential(state, 1) == pytest.approx(0.0)

    def test_more_territory_scores_higher(self, small_board):
        weak = GameState.new_game(small_board, num_players=2, seed=1)
        strong = weak.copy()
        strong.owners = [0] * (small_board.num_territories - 1) + [1]
        assert potential(strong, 0) > potential(weak, 0)


class TestRewardShaper:
    def test_win_pays_out(self, small_game):
        state, _ = small_game
        shaper = RewardShaper(player_id=0)
        shaper.reset(state)
        assert shaper.step(state, win=True, eliminated=False) == RewardShaper.WIN_REWARD

    def test_elimination_costs(self, small_game):
        state, _ = small_game
        shaper = RewardShaper(player_id=0)
        shaper.reset(state)
        assert shaper.step(state, win=False, eliminated=True) == RewardShaper.LOSS_REWARD

    def test_gaining_ground_is_rewarded(self, small_board):
        state = GameState.new_game(small_board, num_players=2, seed=1)
        shaper = RewardShaper(player_id=0)
        shaper.reset(state)
        better = state.copy()
        better.owners = [0] * small_board.num_territories
        assert shaper.step(better, win=False, eliminated=False) > 0

    def test_losing_ground_is_penalised(self, small_board):
        state = GameState.new_game(small_board, num_players=2, seed=1)
        shaper = RewardShaper(player_id=0)
        shaper.reset(state)
        worse = state.copy()
        worse.owners = [1] * (small_board.num_territories - 1) + [0]
        assert shaper.step(worse, win=False, eliminated=False) < 0

    def test_shaping_is_small_next_to_the_terminal_reward(self, small_board):
        """Shaping must not drown out the win/loss signal it is guiding towards."""
        state = GameState.new_game(small_board, num_players=2, seed=1)
        shaper = RewardShaper(player_id=0)
        shaper.reset(state)
        best = state.copy()
        best.owners = [0] * small_board.num_territories
        assert abs(shaper.step(best, win=False, eliminated=False)) < RewardShaper.WIN_REWARD


class TestRiskEnv:
    @pytest.fixture
    def env(self):
        return RiskEnv(config_name="small_20", num_players=2)

    def test_unknown_config_names_the_alternatives(self):
        with pytest.raises(FileNotFoundError, match="small_20"):
            RiskEnv(config_name="not_a_board")

    def test_reset_returns_matching_observation(self, env):
        obs, info = env.reset(seed=0)
        assert env.observation_space.contains(obs)
        assert info == {}

    def test_observation_is_finite(self, env):
        obs, _ = env.reset(seed=0)
        assert np.all(np.isfinite(obs["obs"]))

    def test_mask_marks_exactly_the_legal_actions(self, env):
        obs, _ = env.reset(seed=0)
        expected = min(len(env.legal_actions()), MAX_LEGAL_ACTIONS)
        assert obs["action_mask"].sum() == expected
        assert obs["action_mask"][:expected].all()
        assert not obs["action_mask"][expected:].any()

    def test_agent_always_acts_on_its_own_turn(self, env):
        env.reset(seed=0)
        for _ in range(50):
            assert env._state.current_player == 0
            obs, reward, terminated, truncated, _ = env.step(0)
            if terminated or truncated:
                break

    def test_step_before_reset_raises(self, env):
        with pytest.raises(RuntimeError):
            env.step(0)

    def test_out_of_range_action_is_clamped(self, env):
        env.reset(seed=0)
        obs, reward, terminated, truncated, _ = env.step(MAX_LEGAL_ACTIONS * 10)
        assert np.isfinite(reward)

    def test_episode_terminates_and_pays_out(self, env):
        env.reset(seed=0)
        rewards = []
        for _ in range(env.max_episode_steps + 1):
            _, reward, terminated, truncated, _ = env.step(0)
            rewards.append(reward)
            if terminated or truncated:
                break
        else:
            pytest.fail("episode ran past its own step limit")
        assert terminated or truncated

    def test_truncates_instead_of_running_forever(self, small_board):
        env = RiskEnv(config_name="small_20", num_players=2, max_episode_steps=3)
        env.reset(seed=0)
        flags = [env.step(0)[2:4] for _ in range(3)]
        assert any(terminated or truncated for terminated, truncated in flags)

    def test_reset_is_reproducible(self, env):
        a, _ = env.reset(seed=123)
        b, _ = env.reset(seed=123)
        assert np.array_equal(a["obs"], b["obs"])


class TestSelfPlayTrainer:
    def test_falls_back_to_the_baseline_with_an_empty_pool(self, small_board):
        trainer = SelfPlayTrainer(small_board, num_players=2, seed=0)
        assert trainer.sample_opponent().__class__.__name__ == "RuleBasedAgent"

    def test_unreadable_checkpoint_costs_one_episode_not_the_run(self, small_board, tmp_path):
        broken = tmp_path / "broken.pt"
        broken.write_text("not a torch checkpoint")
        trainer = SelfPlayTrainer(small_board, num_players=2, seed=0)
        for _ in range(5):
            trainer.add_checkpoint(str(broken))
        for _ in range(20):
            assert trainer.sample_opponent() is not None

    def test_pool_is_bounded(self, small_board):
        trainer = SelfPlayTrainer(small_board, num_players=2, pool_size=3, seed=0)
        for i in range(10):
            trainer.add_checkpoint(f"ckpt_{i}.pt")
        assert len(trainer._checkpoint_pool) == 3

    def test_episode_reports_a_result(self, small_board):
        trainer = SelfPlayTrainer(small_board, num_players=2, seed=0)
        stats = trainer.run_episode(RandomAgent(0, seed=0), max_turns=4000)
        assert stats["turns"] > 0
        assert stats["won"] == (stats["winner"] == 0)


def test_engine_and_env_agree_on_legal_actions(small_board):
    """The env must hand the policy exactly what the engine considers legal."""
    env = RiskEnv(config_name="small_20", num_players=2)
    env.reset(seed=5)
    engine = RulesEngine(small_board, num_players=2)
    assert env.legal_actions() == engine.legal_actions(env._state)
