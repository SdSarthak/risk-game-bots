"""
Tests for the learned agent's inference path.

An untrained network is enough: what matters is that whatever the policy emits
is a move the engine accepts, and that a checkpoint from a different action
space is refused rather than mis-played.
"""
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch", reason="torch is needed for the RL agent")

from agents.action_space import ACTION_SPACE_SIZE  # noqa: E402
from agents.neural_net import RiskActorCritic  # noqa: E402
from agents.rl_agent import RLAgent  # noqa: E402
from agents.rule_based import RuleBasedAgent  # noqa: E402
from engine.constants import Phase  # noqa: E402
from engine.rules import RulesEngine  # noqa: E402
from engine.state import (  # noqa: E402
    MAX_CARDS_PER_PLAYER, MAX_PLAYERS, MAX_TERRITORIES, GameState,
)
from training.ppo_trainer import PPOTrainer  # noqa: E402

OBS_SIZE = MAX_TERRITORIES * 2 + 1 + 3 + MAX_PLAYERS * MAX_CARDS_PER_PLAYER


def make_agent(player_id=0, obs_size=OBS_SIZE, deterministic=True):
    model = RiskActorCritic(obs_size=obs_size, hidden_size=32, num_layers=2)
    return RLAgent(player_id, model, deterministic=deterministic)


class TestCheckpointCompatibility:
    def test_rejects_a_head_from_another_action_space(self):
        model = RiskActorCritic(obs_size=OBS_SIZE, action_size=500,
                                hidden_size=16, num_layers=1)
        with pytest.raises(ValueError, match="action space"):
            RLAgent(0, model)

    def test_round_trips_a_non_default_architecture(self, tmp_path, small_board):
        """--hidden/--layers runs were unloadable until the shape was recorded."""
        state = GameState.new_game(small_board, num_players=2, seed=1)
        model = RiskActorCritic(obs_size=state.obs_size, hidden_size=64, num_layers=2)
        env = SimpleNamespace()
        trainer = PPOTrainer.__new__(PPOTrainer)
        trainer.env = env
        trainer.model = model
        trainer.optimizer = torch.optim.Adam(model.parameters())
        trainer.total_steps = 1024

        path = tmp_path / "risk_ppo_1024.pt"
        trainer.save(str(path))

        restored = PPOTrainer.load_model(str(path))
        assert restored.action_size == ACTION_SPACE_SIZE
        assert restored.hidden_size == 64
        assert restored.num_layers == 2
        assert RLAgent(0, restored) is not None

    def test_older_checkpoints_still_load(self, tmp_path, small_board):
        """Checkpoints written before the shape was recorded used the defaults."""
        state = GameState.new_game(small_board, num_players=2, seed=1)
        model = RiskActorCritic(obs_size=state.obs_size)
        path = tmp_path / "risk_ppo_2048.pt"
        torch.save({
            "model_state": model.state_dict(),
            "obs_size": model.obs_size,
            "action_size": model.action_size,
            "total_steps": 2048,
        }, path)
        assert PPOTrainer.load_model(str(path)).hidden_size == 256


class TestInference:
    def test_only_plays_legal_actions(self, small_board):
        state = GameState.new_game(small_board, num_players=2, seed=2)
        engine = RulesEngine(small_board, num_players=2, seed=2)
        agent = make_agent(0, obs_size=state.obs_size)
        opponent = RuleBasedAgent(1)
        for _ in range(150):
            if engine.is_terminal(state):
                break
            legal = engine.legal_actions(state)
            actor = agent if state.current_player == 0 else opponent
            action = actor.choose_action(state, legal)
            assert action in legal, f"policy produced an illegal action {action}"
            state = engine.apply_action(state, action)

    def test_acts_in_every_phase(self, small_board):
        engine = RulesEngine(small_board, num_players=2, seed=3)
        for phase in Phase:
            state = GameState.new_game(small_board, num_players=2, seed=3)
            state.phase = phase
            state.troops_to_place = 4 if phase is Phase.DRAFT else 0
            agent = make_agent(0, obs_size=state.obs_size)
            action = agent.choose_action(state, engine.legal_actions(state))
            assert action.phase is phase

    def test_deterministic_mode_repeats_itself(self, small_board):
        state = GameState.new_game(small_board, num_players=2, seed=4)
        engine = RulesEngine(small_board, num_players=2, seed=4)
        agent = make_agent(0, obs_size=state.obs_size, deterministic=True)
        legal = engine.legal_actions(state)
        choices = {agent.choose_action(state, legal) for _ in range(5)}
        assert len(choices) == 1

    def test_works_across_boards_without_rebuilding(self, small_board, classic_board):
        agent = make_agent(0)
        for board in (small_board, classic_board, small_board):
            state = GameState.new_game(board, num_players=2, seed=5)
            engine = RulesEngine(board, num_players=2, seed=5)
            action = agent.choose_action(state, engine.legal_actions(state))
            assert action in engine.legal_actions(state)


class TestNetwork:
    def test_masked_logits_never_pick_an_illegal_slot(self):
        model = RiskActorCritic(obs_size=OBS_SIZE, hidden_size=32, num_layers=2)
        obs = torch.zeros(4, OBS_SIZE)
        mask = torch.zeros(4, ACTION_SPACE_SIZE)
        mask[:, 3] = 1.0
        dist, value = model(obs, mask)
        assert torch.equal(dist.sample(), torch.full((4,), 3))
        assert value.shape == (4,)

    def test_an_empty_mask_does_not_produce_nans(self):
        """A row with nothing legal used to make the distribution emit NaNs."""
        model = RiskActorCritic(obs_size=OBS_SIZE, hidden_size=32, num_layers=2)
        dist, value = model(torch.zeros(2, OBS_SIZE), torch.zeros(2, ACTION_SPACE_SIZE))
        assert torch.isfinite(dist.probs).all()
        assert torch.isfinite(value).all()

    def test_evaluate_returns_gradients_worth_of_signal(self):
        model = RiskActorCritic(obs_size=OBS_SIZE, hidden_size=32, num_layers=2)
        mask = torch.zeros(3, ACTION_SPACE_SIZE)
        mask[:, :5] = 1.0
        actions = torch.tensor([0, 1, 2])
        log_probs, values, entropy = model.evaluate(torch.randn(3, OBS_SIZE), mask, actions)
        assert log_probs.shape == (3,)
        assert values.shape == (3,)
        assert (entropy > 0).all()
