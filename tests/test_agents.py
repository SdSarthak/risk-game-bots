"""Tests for the non-learned agents and the checkpoint locator."""
import pytest

from agents.checkpoints import find_latest_checkpoint, list_checkpoints
from agents.mcts import MCTSAgent
from agents.random_agent import RandomAgent
from agents.rule_based import RuleBasedAgent
from engine.constants import Phase
from engine.rules import RulesEngine
from engine.state import GameState

AGENT_FACTORIES = [
    lambda pid: RandomAgent(pid, seed=0),
    lambda pid: RuleBasedAgent(pid),
    lambda pid: MCTSAgent(pid, num_simulations=8),
]


def play(board, agents, seed=0, max_steps=20_000):
    state = GameState.new_game(board, num_players=len(agents), seed=seed)
    engine = RulesEngine(board, num_players=len(agents), seed=seed)
    for _ in range(max_steps):
        if engine.is_terminal(state):
            break
        legal = engine.legal_actions(state)
        state = engine.apply_action(state, agents[state.current_player]
                                    .choose_action(state, legal))
    return state, engine


@pytest.mark.parametrize("factory", AGENT_FACTORIES, ids=["random", "rule_based", "mcts"])
class TestAgentContract:
    def test_only_returns_legal_actions(self, small_board, factory):
        agent = factory(0)
        state = GameState.new_game(small_board, num_players=2, seed=1)
        engine = RulesEngine(small_board, num_players=2, seed=1)
        for _ in range(120):
            if engine.is_terminal(state):
                break
            legal = engine.legal_actions(state)
            action = (agent if state.current_player == 0
                      else RuleBasedAgent(1)).choose_action(state, legal)
            assert action in legal, f"{agent} returned an illegal action {action}"
            state = engine.apply_action(state, action)

    def test_handles_every_phase(self, small_board, factory):
        agent = factory(0)
        engine = RulesEngine(small_board, num_players=2, seed=2)
        for phase in Phase:
            state = GameState.new_game(small_board, num_players=2, seed=2)
            state.phase = phase
            state.troops_to_place = 5 if phase is Phase.DRAFT else 0
            action = agent.choose_action(state, engine.legal_actions(state))
            assert action.phase is phase

    def test_reset_is_safe_to_call(self, small_board, factory):
        agent = factory(0)
        agent.reset()
        state = GameState.new_game(small_board, num_players=2, seed=3)
        engine = RulesEngine(small_board, num_players=2, seed=3)
        assert agent.choose_action(state, engine.legal_actions(state)) is not None


class TestRandomAgent:
    def test_same_seed_same_choices(self, small_board):
        state = GameState.new_game(small_board, num_players=2, seed=4)
        engine = RulesEngine(small_board, num_players=2, seed=4)
        legal = engine.legal_actions(state)
        a = [RandomAgent(0, seed=7).choose_action(state, legal) for _ in range(1)]
        b = [RandomAgent(0, seed=7).choose_action(state, legal) for _ in range(1)]
        assert a == b


class TestRuleBasedAgent:
    def test_beats_random_convincingly(self, small_board):
        """The heuristic is the yardstick every other agent is measured against."""
        wins = 0
        for seed in range(12):
            agents = [RuleBasedAgent(0), RandomAgent(1, seed=seed)]
            state, engine = play(small_board, agents, seed=seed)
            wins += engine.winner(state) == 0
        assert wins >= 10, f"rule-based only won {wins}/12 against random"

    def test_drafts_onto_a_threatened_border(self, small_board):
        state = GameState.new_game(small_board, num_players=2, seed=6)
        engine = RulesEngine(small_board, num_players=2, seed=6)
        state.phase = Phase.DRAFT
        state.troops_to_place = 6
        action = RuleBasedAgent(0).choose_action(state, engine.legal_actions(state))
        borders = {t for t in state.territories_of(0)
                   if any(state.owners[nb] != 0 for nb in state.board.adjacent_to(t))}
        assert action.dst in borders
        assert action.troops == 6, "should commit the whole allotment at once"

    def test_declines_hopeless_attacks(self, small_board):
        state = GameState.new_game(small_board, num_players=2, seed=6)
        engine = RulesEngine(small_board, num_players=2, seed=6)
        state.phase = Phase.ATTACK
        # Every enemy territory hugely outnumbers ours
        for tid in range(state.board.num_territories):
            state.troops[tid] = 2 if state.owners[tid] == 0 else 40
        action = RuleBasedAgent(0).choose_action(state, engine.legal_actions(state))
        assert action.is_end_phase()

    def test_takes_a_free_territory(self, small_board):
        state = GameState.new_game(small_board, num_players=2, seed=6)
        engine = RulesEngine(small_board, num_players=2, seed=6)
        state.phase = Phase.ATTACK
        for tid in range(state.board.num_territories):
            state.troops[tid] = 10 if state.owners[tid] == 0 else 1
        action = RuleBasedAgent(0).choose_action(state, engine.legal_actions(state))
        assert not action.is_end_phase()
        assert action.troops == 3, "should attack with the maximum dice"


class TestMCTSAgent:
    def test_simulation_budget_is_respected(self, small_board):
        """num_simulations must override the wall-clock budget so tests stay fast."""
        state = GameState.new_game(small_board, num_players=2, seed=9)
        engine = RulesEngine(small_board, num_players=2, seed=9)
        agent = MCTSAgent(0, time_limit=60.0, num_simulations=5)
        action = agent.choose_action(state, engine.legal_actions(state))
        assert action in engine.legal_actions(state)

    def test_forced_move_short_circuits(self, small_board):
        state = GameState.new_game(small_board, num_players=2, seed=9)
        engine = RulesEngine(small_board, num_players=2, seed=9)
        state.phase = Phase.DRAFT
        state.troops_to_place = 0
        legal = engine.legal_actions(state)
        assert len(legal) == 1
        assert MCTSAgent(0, num_simulations=1).choose_action(state, legal) == legal[0]


class TestCheckpointLocator:
    def test_missing_directory_is_not_an_error(self, tmp_path):
        assert list_checkpoints(tmp_path / "nope") == []
        assert find_latest_checkpoint(tmp_path / "nope") is None

    def test_picks_the_highest_step_count(self, tmp_path):
        for name in ["risk_ppo_2048.pt", "risk_ppo_1001472.pt", "risk_ppo_204800.pt"]:
            (tmp_path / name).touch()
        latest = find_latest_checkpoint(tmp_path, require_compatible=False)
        assert latest.name == "risk_ppo_1001472.pt"

    def test_best_model_wins_over_step_count(self, tmp_path):
        (tmp_path / "risk_ppo_1001472.pt").touch()
        (tmp_path / "best_model.pt").touch()
        latest = find_latest_checkpoint(tmp_path, require_compatible=False)
        assert latest.name == "best_model.pt"

    def test_ignores_non_checkpoints(self, tmp_path):
        (tmp_path / "risk_ppo_500000.zip").touch()
        (tmp_path / "notes.txt").touch()
        assert list_checkpoints(tmp_path) == []

    def test_skips_checkpoints_from_another_action_space(self, tmp_path):
        """An old checkpoint indexes a different layout and must not be picked up."""
        torch = pytest.importorskip("torch")
        from agents.action_space import ACTION_SPACE_SIZE

        stale = tmp_path / "risk_ppo_900000.pt"
        torch.save({"model_state": {}, "obs_size": 118, "action_size": 500}, stale)
        assert find_latest_checkpoint(tmp_path) is None

        current = tmp_path / "risk_ppo_100000.pt"
        torch.save({"model_state": {}, "obs_size": 118,
                    "action_size": ACTION_SPACE_SIZE}, current)
        assert find_latest_checkpoint(tmp_path).name == current.name

    def test_corrupt_checkpoint_is_skipped(self, tmp_path):
        pytest.importorskip("torch")
        (tmp_path / "risk_ppo_100000.pt").write_text("not a checkpoint")
        assert find_latest_checkpoint(tmp_path) is None
