"""
Self-play training loop for the RL agent.

Maintains a pool of past RL checkpoints as opponents to prevent overfitting
to a single fixed opponent. Player 0 trains; opponents are sampled from the pool.
"""
from __future__ import annotations

import pathlib
import random
from collections import deque

from engine.board import BoardConfig
from engine.rules import RulesEngine
from engine.state import GameState
from agents.rule_based import RuleBasedAgent
from training.reward import RewardShaper

CHECKPOINTS_DIR = pathlib.Path(__file__).parent.parent / "checkpoints"


class SelfPlayTrainer:
    """
    Runs episodes of Risk for self-play data collection.
    The actual PPO update happens in run_training.py via SB3.
    This class handles opponent sampling and episode management.
    """

    def __init__(self, board: BoardConfig, num_players: int = 2,
                 pool_size: int = 5, seed: int | None = None):
        self.board = board
        self.num_players = num_players
        self.pool_size = pool_size
        self._rng = random.Random(seed)
        # Checkpoint pool: list of file paths to saved RL models
        self._checkpoint_pool: deque[str] = deque(maxlen=pool_size)
        # Always include a rule-based agent as the baseline opponent
        self._baseline_opponent = RuleBasedAgent(1)

    def add_checkpoint(self, checkpoint_path: str) -> None:
        self._checkpoint_pool.append(checkpoint_path)

    def sample_opponent(self):
        """Return a RuleBasedAgent (baseline) or a loaded RL agent from pool."""
        if not self._checkpoint_pool or self._rng.random() < 0.3:
            return RuleBasedAgent(1)
        # Load PyTorch RL agent from a random checkpoint (.pt file)
        try:
            from agents.rl_agent import RLAgent
            path = self._rng.choice(list(self._checkpoint_pool))
            return RLAgent.load(1, path)
        except Exception:
            return RuleBasedAgent(1)

    def run_episode(self, rl_agent, max_turns: int = 2000) -> dict:
        """
        Run one full game episode collecting experience for rl_agent (player 0).
        Returns episode statistics.
        """
        seed = self._rng.randint(0, 2**31)
        state = GameState.new_game(self.board, self.num_players, seed=seed)
        engine = RulesEngine(self.board, self.num_players, seed=seed)
        opponent = self.sample_opponent()
        shaper = RewardShaper(player_id=0)
        shaper.reset(state)

        total_reward = 0.0
        turns = 0

        while not engine.is_terminal(state) and turns < max_turns:
            player = state.current_player
            legal = engine.legal_actions(state)

            if player == 0:
                action = rl_agent.choose_action(state, legal)
            else:
                action = opponent.choose_action(state, legal)

            new_state = engine.apply_action(state, action)
            win = engine.is_terminal(new_state) and engine.winner(new_state) == 0
            eliminated = new_state.eliminated[0]
            reward = shaper.step(new_state, win=win, eliminated=eliminated)

            if player == 0:
                total_reward += reward

            state = new_state
            turns += 1

        winner = engine.winner(state)
        return {
            "winner": winner,
            "won": winner == 0,
            "turns": turns,
            "total_reward": total_reward,
        }
