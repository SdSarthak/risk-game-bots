"""
Gymnasium-compatible environment for Risk.

The RL agent plays as player 0. All other players are controlled by opponent
agents (default: RuleBasedAgent). The environment steps through the entire
game turn, including all opponent moves, before returning control to the RL agent.

Observation:  Dict of the flat state vector (see GameState.to_observation) and a
              binary mask over the action space.
Action space: Discrete over the fixed layout in agents/action_space.py, where a
              given index always names the same move on the same board. Illegal
              indices are masked out rather than remapped.
"""
from __future__ import annotations

import pathlib
from typing import Any

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    raise ImportError("Install gymnasium: pip install gymnasium")

from engine.board import BoardConfig
from engine.rules import Action, RulesEngine
from engine.state import GameState
from agents.action_space import ACTION_SPACE_SIZE, RiskActionSpace
from agents.rule_based import RuleBasedAgent
from training.reward import RewardShaper

CONFIGS_DIR = pathlib.Path(__file__).parent.parent / "configs"
MAX_LEGAL_ACTIONS = ACTION_SPACE_SIZE

# Stand-offs between cautious policies can run indefinitely; truncate them so a
# single episode can never stall a training run.
MAX_EPISODE_STEPS = 5_000


class RiskEnv(gym.Env):
    """
    Single-agent Risk environment.
    RL agent = player 0. Opponents = RuleBasedAgent by default.
    """

    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(self,
                 config_name: str = "small_20",
                 num_players: int = 2,
                 opponent_agent_cls=None,
                 gamma: float = 0.99,
                 max_episode_steps: int = MAX_EPISODE_STEPS,
                 render_mode: str | None = None):
        super().__init__()
        config_path = CONFIGS_DIR / f"{config_name}.json"
        if not config_path.exists():
            available = ", ".join(sorted(p.stem for p in CONFIGS_DIR.glob("*.json")))
            raise FileNotFoundError(
                f"No board config named '{config_name}'. Available: {available}"
            )
        self.board = BoardConfig.load(str(config_path))
        self.num_players = num_players
        self.max_episode_steps = max_episode_steps
        self.render_mode = render_mode
        self.action_encoder = RiskActionSpace(self.board)
        self._elapsed_steps = 0

        # Opponent factory
        if opponent_agent_cls is None:
            opponent_agent_cls = RuleBasedAgent
        self._opponent_agents = [opponent_agent_cls(i) for i in range(1, num_players)]

        # Reward shaper for player 0
        self._shaper = RewardShaper(player_id=0, gamma=gamma)

        # Internal state (initialized in reset)
        self._state: GameState | None = None
        self._engine: RulesEngine | None = None
        self._mask: np.ndarray = np.zeros(ACTION_SPACE_SIZE, dtype=np.int8)

        # Spaces
        dummy_state = GameState.new_game(self.board, num_players, seed=0)
        obs_size = dummy_state.obs_size

        self.observation_space = spaces.Dict({
            "obs": spaces.Box(low=-1.0, high=1.0, shape=(obs_size,), dtype=np.float32),
            "action_mask": spaces.Box(low=0, high=1,
                                      shape=(MAX_LEGAL_ACTIONS,), dtype=np.int8),
        })
        self.action_space = spaces.Discrete(MAX_LEGAL_ACTIONS)

    def reset(self, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._engine = RulesEngine(self.board, self.num_players, seed=seed)
        self._state = GameState.new_game(self.board, self.num_players, seed=seed)

        for agent in self._opponent_agents:
            agent.reset()

        # Advance past any initial opponent turns before player 0's first turn
        self._state = self._run_opponents_until_our_turn()
        self._mask = self.action_encoder.legal_mask(self._state)
        self._shaper.reset(self._state)
        self._elapsed_steps = 0

        return self._get_obs(), {}

    def step(self, action_idx: int):
        if self._state is None:
            raise RuntimeError("Call reset() before step()")

        action = self._resolve_action(int(action_idx))
        self._state = self._engine.apply_action(self._state, action)
        self._elapsed_steps += 1

        # Check terminal after our action
        terminated = self._engine.is_terminal(self._state)
        win = terminated and self._engine.winner(self._state) == 0
        eliminated = self._state.eliminated[0]

        if not terminated:
            # Let opponents play until it's our turn again
            self._state = self._run_opponents_until_our_turn()
            terminated = self._engine.is_terminal(self._state)
            eliminated = eliminated or self._state.eliminated[0]
            win = win or (terminated and self._engine.winner(self._state) == 0)

        reward = self._shaper.step(self._state, win=win, eliminated=eliminated)

        self._mask = self.action_encoder.legal_mask(self._state)
        truncated = not terminated and self._elapsed_steps >= self.max_episode_steps
        obs = self._get_obs()

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, {}

    def _resolve_action(self, action_idx: int) -> Action:
        """
        Decode a policy output into a legal engine action.

        A masked policy always lands on something playable; anything else falls
        back to the first legal index so a stray output cannot wedge the episode.
        """
        action = self.action_encoder.decode(action_idx, self._state)
        if action is not None:
            return action
        legal_indices = np.flatnonzero(self._mask)
        for index in legal_indices:
            fallback = self.action_encoder.decode(int(index), self._state)
            if fallback is not None:
                return fallback
        # Ending the phase is legal in every state the engine can reach
        return Action(phase=self._state.phase, troops=-1)

    def render(self):
        if self._state is not None:
            print(self._state)

    def _run_opponents_until_our_turn(self) -> GameState:
        """Step all opponent agents until it's player 0's turn (or game over)."""
        state = self._state
        while (not self._engine.is_terminal(state) and
               state.current_player != 0):
            player = state.current_player
            agent = self._opponent_agents[player - 1]
            legal = self._engine.legal_actions(state)
            action = agent.choose_action(state, legal)
            state = self._engine.apply_action(state, action)
        return state

    def _get_obs(self) -> dict:
        return {"obs": self._state.to_observation(), "action_mask": self._mask.copy()}

    def action_mask(self) -> np.ndarray:
        """Current binary mask over the fixed action space."""
        return self._mask.copy()

    def legal_actions(self) -> list[Action]:
        """The actions the encoder can currently express, in index order."""
        decoded = (self.action_encoder.decode(int(i), self._state)
                   for i in np.flatnonzero(self._mask))
        return [a for a in decoded if a is not None]
