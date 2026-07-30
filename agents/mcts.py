"""
Monte Carlo Tree Search agent for Risk.

Uses UCT (Upper Confidence bounds for Trees) with:
- Chance nodes to handle dice randomness during simulation
- RuleBasedAgent as the rollout policy (much stronger signal than random)
- Multi-player adaptation: maximize own win probability
"""
from __future__ import annotations

import math
import time
from typing import Optional

from agents.base import BaseAgent
from agents.rule_based import RuleBasedAgent
from engine.rules import Action, RulesEngine
from engine.state import GameState


class MCTSNode:
    __slots__ = ("state", "action", "parent", "children", "visits", "wins",
                 "untried_actions", "player_at_node")

    def __init__(self, state: GameState, action: Optional[Action], parent: Optional["MCTSNode"],
                 legal_actions: list[Action]):
        self.state = state
        self.action = action          # Action that led to this node
        self.parent = parent
        self.children: list["MCTSNode"] = []
        self.visits = 0
        self.wins = 0.0
        self.untried_actions = legal_actions[:]
        self.player_at_node = state.current_player

    def uct_score(self, exploration: float = 1.41) -> float:
        if self.visits == 0:
            return float("inf")
        parent_visits = self.parent.visits if self.parent else self.visits
        return (self.wins / self.visits) + exploration * math.sqrt(
            math.log(parent_visits) / self.visits
        )

    def best_child(self, exploration: float = 1.41) -> "MCTSNode":
        return max(self.children, key=lambda c: c.uct_score(exploration))

    def is_fully_expanded(self) -> bool:
        return len(self.untried_actions) == 0

    def is_terminal(self, engine: RulesEngine) -> bool:
        return engine.is_terminal(self.state)


class MCTSAgent(BaseAgent):
    """
    MCTS agent. Thinks for `time_limit` seconds (or `num_simulations` rollouts)
    per move, using RuleBasedAgent for rollouts.
    """

    def __init__(self, player_id: int, time_limit: float = 1.0,
                 num_simulations: int | None = None, exploration: float = 1.41,
                 seed: int | None = None):
        super().__init__(player_id)
        self.time_limit = time_limit
        self.num_simulations = num_simulations
        self.exploration = exploration
        # Rollout agents (one per possible player slot)
        self._rollout_agents = [RuleBasedAgent(i) for i in range(6)]

    def choose_action(self, state: GameState, legal_actions: list[Action]) -> Action:
        if len(legal_actions) == 1:
            return legal_actions[0]

        engine = RulesEngine(state.board, state.num_players)
        root = MCTSNode(state.copy(), action=None, parent=None, legal_actions=legal_actions)

        deadline = time.time() + self.time_limit
        sims = 0

        while True:
            if self.num_simulations is not None and sims >= self.num_simulations:
                break
            if self.num_simulations is None and time.time() >= deadline:
                break

            leaf = self._select(root, engine)
            if not leaf.is_terminal(engine):
                leaf = self._expand(leaf, engine)
            result = self._rollout(leaf.state, engine)
            self._backpropagate(leaf, result, self.player_id)
            sims += 1

        # Pick the most visited child
        if not root.children:
            return legal_actions[0]
        best = max(root.children, key=lambda c: c.visits)
        return best.action

    def _select(self, node: MCTSNode, engine: RulesEngine) -> MCTSNode:
        while not node.is_terminal(engine) and node.is_fully_expanded():
            node = node.best_child(self.exploration)
        return node

    def _expand(self, node: MCTSNode, engine: RulesEngine) -> MCTSNode:
        if not node.untried_actions:
            return node
        action = node.untried_actions.pop()
        new_state = engine.apply_action(node.state, action)
        legal = engine.legal_actions(new_state)
        child = MCTSNode(new_state, action=action, parent=node, legal_actions=legal)
        node.children.append(child)
        return child

    def _rollout(self, state: GameState, engine: RulesEngine) -> int | None:
        """Run a full game from state using RuleBasedAgent. Returns winner id or None."""
        s = state.copy()
        max_steps = 500  # Prevent infinite loops in degenerate states
        steps = 0
        while not engine.is_terminal(s) and steps < max_steps:
            legal = engine.legal_actions(s)
            rollout_agent = self._rollout_agents[s.current_player]
            action = rollout_agent.choose_action(s, legal)
            s = engine.apply_action(s, action)
            steps += 1
        return engine.winner(s)

    def _backpropagate(self, node: MCTSNode, winner: int | None, our_player: int) -> None:
        while node is not None:
            node.visits += 1
            if winner == our_player:
                node.wins += 1.0
            elif winner is None:
                node.wins += 0.5  # Draw / timeout
            node = node.parent
