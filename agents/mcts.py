"""
Monte Carlo Tree Search agent for Risk.

Uses UCT (Upper Confidence bounds for Trees) with:
- Dice randomness resolved by the engine during rollout, so a node's statistics
  average over outcomes rather than assuming a fixed one
- RuleBasedAgent as the rollout policy (much stronger signal than random)
- Max-n backup for multi-player games: each node scores the win rate of whoever
  chose the move leading into it, so opponents are modelled as playing for
  themselves rather than helping us
"""
from __future__ import annotations

import math
import time
from typing import Optional

from agents.base import BaseAgent
from agents.rule_based import RuleBasedAgent
from engine.rules import Action, RulesEngine
from engine.state import MAX_PLAYERS, GameState


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
    MCTS agent, using RuleBasedAgent for rollouts.

    Budget: `num_simulations` rollouts per move when set, otherwise `time_limit`
    seconds. Prefer the rollout budget when the run has to be reproducible or
    bounded — wall-clock budgets make a benchmark's runtime unpredictable.
    """

    DEFAULT_ROLLOUT_DEPTH = 40

    def __init__(self, player_id: int, time_limit: float = 1.0,
                 num_simulations: int | None = None, exploration: float = 1.41,
                 rollout_depth: int = DEFAULT_ROLLOUT_DEPTH,
                 seed: int | None = None):
        super().__init__(player_id)
        if num_simulations is not None and num_simulations < 1:
            raise ValueError("num_simulations must be at least 1")
        if rollout_depth < 1:
            raise ValueError("rollout_depth must be at least 1")
        self.time_limit = time_limit
        self.num_simulations = num_simulations
        self.exploration = exploration
        self.rollout_depth = rollout_depth
        self.seed = seed
        # Rollout agents (one per possible player slot)
        self._rollout_agents = [RuleBasedAgent(i) for i in range(MAX_PLAYERS)]
        self._engine: RulesEngine | None = None
        self._engine_board = None

    def reset(self) -> None:
        self._engine = None
        self._engine_board = None

    def _engine_for(self, state: GameState) -> RulesEngine:
        """
        Search engine for the current board.

        Kept between moves so the seed actually determines the search: a fresh
        engine per decision would reset the dice stream and rebuild the deck on
        every call.
        """
        if self._engine is None or self._engine_board is not state.board:
            self._engine = RulesEngine(state.board, state.num_players, seed=self.seed)
            self._engine_board = state.board
        return self._engine

    def choose_action(self, state: GameState, legal_actions: list[Action]) -> Action:
        if len(legal_actions) == 1:
            return legal_actions[0]

        engine = self._engine_for(state)
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
            outcome = self._rollout(leaf.state, engine)
            self._backpropagate(leaf, outcome, engine)
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

    def _rollout(self, state: GameState, engine: RulesEngine) -> GameState:
        """
        Play out at most `rollout_depth` decisions with the heuristic policy.

        Risk games run to hundreds of decisions, so playing every rollout to a
        winner costs thousands of engine steps per simulation and puts a full
        benchmark into the hours. A truncated rollout scored by `evaluate`
        gives the same ranking signal for a fraction of the work.
        """
        s = state.copy()
        for _ in range(self.rollout_depth):
            if engine.is_terminal(s):
                break
            legal = engine.legal_actions(s)
            if not legal:
                break
            rollout_agent = self._rollout_agents[s.current_player]
            s = engine.apply_action(s, rollout_agent.choose_action(s, legal))
        return s

    @staticmethod
    def evaluate(state: GameState, player: int) -> float:
        """
        How good `state` looks for `player`, in [0, 1].

        Territory share dominates because it drives reinforcements; troop share
        breaks ties between positions holding the same ground.
        """
        territory_share = (len(state.territories_of(player))
                           / max(1, state.board.num_territories))
        troop_share = state.troop_count_of(player) / max(1, sum(state.troops))
        return 0.6 * territory_share + 0.4 * troop_share

    def _backpropagate(self, node: MCTSNode, final_state: GameState,
                       engine: RulesEngine) -> None:
        """
        Credit each node to the player who chose the move leading into it.

        Scoring every node by *our* win rate would have opponents selecting the
        moves that help us most, which is the opposite of what they will do.
        A rollout that ended in a win scores 1 or 0; a truncated one scores the
        position it reached.
        """
        winner = engine.winner(final_state)
        while node is not None:
            node.visits += 1
            if node.parent is not None:
                mover = node.parent.player_at_node
                if winner is not None:
                    node.wins += 1.0 if winner == mover else 0.0
                else:
                    node.wins += self.evaluate(final_state, mover)
            node = node.parent
