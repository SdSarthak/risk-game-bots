"""
In-memory game session manager.
Holds {game_id -> GameSession} for all active games.
"""
from __future__ import annotations

import pathlib
import uuid
from dataclasses import dataclass, field

from engine.board import BoardConfig
from engine.rules import Action, RulesEngine
from engine.state import GameState
from agents.base import BaseAgent
from agents.random_agent import RandomAgent
from agents.rule_based import RuleBasedAgent
from agents.mcts import MCTSAgent
from server.schemas import (
    ActionRequest, GameCreateRequest, GameCreateResponse,
    GameStateResponse, GridInfo, PlayerState, TerritoryState
)

CONFIGS_DIR = pathlib.Path(__file__).parent.parent / "configs"


@dataclass
class GameSession:
    game_id: str
    board: BoardConfig
    state: GameState
    engine: RulesEngine
    agents: list[BaseAgent]          # agents[i] = agent for player i (None = human)
    player_types: list[str]
    turn_number: int = 0
    ws_connections: list = field(default_factory=list)  # WebSocket connections


class GameManager:
    def __init__(self):
        self._sessions: dict[str, GameSession] = {}

    def create_game(self, request: GameCreateRequest) -> tuple[str, GameSession]:
        game_id = str(uuid.uuid4())[:8]
        board = BoardConfig.load(str(CONFIGS_DIR / f"{request.board_config}.json"))
        num_players = len(request.players)

        state = GameState.new_game(board, num_players, seed=None)
        engine = RulesEngine(board, num_players, seed=None)

        agents = []
        player_types = []
        for pid, player_cfg in enumerate(request.players):
            agent = self._build_agent(pid, player_cfg.type, player_cfg.checkpoint)
            agents.append(agent)
            player_types.append(player_cfg.type)

        session = GameSession(
            game_id=game_id,
            board=board,
            state=state,
            engine=engine,
            agents=agents,
            player_types=player_types,
        )
        self._sessions[game_id] = session
        return game_id, session

    def get_session(self, game_id: str) -> GameSession | None:
        return self._sessions.get(game_id)

    def apply_human_action(self, game_id: str, action_req: ActionRequest) -> GameStateResponse:
        """Apply a human player's action and then run all subsequent bot turns."""
        session = self._sessions[game_id]
        from engine.constants import Phase
        phase_map = {"DRAFT": Phase.DRAFT, "ATTACK": Phase.ATTACK, "FORTIFY": Phase.FORTIFY}
        action = Action(
            phase=phase_map[action_req.phase],
            src=action_req.src,
            dst=action_req.dst,
            troops=action_req.troops,
        )

        # Validate action is legal
        legal = session.engine.legal_actions(session.state)
        legal_reprs = [(a.phase, a.src, a.dst, a.troops) for a in legal]
        action_repr = (action.phase, action.src, action.dst, action.troops)
        if action_repr not in legal_reprs:
            raise ValueError(f"Illegal action: {action}")

        session.state = session.engine.apply_action(session.state, action)
        session.turn_number += 1

        # Run bot turns until it's a human's turn or game over
        self._run_bots(session)

        return self._build_response(session)

    def step_bots(self, game_id: str) -> GameStateResponse:
        """Advance the game by running bot turns (used when no humans)."""
        session = self._sessions[game_id]
        self._run_bots(session)
        return self._build_response(session)

    def _run_bots(self, session: GameSession) -> None:
        """Run all bot turns until it's a human player's turn or game over."""
        max_bot_steps = 500
        steps = 0
        while not session.engine.is_terminal(session.state) and steps < max_bot_steps:
            player = session.state.current_player
            agent = session.agents[player]
            if agent is None or session.player_types[player] == "human":
                break  # Human's turn
            legal = session.engine.legal_actions(session.state)
            action = agent.choose_action(session.state, legal)
            session.state = session.engine.apply_action(session.state, action)
            session.turn_number += 1
            steps += 1

    def _build_agent(self, player_id: int, agent_type: str,
                     checkpoint: str | None) -> BaseAgent | None:
        if agent_type == "human":
            return None
        if agent_type == "random":
            return RandomAgent(player_id)
        if agent_type == "rule_based":
            return RuleBasedAgent(player_id)
        if agent_type == "mcts":
            return MCTSAgent(player_id, time_limit=1.0)
        if agent_type == "rl":
            from agents.rl_agent import RLAgent
            # Use provided checkpoint or fall back to best/final model in checkpoints/
            if checkpoint is None:
                candidates = [
                    pathlib.Path(__file__).parent.parent / "checkpoints" / "best_model.pt",
                    pathlib.Path(__file__).parent.parent / "checkpoints" / "risk_ppo_final_1000000.pt",
                    pathlib.Path(__file__).parent.parent / "checkpoints" / "risk_ppo_final_500000.pt",
                ]
                checkpoint_path = next(
                    (str(p) for p in candidates if p.exists()), None
                )
                if checkpoint_path is None:
                    raise ValueError(
                        "No trained RL model found. Run: python training/run_training.py"
                    )
            else:
                checkpoint_path = checkpoint
            return RLAgent.load(player_id, checkpoint_path)
        raise ValueError(f"Unknown agent type: {agent_type}")

    def _build_response(self, session: GameSession) -> GameStateResponse:
        state = session.state
        engine = session.engine
        board = session.board

        territories = [
            TerritoryState(
                id=tid,
                name=t.name,
                continent=t.continent,
                owner=state.owners[tid],
                troops=state.troops[tid],
                adjacent=t.adjacent,
                row=t.row,
                col=t.col,
            )
            for tid, t in board.territories.items()
        ]

        players = [
            PlayerState(
                id=pid,
                type=session.player_types[pid],
                is_human=session.player_types[pid] == "human",
                eliminated=state.eliminated[pid],
                territory_count=len(state.territories_of(pid)),
                troop_count=state.troop_count_of(pid),
                card_count=len(state.cards[pid]),
            )
            for pid in range(state.num_players)
        ]

        grid_info = None
        if board.grid:
            grid_info = GridInfo(rows=board.grid["rows"], cols=board.grid["cols"])

        status = "finished" if engine.is_terminal(state) else "active"
        return GameStateResponse(
            game_id=session.game_id,
            status=status,
            winner=engine.winner(state),
            current_player=state.current_player,
            phase=state.phase.name,
            troops_to_place=state.troops_to_place,
            territories=territories,
            players=players,
            turn_number=session.turn_number,
            grid=grid_info,
        )

    def build_create_response(self, game_id: str) -> GameCreateResponse:
        session = self._sessions[game_id]
        state = session.state
        players = [
            PlayerState(
                id=pid,
                type=session.player_types[pid],
                is_human=session.player_types[pid] == "human",
                eliminated=False,
                territory_count=len(state.territories_of(pid)),
                troop_count=state.troop_count_of(pid),
                card_count=0,
            )
            for pid in range(state.num_players)
        ]
        return GameCreateResponse(
            game_id=game_id,
            board_name=session.board.name,
            num_territories=session.board.num_territories,
            players=players,
        )
