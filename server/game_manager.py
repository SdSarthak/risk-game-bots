"""
In-memory game session manager.
Holds {game_id -> GameSession} for all active games.
"""
from __future__ import annotations

import pathlib
import uuid
from dataclasses import dataclass, field

from engine.board import BoardConfig
from engine.constants import Phase
from engine.rules import Action, RulesEngine
from engine.state import GameState
from agents.base import BaseAgent
from agents.random_agent import RandomAgent
from agents.rule_based import RuleBasedAgent
from agents.mcts import MCTSAgent
from server.schemas import (
    ActionOption, ActionRequest, GameCreateRequest, GameCreateResponse,
    GameStateResponse, GridInfo, LegalActionsResponse, PlayerState, TerritoryState
)

CONFIGS_DIR = pathlib.Path(__file__).parent.parent / "configs"

PHASE_BY_NAME = {phase.name: phase for phase in Phase}

# Upper bound on consecutive bot decisions served in one request, so an all-bot
# game can never spin the request thread forever.
MAX_BOT_STEPS_PER_CALL = 5_000


@dataclass
class GameSession:
    game_id: str
    board: BoardConfig
    state: GameState
    engine: RulesEngine
    agents: list[BaseAgent | None]   # agents[i] = agent for player i (None = human)
    player_types: list[str]
    ws_connections: list = field(default_factory=list)  # WebSocket connections

    @property
    def has_human(self) -> bool:
        return any(t == "human" for t in self.player_types)


class GameManager:
    """Holds every in-progress game in memory, keyed by game id."""

    def __init__(self, max_sessions: int = 200):
        self._sessions: dict[str, GameSession] = {}
        self._max_sessions = max_sessions

    def create_game(self, request: GameCreateRequest) -> tuple[str, GameSession]:
        config_path = CONFIGS_DIR / f"{request.board_config}.json"
        if not config_path.exists():
            raise ValueError(f"Unknown board config: {request.board_config}")

        board = BoardConfig.load(str(config_path))
        num_players = len(request.players)
        # Raises ValueError for player counts the board cannot seat
        state = GameState.new_game(board, num_players, seed=request.seed)
        engine = RulesEngine(board, num_players, seed=request.seed)

        agents: list[BaseAgent | None] = []
        player_types: list[str] = []
        for pid, player_cfg in enumerate(request.players):
            agents.append(self._build_agent(pid, player_cfg.type, player_cfg.checkpoint))
            player_types.append(player_cfg.type)

        game_id = str(uuid.uuid4())[:8]
        session = GameSession(
            game_id=game_id,
            board=board,
            state=state,
            engine=engine,
            agents=agents,
            player_types=player_types,
        )
        self._evict_if_full()
        self._sessions[game_id] = session

        # An all-bot game should already be under way when the client first polls
        if not session.has_human:
            self._run_bots(session)
        return game_id, session

    def _evict_if_full(self) -> None:
        """Drop the oldest sessions once the in-memory cap is reached."""
        while len(self._sessions) >= self._max_sessions:
            oldest = next(iter(self._sessions))
            del self._sessions[oldest]

    def get_session(self, game_id: str) -> GameSession | None:
        return self._sessions.get(game_id)

    def apply_human_action(self, game_id: str, action_req: ActionRequest) -> GameStateResponse:
        """Apply a human player's action and then run all subsequent bot turns."""
        session = self._sessions[game_id]

        if session.engine.is_terminal(session.state):
            raise ValueError("Game is already finished")

        current = session.state.current_player
        if session.player_types[current] != "human":
            raise ValueError(f"It is player {current}'s turn, and they are a bot")

        phase = PHASE_BY_NAME.get(action_req.phase.upper())
        if phase is None:
            raise ValueError(
                f"Unknown phase '{action_req.phase}'. "
                f"Expected one of {sorted(PHASE_BY_NAME)}"
            )

        action = Action(
            phase=phase,
            src=action_req.src,
            dst=action_req.dst,
            troops=action_req.troops,
        )

        # Action must be one the engine currently offers
        if action not in session.engine.legal_actions(session.state):
            raise ValueError(f"Illegal action: {action}")

        session.state = session.engine.apply_action(session.state, action)

        # Run bot turns until it's a human's turn or game over
        self._run_bots(session)

        return self.build_state_response(session)

    def step_bots(self, game_id: str) -> GameStateResponse:
        """Advance the game by running bot turns (used when no humans)."""
        session = self._sessions[game_id]
        self._run_bots(session)
        return self.build_state_response(session)

    def _run_bots(self, session: GameSession) -> None:
        """Run all bot turns until it's a human player's turn or game over."""
        steps = 0
        while not session.engine.is_terminal(session.state) and steps < MAX_BOT_STEPS_PER_CALL:
            player = session.state.current_player
            agent = session.agents[player]
            if agent is None or session.player_types[player] == "human":
                break  # Human's turn
            legal = session.engine.legal_actions(session.state)
            if not legal:
                break
            action = agent.choose_action(session.state, legal)
            session.state = session.engine.apply_action(session.state, action)
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
            from agents.checkpoints import find_latest_checkpoint
            from agents.rl_agent import RLAgent
            if checkpoint is None:
                resolved = find_latest_checkpoint()
                if resolved is None:
                    raise ValueError(
                        "No trained RL model found under checkpoints/. "
                        "Train one with: python training/run_training.py"
                    )
            else:
                resolved = pathlib.Path(checkpoint)
                if not resolved.exists():
                    raise ValueError(f"Checkpoint not found: {checkpoint}")
            return RLAgent.load(player_id, str(resolved))
        raise ValueError(f"Unknown agent type: {agent_type}")

    def build_state_response(self, session: GameSession) -> GameStateResponse:
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

        players = self._build_players(session)

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
            turn_number=state.turn_number,
            grid=grid_info,
        )

    def build_create_response(self, game_id: str) -> GameCreateResponse:
        session = self._sessions[game_id]
        return GameCreateResponse(
            game_id=game_id,
            board_name=session.board.name,
            num_territories=session.board.num_territories,
            players=self._build_players(session),
        )

    def build_legal_actions(self, session: GameSession) -> LegalActionsResponse:
        """Every action the player to move may legally take right now."""
        state = session.state
        if session.engine.is_terminal(state):
            actions: list[Action] = []
        else:
            actions = session.engine.legal_actions(state)
        return LegalActionsResponse(
            game_id=session.game_id,
            current_player=state.current_player,
            phase=state.phase.name,
            actions=[
                ActionOption(
                    phase=a.phase.name,
                    src=a.src,
                    dst=a.dst,
                    troops=a.troops,
                    end_phase=a.is_end_phase(),
                )
                for a in actions
            ],
        )

    def _build_players(self, session: GameSession) -> list[PlayerState]:
        state = session.state
        return [
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
