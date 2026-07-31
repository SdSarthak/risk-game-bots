"""
In-memory game session manager.
Holds {game_id -> GameSession} for all active games.
"""
from __future__ import annotations

import pathlib
import threading
import time
import uuid
from dataclasses import dataclass, field
from itertools import islice

from engine.board import BoardConfig, BoardConfigError
from engine.constants import Phase
from engine.rules import Action, RulesEngine
from engine.state import GameState
from agents.base import BaseAgent
from agents.random_agent import RandomAgent
from agents.rule_based import RuleBasedAgent
from agents.mcts import MCTSAgent
from server.schemas import (
    ActionOption, ActionRequest, DEFAULT_ACTION_LIMIT, GameCreateRequest,
    GameCreateResponse, GameStateResponse, GridInfo, LegalActionsResponse,
    MAX_ACTION_LIMIT, PlayerState, TerritoryState
)

CONFIGS_DIR = pathlib.Path(__file__).parent.parent / "configs"

PHASE_BY_NAME = {phase.name: phase for phase in Phase}

# Upper bound on consecutive bot decisions served in one request, so an all-bot
# game can never spin the request thread forever.
MAX_BOT_STEPS_PER_CALL = 5_000

# ...and an upper bound in seconds, because a decision is not O(1): a search
# agent thinking for a second a move would hold the thread for over an hour
# before the step ceiling above was anywhere near reached.
MAX_BOT_SECONDS_PER_CALL = 10.0

# Rollouts a server-side MCTS bot runs per decision. A wall-clock budget makes
# request latency a function of how fast the box is; a rollout budget does not.
SERVER_MCTS_SIMULATIONS = 30


@dataclass
class GameSession:
    game_id: str
    board: BoardConfig
    state: GameState
    engine: RulesEngine
    agents: list[BaseAgent | None]   # agents[i] = agent for player i (None = human)
    player_types: list[str]
    ws_connections: list = field(default_factory=list)  # WebSocket connections
    # FastAPI runs sync endpoints in a threadpool, so two requests for the same
    # game really do overlap. Without this, concurrent /action calls both read
    # the same state and the second write silently discards the first move.
    lock: threading.RLock = field(default_factory=threading.RLock)

    @property
    def has_human(self) -> bool:
        return any(t == "human" for t in self.player_types)


class GameManager:
    """Holds every in-progress game in memory, keyed by game id."""

    def __init__(self, max_sessions: int = 200):
        self._sessions: dict[str, GameSession] = {}
        self._max_sessions = max_sessions
        self._registry_lock = threading.Lock()

    def create_game(self, request: GameCreateRequest) -> tuple[str, GameSession]:
        config_path = CONFIGS_DIR / f"{request.board_config}.json"
        if not config_path.exists():
            raise ValueError(f"Unknown board config: {request.board_config}")

        try:
            board = BoardConfig.load(str(config_path))
        except BoardConfigError as exc:
            raise ValueError(f"Board config '{request.board_config}' is unusable: {exc}") from exc
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
        with self._registry_lock:
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
        with self._registry_lock:
            return self._sessions.get(game_id)

    def _require_session(self, game_id: str) -> GameSession:
        session = self.get_session(game_id)
        if session is None:
            raise KeyError(game_id)
        return session

    def apply_human_action(self, game_id: str, action_req: ActionRequest) -> GameStateResponse:
        """Apply a human player's action and then run all subsequent bot turns."""
        session = self._require_session(game_id)

        with session.lock:
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

            # Must be a move the engine currently offers. Checked directly
            # rather than by membership in legal_actions(), which is
            # combinatorial in troop counts and can run to ~10^5 entries.
            if not session.engine.is_legal(session.state, action):
                raise ValueError(f"Illegal action: {action}")

            session.state = session.engine.apply_action(session.state, action)

            # Run bot turns until it's a human's turn or game over
            self._run_bots(session)

            return self.build_state_response(session)

    def step_bots(self, game_id: str) -> GameStateResponse:
        """Advance the game by running bot turns (used when no humans)."""
        session = self._require_session(game_id)
        with session.lock:
            self._run_bots(session)
            return self.build_state_response(session)

    def _run_bots(self, session: GameSession) -> None:
        """
        Run bot turns until it's a human player's turn or the game is over.

        Bounded by both a decision count and a wall-clock budget: a search agent
        can spend real time on one move, so the step ceiling alone does not stop
        a request from occupying its thread for hours.
        """
        steps = 0
        deadline = time.monotonic() + MAX_BOT_SECONDS_PER_CALL
        with session.lock:
            while (not session.engine.is_terminal(session.state)
                   and steps < MAX_BOT_STEPS_PER_CALL
                   and time.monotonic() < deadline):
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
            return MCTSAgent(player_id, num_simulations=SERVER_MCTS_SIMULATIONS)
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
                if not resolved.is_file():
                    raise ValueError(f"Checkpoint not found: {checkpoint}")
            try:
                return RLAgent.load(player_id, str(resolved))
            except ValueError:
                raise
            except Exception as exc:  # noqa: BLE001 - torch raises many types
                raise ValueError(
                    f"Could not load RL checkpoint {resolved}: {exc}"
                ) from exc
        raise ValueError(f"Unknown agent type: {agent_type}")

    def build_state_response(self, session: GameSession) -> GameStateResponse:
        # Snapshot under the lock: a bot loop running in another thread swaps
        # `session.state` wholesale, and a response built across that swap would
        # mix territories from one turn with players from the next.
        with session.lock:
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

        players = self._build_players(session, state)

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
        session = self._require_session(game_id)
        return GameCreateResponse(
            game_id=game_id,
            board_name=session.board.name,
            num_territories=session.board.num_territories,
            players=self._build_players(session),
        )

    def build_legal_actions(self, session: GameSession,
                            limit: int = DEFAULT_ACTION_LIMIT) -> LegalActionsResponse:
        """
        The actions the player to move may take, capped at `limit`.

        The full list is combinatorial in troop counts: a late-game fortify
        phase on the classic board enumerates ~10^5 "move n troops from A to B"
        variants, which serialises to several megabytes. Actions are pulled from
        the engine's generator so the ones past the cap are never built at all.
        """
        limit = max(1, min(int(limit), MAX_ACTION_LIMIT))
        state = session.state
        with session.lock:
            if session.engine.is_terminal(state):
                actions: list[Action] = []
            else:
                # One extra tells us whether anything was left behind
                actions = list(islice(session.engine.iter_legal_actions(state), limit + 1))
        truncated = len(actions) > limit
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
                for a in actions[:limit]
            ],
            truncated=truncated,
            limit=limit,
        )

    def _build_players(self, session: GameSession,
                       state: GameState | None = None) -> list[PlayerState]:
        if state is None:
            with session.lock:
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
