#!/usr/bin/env python3
"""
CLI script to run Risk games between agents.

Usage:
  python play_game.py --config small_20 --p1 rule_based --p2 random --games 10
  python play_game.py --config classic_42 --p1 mcts --p2 rule_based --games 1 --verbose
"""
import argparse
import pathlib
import sys
import time
from collections import defaultdict

from engine.board import BoardConfig
from engine.rules import RulesEngine
from engine.state import GameState
from agents.base import BaseAgent
from agents.random_agent import RandomAgent
from agents.rule_based import RuleBasedAgent
from agents.mcts import MCTSAgent

CONFIGS_DIR = pathlib.Path(__file__).parent / "configs"

CHECKPOINTS_DIR = pathlib.Path(__file__).parent / "checkpoints"

AGENT_TYPES = {
    "random": lambda pid: RandomAgent(pid),
    "rule_based": lambda pid: RuleBasedAgent(pid),
    "mcts": lambda pid: MCTSAgent(pid, time_limit=1.0),
    "rl": lambda pid: _load_rl_agent(pid),
}


def _load_rl_agent(player_id: int):
    from agents.rl_agent import RLAgent
    # Find the best model if available, otherwise the final
    best = CHECKPOINTS_DIR / "best" / "best_model.zip"
    final = CHECKPOINTS_DIR / "risk_ppo_final_500000.zip"
    # SB3 appends .zip automatically, check both
    if best.exists():
        path = str(best)
    elif final.exists():
        path = str(final)
    else:
        # Try without .zip suffix (SB3 adds it on load)
        path = str(CHECKPOINTS_DIR / "risk_ppo_final_500000")
    print(f"  Loading RL model from: {path}")
    return RLAgent.load(player_id, path)


def build_agent(agent_type: str, player_id: int) -> BaseAgent:
    if agent_type not in AGENT_TYPES:
        print(f"Unknown agent type '{agent_type}'. Choose from: {list(AGENT_TYPES)}")
        sys.exit(1)
    return AGENT_TYPES[agent_type](player_id)


def run_game(board: BoardConfig, agents: list[BaseAgent], seed: int,
             verbose: bool = False, max_turns: int = 1000) -> int | None:
    """
    Run a single game. Returns winner player id, or None if max_turns exceeded.
    """
    state = GameState.new_game(board, num_players=len(agents), seed=seed)
    engine = RulesEngine(board, num_players=len(agents), seed=seed)

    for agent in agents:
        agent.reset()

    # Set first player's draft to the normal reinforcement calculation
    state.troops_to_place = engine._calculate_reinforcements(state, state.current_player)

    turn = 0
    while not engine.is_terminal(state) and turn < max_turns:
        player = state.current_player
        agent = agents[player]
        legal = engine.legal_actions(state)

        if not legal:
            break

        action = agent.choose_action(state, legal)
        state = engine.apply_action(state, action)

        if verbose:
            print(f"Turn {turn:4d} | P{player} [{agent.__class__.__name__}] | {action}")

        turn += 1

    winner = engine.winner(state)
    if verbose:
        if winner is not None:
            print(f"\nWinner: Player {winner} ({agents[winner].__class__.__name__})")
        else:
            print(f"\nGame ended after {max_turns} turns with no winner (draw/timeout)")
        print(state)
    return winner


def main():
    parser = argparse.ArgumentParser(description="Run Risk games between agents.")
    parser.add_argument("--config", default="small_20",
                        help="Board config name (small_20 or classic_42)")
    parser.add_argument("--p1", default="rule_based",
                        help=f"Agent type for player 1: {list(AGENT_TYPES)}")
    parser.add_argument("--p2", default="random",
                        help=f"Agent type for player 2: {list(AGENT_TYPES)}")
    parser.add_argument("--p3", default=None,
                        help="Agent type for player 3 (optional)")
    parser.add_argument("--p4", default=None,
                        help="Agent type for player 4 (optional)")
    parser.add_argument("--games", type=int, default=10,
                        help="Number of games to run")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed for first game (increments per game)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print every action")
    parser.add_argument("--max-turns", type=int, default=2000,
                        help="Max turns before declaring a draw")
    args = parser.parse_args()

    config_path = CONFIGS_DIR / f"{args.config}.json"
    if not config_path.exists():
        print(f"Config not found: {config_path}")
        sys.exit(1)

    board = BoardConfig.load(str(config_path))

    # Build player list
    player_types = [args.p1, args.p2]
    if args.p3:
        player_types.append(args.p3)
    if args.p4:
        player_types.append(args.p4)

    agents = [build_agent(t, i) for i, t in enumerate(player_types)]
    num_players = len(agents)

    print(f"Board: {board.name} ({board.num_territories} territories)")
    print(f"Players: {', '.join(f'P{i}={t}' for i, t in enumerate(player_types))}")
    print(f"Running {args.games} games...\n")

    wins: dict[int, int] = defaultdict(int)
    draws = 0
    start = time.time()

    for g in range(args.games):
        verbose = args.verbose and (args.games == 1)
        winner = run_game(board, agents, seed=args.seed + g,
                          verbose=verbose, max_turns=args.max_turns)
        if winner is None:
            draws += 1
        else:
            wins[winner] += 1

        if not verbose and (g + 1) % max(1, args.games // 10) == 0:
            pct = (g + 1) / args.games * 100
            print(f"  {g+1}/{args.games} games ({pct:.0f}%)...")

    elapsed = time.time() - start
    print(f"\n--- Results ({args.games} games in {elapsed:.1f}s) ---")
    for pid, agent_type in enumerate(player_types):
        win_count = wins[pid]
        win_pct = win_count / args.games * 100
        print(f"  P{pid} ({agent_type}): {win_count} wins ({win_pct:.1f}%)")
    if draws:
        print(f"  Draws/timeouts: {draws} ({draws/args.games*100:.1f}%)")


if __name__ == "__main__":
    main()
