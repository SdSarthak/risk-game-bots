#!/usr/bin/env python3
"""
Benchmark all agents against each other and print a leaderboard.

Usage:
  python benchmark.py
  python benchmark.py --games 100 --config classic_42
"""
import argparse
import pathlib
import sys
import time
from collections import defaultdict
from itertools import permutations

sys.path.insert(0, str(pathlib.Path(__file__).parent))

from engine.board import BoardConfig
from engine.rules import RulesEngine
from engine.state import GameState
from agents.random_agent import RandomAgent
from agents.rule_based import RuleBasedAgent
from agents.mcts import MCTSAgent

CONFIGS_DIR = pathlib.Path(__file__).parent / "configs"
DEFAULT_MAX_STEPS = 50_000


def load_agents(rl_checkpoint: pathlib.Path | None, include_mcts: bool) -> dict:
    agents = {
        "random":     lambda pid: RandomAgent(pid),
        "rule_based": lambda pid: RuleBasedAgent(pid),
    }
    if include_mcts:
        agents["mcts"] = lambda pid: MCTSAgent(pid, time_limit=0.5)
    if rl_checkpoint is not None:
        from agents.rl_agent import RLAgent
        path = str(rl_checkpoint)
        agents["rl"] = lambda pid, p=path: RLAgent.load(pid, p)
    return agents


def run_game(board, agents_list, seed, max_steps=DEFAULT_MAX_STEPS):
    """Play one game and return the winner id (None if the step budget ran out)."""
    state = GameState.new_game(board, num_players=len(agents_list), seed=seed)
    engine = RulesEngine(board, num_players=len(agents_list), seed=seed)

    for agent in agents_list:
        agent.reset()

    for _ in range(max_steps):
        if engine.is_terminal(state):
            break
        legal = engine.legal_actions(state)
        if not legal:
            break
        action = agents_list[state.current_player].choose_action(state, legal)
        state = engine.apply_action(state, action)

    return engine.winner(state)


def run_matchup(board, name_a, factory_a, name_b, factory_b, games, seed_offset,
                max_steps=DEFAULT_MAX_STEPS):
    wins = {name_a: 0, name_b: 0, "draw": 0}
    # Play half as P0, half as P1 to cancel positional bias
    half = games // 2
    for g in range(half):
        agents = [factory_a(0), factory_b(1)]
        w = run_game(board, agents, seed=seed_offset + g, max_steps=max_steps)
        if w == 0:     wins[name_a] += 1
        elif w == 1:   wins[name_b] += 1
        else:          wins["draw"] += 1
    for g in range(games - half):
        agents = [factory_b(0), factory_a(1)]
        w = run_game(board, agents, seed=seed_offset + half + g, max_steps=max_steps)
        if w == 1:     wins[name_a] += 1
        elif w == 0:   wins[name_b] += 1
        else:          wins["draw"] += 1
    return wins


def main():
    parser = argparse.ArgumentParser(
        description="Round-robin all agents against each other and print a leaderboard."
    )
    parser.add_argument("--config", default="small_20",
                        choices=sorted(p.stem for p in CONFIGS_DIR.glob("*.json")))
    parser.add_argument("--games", type=int, default=20,
                        help="Games per matchup (split evenly as P0/P1)")
    parser.add_argument("--no-mcts", action="store_true", help="Skip MCTS (slow)")
    parser.add_argument("--no-rl",   action="store_true", help="Skip RL agent")
    parser.add_argument("--checkpoint", default=None,
                        help="RL checkpoint to benchmark (default: newest under checkpoints/)")
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS,
                        help="Max agent decisions per game before scoring it a draw")
    args = parser.parse_args()

    if args.games < 2:
        print("--games must be at least 2 so each agent plays both seats")
        sys.exit(1)

    board = BoardConfig.load(str(CONFIGS_DIR / f"{args.config}.json"))

    rl_checkpoint = None
    if not args.no_rl:
        from agents.checkpoints import find_latest_checkpoint
        rl_checkpoint = (pathlib.Path(args.checkpoint) if args.checkpoint
                         else find_latest_checkpoint())
        if rl_checkpoint is None or not rl_checkpoint.exists():
            print("No RL checkpoint found — skipping RL agent.")
            print("Train first with: python training/run_training.py --config small_20\n")
            rl_checkpoint = None

    agents = load_agents(rl_checkpoint=rl_checkpoint, include_mcts=not args.no_mcts)
    names = list(agents.keys())

    print(f"Board:   {board.name}")
    print(f"Agents:  {', '.join(names)}")
    print(f"Games:   {args.games} per matchup ({args.games // 2} as P0, {args.games - args.games // 2} as P1)")
    print(f"Matchups: {len(names) * (len(names) - 1) // 2}")
    print()

    # Track ELO-style win counts
    total_wins = defaultdict(int)
    total_games = defaultdict(int)
    results_table = {}

    name_pairs = [(names[i], names[j]) for i in range(len(names)) for j in range(i+1, len(names))]
    total_matchups = len(name_pairs)

    for idx, (a, b) in enumerate(name_pairs):
        print(f"[{idx+1}/{total_matchups}] {a} vs {b} ... ", end="", flush=True)
        t0 = time.time()
        result = run_matchup(board, a, agents[a], b, agents[b],
                             games=args.games, seed_offset=idx * 1000,
                             max_steps=args.max_steps)
        elapsed = time.time() - t0

        wa, wb = result[a], result[b]
        draws = result["draw"]
        total_wins[a] += wa
        total_wins[b] += wb
        total_games[a] += args.games
        total_games[b] += args.games
        results_table[(a, b)] = (wa, wb, draws)

        print(f"{a} {wa}–{wb} {b}  (draws: {draws})  [{elapsed:.1f}s]")

    # --- Results table ---
    print()
    print("=" * 60)
    print("HEAD-TO-HEAD RESULTS")
    print("=" * 60)
    col_w = max(len(n) for n in names) + 2
    header = f"{'':>{col_w}}" + "".join(f"{n:>{col_w}}" for n in names)
    print(header)
    for a in names:
        row = f"{a:>{col_w}}"
        for b in names:
            if a == b:
                row += f"{'—':>{col_w}}"
            elif (a, b) in results_table:
                wa, wb, _ = results_table[(a, b)]
                row += f"{wa:>{col_w}}"
            else:
                wa, wb, _ = results_table[(b, a)]
                row += f"{wb:>{col_w}}"
        print(row)

    # --- Leaderboard ---
    print()
    print("=" * 60)
    print("LEADERBOARD  (win rate across all matchups)")
    print("=" * 60)
    ranked = sorted(names, key=lambda n: total_wins[n] / max(1, total_games[n]), reverse=True)
    for rank, name in enumerate(ranked, 1):
        wr = total_wins[name] / max(1, total_games[name]) * 100
        print(f"  #{rank}  {name:<14}  {total_wins[name]:>3}W / {total_games[name]:>3}G  ({wr:.1f}%)")
    print()


if __name__ == "__main__":
    main()
