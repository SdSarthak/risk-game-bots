# Risk Game Bots

A complete implementation of the board game Risk — engine, four kinds of bot,
a training pipeline, a REST/WebSocket API and a browser UI — built so that
agents can be pitted against each other and measured.

Everything runs locally. There are no accounts, API keys or downloads: the
boards are JSON files in this repo, and the only artifact that has to be
produced rather than cloned is the trained reinforcement-learning policy.

```
engine/     rules, board, cards, state          (numpy only)
agents/     random, rule-based, MCTS, RL        (torch only for RL)
training/   Gymnasium env, PPO trainer, reward shaping
server/     FastAPI app: REST + WebSocket
frontend/   React + Vite board UI
tests/      pytest suite, no downloads required
```

## Quick start

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

pytest                            # 162 tests, ~11 seconds
python play_game.py --config small_20 --p1 rule_based --p2 random --games 20
```

`play_game.py` runs headless matches and prints win rates:

```
Board: Small Risk (20 territories) (20 territories)
Players: P0=rule_based, P1=random
--- Results (20 games in 0.2s) ---
  P0 (rule_based): 20 wins (100.0%)
  P1 (random): 0 wins (0.0%)
```

Add `--p3`/`--p4` for three- and four-player games, `--verbose` with
`--games 1` to watch every decision, and `--seed` to reproduce a match exactly.

## Boards

| Config | Territories | Continents | Max players |
| --- | --- | --- | --- |
| `small_20` | 20 | 4 | 4 |
| `grid_6x6` | 36 | 4 | 4 |
| `classic_42` | 42 | 6 | 6 |

Boards are plain JSON under `configs/`; adding one is a matter of listing
territories, their adjacency and the continent bonuses. `tests/test_engine.py`
validates every config in the directory, so a new board is checked for
symmetric adjacency and complete continent coverage automatically.

## Rules implemented

Standard Risk, with the deal and placement collapsed into the opening state:

- Territories are dealt evenly at random, one army each, and the rest of each
  player's starting allotment (40/35/30/25/20 by player count) is scattered
  over the territories they hold.
- Each turn is **draft → attack → fortify**. Reinforcements are
  `max(3, territories // 3)` plus continent bonuses.
- Attacks roll up to three attacker dice against up to two defender dice,
  highest against highest, ties to the defender.
- Taking at least one territory in a turn earns a card. Sets are cashed in at
  the start of a draft for the escalating 4/6/8/10/12/15/+5 bonus. A player who
  is knocked out hands their cards to the attacker.
- Fortifying moves troops between any two connected friendly territories, once
  per turn, leaving at least one army behind.

Two deliberate departures from the physical game, both to keep self-play
terminating:

- The trade bonus is capped (`CARD_TRADE_BONUS_CAP`, default 30). Uncapped
  escalation outgrows combat losses, and games between competent bots stop
  being winnable by anyone — both sides just stockpile.
- Card sets are traded automatically rather than as a player decision, so
  agents need no trade action. Set `RulesEngine(..., eager_card_trades=False)`
  to trade only when a hand exceeds the five-card limit.

## The bots

| Agent | How it decides |
| --- | --- |
| `random` | Uniform over legal actions. The floor. |
| `rule_based` | Reinforces its most threatened border, attacks when it holds a 1.5:1 advantage, prefers captures that complete a continent, fortifies interior troops to its weakest border. |
| `mcts` | UCT with a time budget per move, using `rule_based` as the rollout policy. |
| `rl` | PPO actor-critic (`agents/neural_net.py`) trained by self-play against a fixed opponent. |

Benchmark them round-robin, with each pairing split evenly between seats:

```bash
python benchmark.py --games 20                    # all agents
python benchmark.py --games 40 --no-mcts          # fast, MCTS is the slow one
python benchmark.py --config classic_42 --games 10
```

## Training the RL agent

No checkpoint ships with this repo — weights are build output, not source. To
produce one:

```bash
python training/run_training.py --config small_20 --timesteps 500000
```

Checkpoints land in `checkpoints/` as `.pt` files every `--save-interval`
steps. `play_game.py --p1 rl`, `benchmark.py` and the API all pick up the
newest one automatically (`agents/checkpoints.py`), or you can name one:

```bash
python play_game.py --p1 rl --p2 rule_based --games 20
python benchmark.py --checkpoint checkpoints/risk_ppo_final_500000.pt
```

Useful flags: `--opponent {random,rule_based,mcts}` picks who the policy trains
against, `--num-players` scales the game, `--hidden`/`--layers` size the
network, and `--checkpoint` resumes an interrupted run.

The observation is a flat vector: per-territory ownership from the mover's
point of view, per-territory troop share, whose turn it is, the phase, and the
mover's hand. Rewards are sparse win/loss shaped with a potential function over
territory, continent and troop share — potential-based shaping (Ng et al. 1999)
leaves the optimal policy unchanged.

## Running the web app

Two processes. Backend:

```bash
cp .env.example .env              # optional; sane defaults are built in
uvicorn server.main:app --reload --port 8000
```

Frontend:

```bash
cd frontend
npm install
cp .env.example .env.local        # optional; defaults to http://localhost:8000
npm run dev                       # http://localhost:5173
```

Pick a board and an opponent, and you play as player 0. Interactive docs are at
`http://localhost:8000/docs`.

| Endpoint | Purpose |
| --- | --- |
| `POST /games` | Create a game from a board config and a player list |
| `GET /games/{id}` | Current state: territories, troops, players, phase |
| `GET /games/{id}/legal-actions` | Every action the player to move may take |
| `POST /games/{id}/action` | Submit a human action (validated against the engine) |
| `POST /games/{id}/step` | Run pending bot turns (all-bot games) |
| `WS /ws/{id}` | State pushed on connect and on any client message |

Games live in the server process only; restarting drops them.

### Configuration

Nothing here is secret — the app needs no credentials.

| Variable | Where | Default |
| --- | --- | --- |
| `RISK_CORS_ORIGINS` | backend | `http://localhost:5173,http://localhost:3000` |
| `RISK_MAX_SESSIONS` | backend | `200` |
| `VITE_API_BASE` | frontend | `http://localhost:8000` |

## Tests

```bash
pytest                      # everything
pytest tests/test_engine.py # rules, cards and dice
pytest tests/test_agents.py # agent behaviour and the checkpoint locator
pytest tests/test_server.py # REST and WebSocket, in-process
```

The suite is deterministic and needs no checkpoint, no network and no dataset.
`tests/test_engine.py` runs its board checks against every config in
`configs/`, so a new board is validated the moment it is added.

## What is not committed

`checkpoints/`, `logs/` and `frontend/dist/` are build output and stay out of
git. Train a policy to recreate `checkpoints/`; the boards and code are all the
input anything else needs.
