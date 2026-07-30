"""
Potential-based reward shaping for Risk RL training.

Using potential-based shaping (Ng et al. 1999) guarantees that the optimal
policy under shaped rewards is the same as under sparse win/loss rewards.

Shaped reward: r'(s, a, s') = r(s, a, s') + gamma * Phi(s') - Phi(s)

Where Phi(s) is a "potential" function measuring how good state s is.
"""
from engine.state import GameState


def potential(state: GameState, player: int) -> float:
    """
    State potential for a given player. Higher = better position.
    Components:
      - Territory control (fraction of total territories owned)
      - Continent bonuses controlled
      - Troop ratio
    """
    n = state.board.num_territories
    my_territories = len(state.territories_of(player))
    territory_frac = my_territories / n  # [0, 1]

    continent_score = sum(
        bonus
        for continent, bonus in state.board.continent_bonuses.items()
        if state.controls_continent(player, continent)
    )
    max_continent_bonus = sum(state.board.continent_bonuses.values())
    continent_frac = continent_score / max(1, max_continent_bonus)

    total_troops = sum(state.troops)
    my_troops = state.troop_count_of(player)
    troop_frac = my_troops / max(1, total_troops)

    # Weighted sum
    return 0.4 * territory_frac + 0.3 * continent_frac + 0.3 * troop_frac


class RewardShaper:
    """
    Computes shaped reward for RL training.

    Usage:
        shaper = RewardShaper(player_id=0, gamma=0.99)
        shaper.reset(initial_state)
        ...
        r = shaper.step(new_state, win=False, eliminated=False)
    """

    WIN_REWARD = 1.0
    LOSS_REWARD = -1.0

    def __init__(self, player_id: int, gamma: float = 0.99, shaping_weight: float = 0.3):
        self.player_id = player_id
        self.gamma = gamma
        self.shaping_weight = shaping_weight
        self._prev_potential = 0.0

    def reset(self, state: GameState) -> None:
        self._prev_potential = potential(state, self.player_id)

    def step(self, state: GameState, win: bool, eliminated: bool,
             truncated: bool = False) -> float:
        """Compute shaped reward after transitioning to `state`."""
        # Sparse terminal reward
        if win:
            return self.WIN_REWARD
        if eliminated:
            return self.LOSS_REWARD
        if truncated:
            return self.truncation_reward(state)

        # Potential-based shaping
        new_potential = potential(state, self.player_id)
        shaped = self.shaping_weight * (self.gamma * new_potential - self._prev_potential)
        self._prev_potential = new_potential
        return shaped

    def truncation_reward(self, state: GameState) -> float:
        """
        Payout for an episode that hit its step limit without a winner.

        Leaving truncation unrewarded makes stalling strictly better than
        fighting and losing, and a policy will find that: it stops attacking,
        runs out the clock and never collects the -1. Scoring the position it
        reached removes the dodge — a player behind on the board still loses.
        """
        share = potential(state, self.player_id)
        return self.LOSS_REWARD + (self.WIN_REWARD - self.LOSS_REWARD) * share
