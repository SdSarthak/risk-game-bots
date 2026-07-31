"""Tests for the game engine core."""
import pytest

from engine.cards import CardDeck
from engine.constants import CARD_TRADE_BONUS_CAP, MAX_CARDS_IN_HAND, CardType, Phase
from engine.rules import Action, RulesEngine
from engine.state import MAX_PLAYERS, GameState


# ------------------------------------------------------------------
# BoardConfig tests
# ------------------------------------------------------------------

class TestBoardConfig:
    def test_small_board_loads(self, small_board):
        assert small_board.num_territories == 20

    def test_classic_board_loads(self, classic_board):
        assert classic_board.num_territories == 42

    def test_adjacency_symmetric(self, any_board):
        for tid, terr in any_board.territories.items():
            for adj in terr.adjacent:
                assert tid in any_board.territories[adj].adjacent, (
                    f"Territory {tid} lists {adj} as adjacent but not vice versa"
                )

    def test_no_self_adjacency(self, any_board):
        for tid, terr in any_board.territories.items():
            assert tid not in terr.adjacent

    def test_territory_ids_are_contiguous(self, any_board):
        assert set(any_board.territories) == set(range(any_board.num_territories))

    def test_continent_members_cover_all_territories(self, any_board):
        all_in_continents = set()
        for members in any_board.continent_members.values():
            all_in_continents.update(members)
        assert all_in_continents == set(range(any_board.num_territories))

    def test_every_continent_has_a_bonus(self, any_board):
        assert set(any_board.continent_members) == set(any_board.continent_bonuses)

    def test_territory_names_are_unique(self, any_board):
        names = [t.name for t in any_board.territories.values()]
        assert len(set(names)) == len(names)

    def test_classic_matches_the_real_risk_map(self, classic_board):
        sizes = {c: len(m) for c, m in classic_board.continent_members.items()}
        assert sizes == {
            "North America": 9, "South America": 4, "Europe": 7,
            "Africa": 6, "Asia": 12, "Australia": 4,
        }
        # The only continent-to-continent crossings on the real board
        assert set(classic_board.adjacent_to(0)) == {1, 3, 41}      # Alaska–Kamchatka
        assert 20 in classic_board.adjacent_to(11)                  # Brazil–North Africa
        assert 37 in classic_board.adjacent_to(26)                  # Siam–Indonesia

    def test_board_is_connected(self, any_board):
        """A disconnected board would make some territories unwinnable."""
        seen = {0}
        queue = [0]
        while queue:
            for nb in any_board.adjacent_to(queue.pop()):
                if nb not in seen:
                    seen.add(nb)
                    queue.append(nb)
        assert len(seen) == any_board.num_territories

    def test_are_adjacent(self, small_board):
        # Territory 0 adjacent to 1, 2, 4
        assert small_board.are_adjacent(0, 1)
        assert small_board.are_adjacent(0, 2)
        assert not small_board.are_adjacent(0, 19)


# ------------------------------------------------------------------
# GameState tests
# ------------------------------------------------------------------

class TestGameState:
    def test_new_game_all_territories_owned(self, small_game):
        state, _ = small_game
        assert all(o >= 0 for o in state.owners)

    def test_new_game_all_territories_have_troops(self, small_game):
        state, _ = small_game
        assert all(t >= 1 for t in state.troops)

    def test_territories_of_covers_all(self, small_game):
        state, _ = small_game
        all_owned = set()
        for p in range(state.num_players):
            all_owned.update(state.territories_of(p))
        assert all_owned == set(range(state.board.num_territories))

    def test_observation_shape(self, small_game):
        state, _ = small_game
        obs = state.to_observation()
        assert obs.shape == (state.obs_size,)
        assert obs.dtype.name == "float32"

    def test_observation_no_nan(self, small_game):
        state, _ = small_game
        import numpy as np
        obs = state.to_observation()
        assert not np.any(np.isnan(obs))

    def test_copy_independence(self, small_game):
        state, _ = small_game
        copy = state.copy()
        copy.troops[0] += 999
        assert state.troops[0] != 999

    def test_copy_does_not_share_hands(self, small_game):
        state, _ = small_game
        copy = state.copy()
        copy.cards[0].append(CardType.WILD)
        assert state.cards[0] == []

    def test_territories_dealt_evenly(self, any_board):
        state = GameState.new_game(any_board, num_players=2, seed=3)
        counts = [len(state.territories_of(p)) for p in range(2)]
        assert max(counts) - min(counts) <= 1

    def test_starting_armies_match_allotment(self, any_board):
        """Every player must field their full starting allotment, not one per territory."""
        for num_players in range(2, any_board.max_players + 1):
            state = GameState.new_game(any_board, num_players, seed=5)
            expected = GameState.initial_troops(num_players)
            for player in range(num_players):
                owned = len(state.territories_of(player))
                assert state.troop_count_of(player) == max(expected, owned)

    def test_every_territory_garrisoned(self, any_board):
        state = GameState.new_game(any_board, num_players=3, seed=8)
        assert all(t >= 1 for t in state.troops)

    def test_new_game_is_seed_reproducible(self, small_board):
        a = GameState.new_game(small_board, num_players=2, seed=11)
        b = GameState.new_game(small_board, num_players=2, seed=11)
        assert a.owners == b.owners
        assert a.troops == b.troops

    def test_first_player_gets_normal_reinforcements(self, small_game):
        state, _ = small_game
        assert state.troops_to_place == state.reinforcements_for(0)

    def test_reinforcements_have_a_floor_of_three(self, small_board):
        state = GameState.new_game(small_board, num_players=2, seed=1)
        state.owners = [0] + [1] * (small_board.num_territories - 1)
        assert state.reinforcements_for(0) == 3

    def test_reinforcements_include_continent_bonus(self, small_board):
        state = GameState.new_game(small_board, num_players=2, seed=1)
        continent, bonus = next(iter(small_board.continent_bonuses.items()))
        members = small_board.territories_in_continent(continent)
        state.owners = [1] * small_board.num_territories
        for tid in members:
            state.owners[tid] = 0
        assert state.reinforcements_for(0) == max(3, len(members) // 3) + bonus

    @pytest.mark.parametrize("num_players", [0, 1, MAX_PLAYERS + 1])
    def test_rejects_impossible_player_counts(self, small_board, num_players):
        with pytest.raises(ValueError):
            GameState.new_game(small_board, num_players=num_players, seed=0)

    def test_rejects_more_players_than_the_board_seats(self, small_board):
        with pytest.raises(ValueError):
            GameState.new_game(small_board, num_players=small_board.max_players + 1, seed=0)


# ------------------------------------------------------------------
# RulesEngine tests
# ------------------------------------------------------------------

class TestRulesEngine:
    def test_legal_draft_actions_nonempty(self, small_game):
        state, engine = small_game
        state.phase = Phase.DRAFT
        state.troops_to_place = 3
        actions = engine.legal_actions(state)
        assert len(actions) > 0
        assert all(a.phase == Phase.DRAFT for a in actions)

    def test_legal_draft_end_when_no_troops(self, small_game):
        state, engine = small_game
        state.phase = Phase.DRAFT
        state.troops_to_place = 0
        actions = engine.legal_actions(state)
        assert len(actions) == 1
        assert actions[0].is_end_phase()

    def test_draft_places_troops(self, small_game):
        state, engine = small_game
        state.phase = Phase.DRAFT
        state.troops_to_place = 5
        my_terr = state.territories_of(state.current_player)[0]
        before = state.troops[my_terr]
        action = Action(phase=Phase.DRAFT, dst=my_terr, troops=3)
        new_state = engine.apply_action(state, action)
        assert new_state.troops[my_terr] == before + 3
        assert new_state.troops_to_place == 2

    def test_draft_exhaustion_triggers_phase_change(self, small_game):
        state, engine = small_game
        state.phase = Phase.DRAFT
        state.troops_to_place = 0
        action = Action(phase=Phase.DRAFT, troops=-1)
        new_state = engine.apply_action(state, action)
        assert new_state.phase == Phase.ATTACK

    def test_legal_attack_includes_end_phase(self, small_game):
        state, engine = small_game
        state.phase = Phase.ATTACK
        actions = engine.legal_actions(state)
        assert any(a.is_end_phase() for a in actions)

    def test_attack_only_targets_enemy(self, small_game):
        state, engine = small_game
        state.phase = Phase.ATTACK
        actions = engine.legal_actions(state)
        attack_actions = [a for a in actions if not a.is_end_phase()]
        for a in attack_actions:
            assert state.owners[a.dst] != state.current_player

    def test_attack_requires_enough_troops(self, small_game):
        state, engine = small_game
        state.phase = Phase.ATTACK
        # Set all player 0 territories to 1 troop (minimum, can't attack)
        player = state.current_player
        for t in state.territories_of(player):
            state.troops[t] = 1
        actions = engine.legal_actions(state)
        attack_actions = [a for a in actions if not a.is_end_phase()]
        assert len(attack_actions) == 0

    def test_attack_dice_capped_at_3(self, small_game):
        state, engine = small_game
        state.phase = Phase.ATTACK
        # Give current player massive troops on one territory
        player = state.current_player
        my_terr = state.territories_of(player)[0]
        state.troops[my_terr] = 20
        actions = engine.legal_actions(state)
        for a in actions:
            if not a.is_end_phase() and a.src == my_terr:
                assert a.troops <= 3

    def test_fortify_end_advances_turn(self, small_game):
        state, engine = small_game
        state.phase = Phase.FORTIFY
        old_player = state.current_player
        action = Action(phase=Phase.FORTIFY, troops=-1)
        new_state = engine.apply_action(state, action)
        assert new_state.phase == Phase.DRAFT
        assert new_state.current_player != old_player
        assert new_state.turn_number == state.turn_number + 1

    def test_fortify_moves_troops_and_leaves_one_behind(self, small_game):
        state, engine = small_game
        state.phase = Phase.FORTIFY
        actions = [a for a in engine.legal_actions(state) if not a.is_end_phase()]
        move = max(actions, key=lambda a: a.troops)
        before_src, before_dst = state.troops[move.src], state.troops[move.dst]
        new_state = engine.apply_action(state, move)
        assert new_state.troops[move.src] == before_src - move.troops
        assert new_state.troops[move.dst] == before_dst + move.troops
        assert new_state.troops[move.src] >= 1

    def test_fortify_only_targets_connected_friendly_territory(self, small_game):
        state, engine = small_game
        state.phase = Phase.FORTIFY
        player = state.current_player
        for a in engine.legal_actions(state):
            if a.is_end_phase():
                continue
            assert state.owners[a.src] == player
            assert state.owners[a.dst] == player
            assert a.src != a.dst

    def test_apply_action_does_not_mutate_input(self, small_game):
        state, engine = small_game
        state.phase = Phase.DRAFT
        state.troops_to_place = 4
        before = list(state.troops)
        dst = state.territories_of(state.current_player)[0]
        engine.apply_action(state, Action(phase=Phase.DRAFT, dst=dst, troops=4))
        assert state.troops == before
        assert state.troops_to_place == 4

    def test_capture_transfers_ownership_and_troops(self, small_game):
        state, engine = small_game
        player = state.current_player
        state.phase = Phase.ATTACK
        src = state.territories_of(player)[0]
        dst = next(n for n in state.board.adjacent_to(src) if state.owners[n] != player)
        state.troops[src] = 20
        state.troops[dst] = 1
        # Roll until the single defender falls
        for _ in range(50):
            result = engine.apply_action(state, Action(phase=Phase.ATTACK, src=src,
                                                       dst=dst, troops=3))
            if result.owners[dst] == player:
                assert result.troops[dst] == 3
                assert result.troops[src] >= 1
                assert result.conquered_this_turn
                return
            state = result
        pytest.fail("A 20-troop stack never took a 1-troop territory in 50 rolls")

    def test_eliminating_a_player_transfers_their_cards(self, small_game):
        state, engine = small_game
        player = state.current_player
        victim = 1
        # Leave the victim a single territory next to a large stack of ours
        src = state.territories_of(player)[0]
        dst = next(n for n in state.board.adjacent_to(src) if n != src)
        state.owners = [player] * state.board.num_territories
        state.owners[dst] = victim
        state.troops[src] = 30
        state.troops[dst] = 1
        state.cards[victim] = [CardType.WILD, CardType.INFANTRY]
        state.phase = Phase.ATTACK

        for _ in range(50):
            state = engine.apply_action(state, Action(phase=Phase.ATTACK, src=src,
                                                      dst=dst, troops=3))
            if state.eliminated[victim]:
                assert state.cards[victim] == []
                assert set(state.cards[player]) == {CardType.WILD, CardType.INFANTRY}
                assert engine.is_terminal(state)
                return
        pytest.fail("A 30-troop stack never eliminated a 1-troop player in 50 rolls")

    def test_not_terminal_at_start(self, small_game):
        state, engine = small_game
        assert not engine.is_terminal(state)

    def test_terminal_when_one_player(self, small_game):
        state, engine = small_game
        # Eliminate all but player 0
        for t in range(state.board.num_territories):
            state.owners[t] = 0
        state.eliminated[1] = True
        assert engine.is_terminal(state)
        assert engine.winner(state) == 0

    def test_turn_order_skips_eliminated_players(self, small_board):
        state = GameState.new_game(small_board, num_players=3, seed=4)
        engine = RulesEngine(small_board, num_players=3, seed=4)
        state.phase = Phase.FORTIFY
        state.eliminated[1] = True
        after = engine.apply_action(state, Action(phase=Phase.FORTIFY, troops=-1))
        assert after.current_player == 2


# ------------------------------------------------------------------
# Card trading during play
# ------------------------------------------------------------------

class TestIsLegal:
    """
    `is_legal` exists so a submitted move can be validated without building the
    whole legal-action list, which is combinatorial in troop counts. It is only
    useful if it agrees with that list exactly, in both directions.
    """

    def _states(self, board):
        """A spread of real positions: every phase, several turns in."""
        import random as _random

        rng = _random.Random(4)
        engine = RulesEngine(board, 2, seed=4)
        state = GameState.new_game(board, num_players=2, seed=4)
        seen = []
        for _ in range(120):
            if engine.is_terminal(state):
                break
            seen.append(state)
            legal = engine.legal_actions(state)
            state = engine.apply_action(state, rng.choice(legal))
        return engine, seen

    def test_accepts_every_action_the_engine_offers(self, small_board):
        engine, states = self._states(small_board)
        for state in states:
            for action in engine.legal_actions(state):
                assert engine.is_legal(state, action), action

    def test_rejects_actions_from_other_states(self, small_board):
        """Actions legal elsewhere must not be waved through here."""
        engine, states = self._states(small_board)
        for i, state in enumerate(states):
            other = states[(i + 7) % len(states)]
            offered = set(engine.legal_actions(state))
            for action in engine.legal_actions(other):
                assert engine.is_legal(state, action) == (action in offered), action

    @pytest.mark.parametrize("bad", [
        Action(phase=Phase.ATTACK, src=-5, dst=0, troops=1),
        Action(phase=Phase.ATTACK, src=0, dst=999, troops=1),
        Action(phase=Phase.FORTIFY, src=0, dst=0, troops=1),
        Action(phase=Phase.DRAFT, dst=0, troops=0),
        Action(phase=Phase.DRAFT, dst=0, troops=-3),
        Action(phase=Phase.DRAFT, dst=999, troops=1),
    ])
    def test_rejects_malformed_actions_without_crashing(self, small_game, bad):
        state, engine = small_game
        for phase in Phase:
            state.phase = phase
            assert engine.is_legal(state, bad) is False

    def test_rejects_an_action_for_the_wrong_phase(self, small_game):
        state, engine = small_game
        state.phase = Phase.DRAFT
        assert not engine.is_legal(state, Action(phase=Phase.ATTACK, troops=-1))

    def test_rejects_fortifying_across_a_gap(self, small_board):
        """A fortify has to travel through friendly ground, not teleport."""
        state = GameState.new_game(small_board, num_players=2, seed=4)
        engine = RulesEngine(small_board, 2, seed=4)
        state.phase = Phase.FORTIFY
        state.current_player = 0
        state.owners = [1] * small_board.num_territories
        far = [t for t in range(small_board.num_territories)
               if t != 0 and 0 not in small_board.adjacent_to(t)]
        state.owners[0] = 0
        state.owners[far[-1]] = 0
        state.troops[0] = 5
        assert not engine.is_legal(
            state, Action(phase=Phase.FORTIFY, src=0, dst=far[-1], troops=2))

    def test_iter_matches_the_list(self, small_board):
        engine, states = self._states(small_board)
        for state in states:
            assert list(engine.iter_legal_actions(state)) == engine.legal_actions(state)

    def test_iter_is_lazy(self, small_board):
        """Taking a handful must not pay for the whole combinatorial list."""
        import itertools

        state = GameState.new_game(small_board, num_players=2, seed=4)
        engine = RulesEngine(small_board, 2, seed=4)
        state.phase = Phase.FORTIFY
        state.current_player = 0
        state.owners = [0] * small_board.num_territories
        state.troops = [400] * small_board.num_territories
        first = list(itertools.islice(engine.iter_legal_actions(state), 5))
        assert len(first) == 5
        assert all(engine.is_legal(state, a) for a in first)


class TestCardTrading:
    def _end_turn(self, engine, state):
        return engine.apply_action(state, Action(phase=Phase.FORTIFY, troops=-1))

    def test_conquest_earns_exactly_one_card(self, small_game):
        state, engine = small_game
        state.phase = Phase.ATTACK
        state.conquered_this_turn = True
        after = engine.apply_action(state, Action(phase=Phase.ATTACK, troops=-1))
        assert len(after.cards[state.current_player]) == 1
        assert not after.conquered_this_turn

    def test_no_conquest_earns_no_card(self, small_game):
        state, engine = small_game
        state.phase = Phase.ATTACK
        state.conquered_this_turn = False
        after = engine.apply_action(state, Action(phase=Phase.ATTACK, troops=-1))
        assert after.cards[state.current_player] == []

    def test_sets_are_cashed_in_at_the_start_of_a_draft(self, small_game):
        """The whole card mechanic is dead if nothing ever trades a set."""
        state, engine = small_game
        state.phase = Phase.FORTIFY
        state.cards[1] = [CardType.INFANTRY, CardType.CAVALRY, CardType.ARTILLERY]
        after = self._end_turn(engine, state)
        assert after.current_player == 1
        assert after.cards[1] == []
        assert after.card_trade_count == 1
        assert after.troops_to_place == after.reinforcements_for(1) + 4

    def test_incomplete_hand_is_not_traded(self, small_game):
        state, engine = small_game
        state.phase = Phase.FORTIFY
        state.cards[1] = [CardType.INFANTRY, CardType.INFANTRY]
        after = self._end_turn(engine, state)
        assert len(after.cards[1]) == 2
        assert after.card_trade_count == 0

    def test_lazy_engine_only_trades_over_the_hand_limit(self, small_board):
        state = GameState.new_game(small_board, num_players=2, seed=2)
        engine = RulesEngine(small_board, 2, seed=2, eager_card_trades=False)
        state.phase = Phase.FORTIFY
        state.cards[1] = [CardType.INFANTRY, CardType.CAVALRY, CardType.ARTILLERY]
        assert len(self._end_turn(engine, state).cards[1]) == 3

        state.cards[1] = [CardType.INFANTRY] * 3 + [CardType.CAVALRY] * 3
        after = self._end_turn(engine, state)
        assert len(after.cards[1]) == 3
        assert after.card_trade_count == 1

    def test_trade_bonus_escalates_across_players(self, small_game):
        state, engine = small_game
        state.phase = Phase.FORTIFY
        state.card_trade_count = 2  # third trade is worth 8
        state.cards[1] = [CardType.WILD, CardType.WILD, CardType.WILD]
        after = self._end_turn(engine, state)
        assert after.troops_to_place == after.reinforcements_for(1) + 8

    def test_hand_over_the_limit_is_always_brought_back_under_it(self, small_board):
        """
        The forced-trade rule is only enforceable if every over-limit hand has a
        set in it. A hand like infantry + cavalry + wild used to be judged
        untradeable, so a player could sit above the limit indefinitely.
        """
        import itertools
        engine = RulesEngine(small_board, 2, seed=3, eager_card_trades=False)
        for hand in itertools.combinations_with_replacement(list(CardType), 6):
            state = GameState.new_game(small_board, num_players=2, seed=3)
            state.cards[0] = list(hand)
            engine.auto_trade_cards(state, 0)
            assert len(state.cards[0]) <= MAX_CARDS_IN_HAND, hand

    def test_lazy_trading_stops_at_the_limit(self, small_board):
        engine = RulesEngine(small_board, 2, seed=3, eager_card_trades=False)
        state = GameState.new_game(small_board, num_players=2, seed=3)
        state.cards[0] = [CardType.INFANTRY, CardType.CAVALRY, CardType.WILD,
                          CardType.ARTILLERY, CardType.INFANTRY, CardType.CAVALRY]
        bonus = engine.auto_trade_cards(state, 0)
        assert len(state.cards[0]) == 3  # exactly one set cashed in
        assert bonus > 0

    def test_manual_trade_rejects_bad_indices(self, small_game):
        state, engine = small_game
        state.cards[0] = [CardType.INFANTRY, CardType.CAVALRY, CardType.ARTILLERY]
        with pytest.raises(ValueError):
            engine.trade_cards(state, [0, 1])
        with pytest.raises(ValueError):
            engine.trade_cards(state, [0, 0, 1])
        with pytest.raises(ValueError):
            engine.trade_cards(state, [0, 1, 9])

    def test_manual_trade_returns_a_new_state(self, small_game):
        state, engine = small_game
        state.cards[0] = [CardType.INFANTRY, CardType.CAVALRY, CardType.ARTILLERY]
        after = engine.trade_cards(state, [0, 1, 2])
        assert len(state.cards[0]) == 3
        assert after.cards[0] == []


# ------------------------------------------------------------------
# Dice simulation tests
# ------------------------------------------------------------------

class TestDiceSimulation:
    def test_dice_losses_sum_to_pairs(self, small_board):
        engine = RulesEngine(small_board, num_players=2, seed=0)
        for _ in range(100):
            atk_loss, def_loss = engine.simulate_attack(3, 2)
            assert atk_loss + def_loss == 2  # 2 pairs compared

    def test_dice_losses_in_range(self, small_board):
        engine = RulesEngine(small_board, num_players=2, seed=0)
        for _ in range(100):
            atk_loss, def_loss = engine.simulate_attack(3, 2)
            assert 0 <= atk_loss <= 2
            assert 0 <= def_loss <= 2

    def test_dice_1v1_always_one_loss(self, small_board):
        engine = RulesEngine(small_board, num_players=2, seed=0)
        for _ in range(100):
            atk_loss, def_loss = engine.simulate_attack(1, 1)
            assert atk_loss + def_loss == 1

    def test_seeded_engines_roll_identically(self, small_board):
        a = RulesEngine(small_board, num_players=2, seed=99)
        b = RulesEngine(small_board, num_players=2, seed=99)
        assert [a.simulate_attack(3, 2) for _ in range(20)] == \
               [b.simulate_attack(3, 2) for _ in range(20)]

    def test_attacker_edge_at_3v2_matches_theory(self, small_board):
        """
        At 3 dice against 2 the attacker takes 1.079 of the 2 losses on average
        (0.5396 per pair). A wrong comparison or tie-break shows up here.
        """
        engine = RulesEngine(small_board, num_players=2, seed=7)
        rolls = 20_000
        defender_losses = sum(engine.simulate_attack(3, 2)[1] for _ in range(rolls))
        assert 0.52 < defender_losses / (2 * rolls) < 0.56

    def test_ties_go_to_the_defender(self, small_board):
        """1v1 across many rolls: the defender wins ties, so it wins ~58%."""
        engine = RulesEngine(small_board, num_players=2, seed=13)
        rolls = 20_000
        attacker_losses = sum(engine.simulate_attack(1, 1)[0] for _ in range(rolls))
        assert 0.56 < attacker_losses / rolls < 0.61


# ------------------------------------------------------------------
# CardDeck tests
# ------------------------------------------------------------------

class TestCardDeck:
    def test_deck_size(self, small_board):
        deck = CardDeck(small_board.num_territories)
        drawn = [deck.draw() for _ in range(22)]  # 20 + 2 wilds
        assert len(drawn) == 22

    def test_reshuffle_on_empty(self, small_board):
        deck = CardDeck(small_board.num_territories, seed=1)
        cards = [deck.draw() for _ in range(22)]
        deck.discard(cards)
        # Should reshuffle and allow more draws
        card = deck.draw()
        assert isinstance(card, CardType)

    def test_find_valid_set_three_of_a_kind(self):
        hand = [CardType.INFANTRY, CardType.INFANTRY, CardType.INFANTRY]
        result = CardDeck.find_valid_set(hand)
        assert result is not None
        assert len(result) == 3

    def test_find_valid_set_one_each(self):
        hand = [CardType.INFANTRY, CardType.CAVALRY, CardType.ARTILLERY]
        result = CardDeck.find_valid_set(hand)
        assert result is not None
        assert len(result) == 3

    def test_find_valid_set_none_when_too_few(self):
        hand = [CardType.INFANTRY, CardType.CAVALRY]
        result = CardDeck.find_valid_set(hand)
        assert result is None

    def test_find_valid_set_none_for_two_of_a_kind_plus_odd_card(self):
        hand = [CardType.INFANTRY, CardType.INFANTRY, CardType.CAVALRY]
        assert CardDeck.find_valid_set(hand) is None

    def test_find_valid_set_uses_a_wild(self):
        hand = [CardType.INFANTRY, CardType.INFANTRY, CardType.WILD]
        result = CardDeck.find_valid_set(hand)
        assert result is not None
        assert sorted(result) == [0, 1, 2]

    def test_find_valid_set_returns_distinct_indices(self):
        hand = [CardType.WILD, CardType.WILD, CardType.CAVALRY, CardType.ARTILLERY]
        result = CardDeck.find_valid_set(hand)
        assert result is not None
        assert len(set(result)) == 3

    def test_find_valid_set_wild_completes_two_different_types(self):
        """A wild stands in for the missing third type, so this is a set."""
        hand = [CardType.INFANTRY, CardType.CAVALRY, CardType.WILD]
        result = CardDeck.find_valid_set(hand)
        assert result is not None
        assert sorted(result) == [0, 1, 2]

    @pytest.mark.parametrize("hand", [
        [CardType.INFANTRY, CardType.ARTILLERY, CardType.WILD],
        [CardType.CAVALRY, CardType.ARTILLERY, CardType.WILD],
        [CardType.WILD, CardType.CAVALRY, CardType.ARTILLERY],
        [CardType.INFANTRY, CardType.WILD, CardType.CAVALRY],
    ])
    def test_every_two_non_wilds_plus_a_wild_is_a_set(self, hand):
        result = CardDeck.find_valid_set(hand)
        assert result is not None
        assert len(set(result)) == 3

    def test_a_full_hand_always_yields_a_set(self):
        """
        Six cards over the hand limit must always be tradeable down; by the
        pigeonhole principle three of the four types force either a triple or
        one of each, and any wild completes whatever is left.
        """
        import itertools
        types = list(CardType)
        for hand in itertools.combinations_with_replacement(types, 6):
            assert CardDeck.find_valid_set(list(hand)) is not None, hand

    def test_found_indices_address_real_cards(self):
        hand = [CardType.WILD, CardType.INFANTRY, CardType.CAVALRY,
                CardType.INFANTRY, CardType.WILD]
        result = CardDeck.find_valid_set(hand)
        assert result is not None
        assert all(0 <= i < len(hand) for i in result)
        assert len(set(result)) == 3

    def test_trade_bonuses_escalate(self):
        bonuses = [CardDeck.bonus_for_trade(i) for i in range(8)]
        for i in range(len(bonuses) - 1):
            assert bonuses[i] < bonuses[i + 1]

    def test_trade_bonus_is_capped(self):
        """Uncapped escalation outgrows combat losses and games stop terminating."""
        assert CardDeck.bonus_for_trade(500) == CARD_TRADE_BONUS_CAP
        assert max(CardDeck.bonus_for_trade(i) for i in range(200)) == CARD_TRADE_BONUS_CAP

    def test_trade_bonus_rejects_negative_counts(self):
        with pytest.raises(ValueError):
            CardDeck.bonus_for_trade(-1)
