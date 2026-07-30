"""Tests for the game engine core."""
import pathlib
import pytest

from engine.board import BoardConfig
from engine.constants import Phase
from engine.rules import Action, RulesEngine
from engine.state import GameState
from engine.cards import CardDeck
from engine.constants import CardType

CONFIGS_DIR = pathlib.Path(__file__).parent.parent / "configs"
SMALL_CONFIG = str(CONFIGS_DIR / "small_20.json")
CLASSIC_CONFIG = str(CONFIGS_DIR / "classic_42.json")


@pytest.fixture
def small_board():
    return BoardConfig.load(SMALL_CONFIG)


@pytest.fixture
def classic_board():
    return BoardConfig.load(CLASSIC_CONFIG)


@pytest.fixture
def small_game(small_board):
    state = GameState.new_game(small_board, num_players=2, seed=42)
    engine = RulesEngine(small_board, num_players=2, seed=42)
    return state, engine


# ------------------------------------------------------------------
# BoardConfig tests
# ------------------------------------------------------------------

class TestBoardConfig:
    def test_small_board_loads(self, small_board):
        assert small_board.num_territories == 20

    def test_classic_board_loads(self, classic_board):
        assert classic_board.num_territories == 42

    def test_adjacency_symmetric(self, small_board):
        for tid, terr in small_board.territories.items():
            for adj in terr.adjacent:
                assert tid in small_board.territories[adj].adjacent, (
                    f"Territory {tid} lists {adj} as adjacent but not vice versa"
                )

    def test_continent_members_cover_all_territories(self, small_board):
        all_in_continents = set()
        for members in small_board.continent_members.values():
            all_in_continents.update(members)
        assert all_in_continents == set(range(small_board.num_territories))

    def test_continent_bonuses_match_continent_members(self, small_board):
        for continent in small_board.continent_bonuses:
            assert continent in small_board.continent_members

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


# ------------------------------------------------------------------
# Dice simulation tests
# ------------------------------------------------------------------

class TestDiceSimulation:
    def test_dice_losses_sum_to_pairs(self):
        board = BoardConfig.load(str(CONFIGS_DIR / "small_20.json"))
        engine = RulesEngine(board, num_players=2, seed=0)
        for _ in range(100):
            atk_loss, def_loss = engine.simulate_attack(3, 2)
            assert atk_loss + def_loss == 2  # 2 pairs compared

    def test_dice_losses_in_range(self):
        board = BoardConfig.load(str(CONFIGS_DIR / "small_20.json"))
        engine = RulesEngine(board, num_players=2, seed=0)
        for _ in range(100):
            atk_loss, def_loss = engine.simulate_attack(3, 2)
            assert 0 <= atk_loss <= 2
            assert 0 <= def_loss <= 2

    def test_dice_1v1_always_one_loss(self):
        board = BoardConfig.load(str(CONFIGS_DIR / "small_20.json"))
        engine = RulesEngine(board, num_players=2, seed=0)
        for _ in range(100):
            atk_loss, def_loss = engine.simulate_attack(1, 1)
            assert atk_loss + def_loss == 1


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

    def test_trade_bonuses_escalate(self):
        bonuses = [CardDeck.bonus_for_trade(i) for i in range(8)]
        for i in range(len(bonuses) - 1):
            assert bonuses[i] < bonuses[i + 1]
