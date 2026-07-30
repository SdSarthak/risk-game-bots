"""
Tests for the fixed action encoding.

The property that matters is that a masked index always decodes to an action the
engine considers legal — otherwise a trained policy makes moves the rules reject.
"""
import numpy as np
import pytest

from agents.action_space import (
    ACTION_SPACE_SIZE,
    END_PHASE_INDEX,
    MAX_ADJACENT,
    RiskActionSpace,
)
from engine.constants import Phase
from engine.rules import RulesEngine
from engine.state import GameState


@pytest.fixture
def encoder(small_board):
    return RiskActionSpace(small_board)


class TestLayout:
    def test_every_board_fits(self, any_board):
        RiskActionSpace(any_board)  # must not raise

    def test_indices_do_not_collide(self, encoder, small_board):
        seen = set()
        for t in range(small_board.num_territories):
            seen.add(encoder.draft_index(t))
            for k in range(MAX_ADJACENT):
                seen.add(encoder.attack_index(t, k))
                seen.add(encoder.fortify_index(t, k))
        seen.add(END_PHASE_INDEX)
        expected = small_board.num_territories * (1 + 2 * MAX_ADJACENT) + 1
        assert len(seen) == expected
        assert max(seen) < ACTION_SPACE_SIZE

    def test_rejects_a_board_with_too_many_neighbours(self, small_board):
        crowded = small_board.territories[0]
        crowded.adjacent = list(range(1, MAX_ADJACENT + 3))
        with pytest.raises(ValueError, match="neighbours"):
            RiskActionSpace(small_board)


class TestDecoding:
    def test_draft_commits_the_whole_allotment(self, encoder, small_game):
        state, _ = small_game
        state.phase = Phase.DRAFT
        state.troops_to_place = 7
        owned = state.territories_of(state.current_player)[0]
        action = encoder.decode(encoder.draft_index(owned), state)
        assert action.phase is Phase.DRAFT
        assert action.dst == owned
        assert action.troops == 7

    def test_draft_on_an_enemy_territory_is_rejected(self, encoder, small_game):
        state, _ = small_game
        state.phase = Phase.DRAFT
        state.troops_to_place = 3
        enemy = next(t for t in range(state.board.num_territories)
                     if state.owners[t] != state.current_player)
        assert encoder.decode(encoder.draft_index(enemy), state) is None

    def test_attack_targets_the_named_neighbour(self, encoder, small_game):
        state, _ = small_game
        state.phase = Phase.ATTACK
        player = state.current_player
        src = next(t for t in state.territories_of(player)
                   if any(state.owners[n] != player
                          for n in state.board.adjacent_to(t)))
        state.troops[src] = 5
        slot = next(i for i, n in enumerate(state.board.adjacent_to(src))
                    if state.owners[n] != player)
        action = encoder.decode(encoder.attack_index(src, slot), state)
        assert action.src == src
        assert action.dst == state.board.adjacent_to(src)[slot]
        assert action.troops == 3, "should roll the maximum dice"

    def test_attack_from_a_single_garrison_is_rejected(self, encoder, small_game):
        state, _ = small_game
        state.phase = Phase.ATTACK
        player = state.current_player
        src = state.territories_of(player)[0]
        state.troops[src] = 1
        for slot in range(len(state.board.adjacent_to(src))):
            assert encoder.decode(encoder.attack_index(src, slot), state) is None

    def test_fortify_leaves_one_army_behind(self, encoder, small_game):
        state, _ = small_game
        state.phase = Phase.FORTIFY
        player = state.current_player
        pair = next(((t, i) for t in state.territories_of(player)
                     for i, n in enumerate(state.board.adjacent_to(t))
                     if state.owners[n] == player), None)
        if pair is None:
            pytest.skip("this deal left the player no two adjacent territories")
        src, slot = pair
        state.troops[src] = 6
        action = encoder.decode(encoder.fortify_index(src, slot), state)
        assert action.troops == 5

    def test_phase_mismatch_is_rejected(self, encoder, small_game):
        state, _ = small_game
        state.phase = Phase.ATTACK
        owned = state.territories_of(state.current_player)[0]
        assert encoder.decode(encoder.draft_index(owned), state) is None

    def test_end_phase_is_blocked_while_armies_are_unplaced(self, encoder, small_game):
        state, _ = small_game
        state.phase = Phase.DRAFT
        state.troops_to_place = 4
        assert encoder.decode(END_PHASE_INDEX, state) is None
        state.troops_to_place = 0
        assert encoder.decode(END_PHASE_INDEX, state).is_end_phase()

    @pytest.mark.parametrize("index", [-1, ACTION_SPACE_SIZE, ACTION_SPACE_SIZE * 3])
    def test_out_of_range_indices_decode_to_nothing(self, encoder, small_game, index):
        state, _ = small_game
        assert encoder.decode(index, state) is None

    def test_unused_neighbour_slots_decode_to_nothing(self, encoder, small_game):
        state, _ = small_game
        state.phase = Phase.ATTACK
        src = state.territories_of(state.current_player)[0]
        beyond = len(state.board.adjacent_to(src))
        assert encoder.decode(encoder.attack_index(src, beyond), state) is None


class TestMask:
    def test_mask_and_decode_agree(self, encoder, small_game):
        state, _ = small_game
        for phase in Phase:
            state.phase = phase
            state.troops_to_place = 5 if phase is Phase.DRAFT else 0
            mask = encoder.legal_mask(state)
            for index in range(ACTION_SPACE_SIZE):
                decoded = encoder.decode(index, state)
                assert bool(mask[index]) == (decoded is not None), (
                    f"mask and decode disagree at {index} in {phase.name}"
                )

    def test_mask_is_never_empty_over_a_whole_game(self, any_board):
        """A policy with nothing legal to pick would stall the episode."""
        encoder = RiskActionSpace(any_board)
        state = GameState.new_game(any_board, num_players=2, seed=21)
        engine = RulesEngine(any_board, num_players=2, seed=21)
        for _ in range(600):
            if engine.is_terminal(state):
                break
            mask = encoder.legal_mask(state)
            assert mask.any(), f"no legal action in {state.phase.name}"
            index = int(np.flatnonzero(mask)[0])
            action = encoder.decode(index, state)
            assert action is not None
            state = engine.apply_action(state, action)

    def test_every_masked_action_is_legal_by_the_engine(self, any_board):
        encoder = RiskActionSpace(any_board)
        engine = RulesEngine(any_board, num_players=3, seed=31)
        state = GameState.new_game(any_board, num_players=3, seed=31)
        for _ in range(200):
            if engine.is_terminal(state):
                break
            legal = engine.legal_actions(state)
            for index in np.flatnonzero(encoder.legal_mask(state)):
                action = encoder.decode(int(index), state)
                assert action in legal, f"{action} is not in the engine's legal set"
            state = engine.apply_action(state, legal[0])

    def test_draft_mask_covers_exactly_the_owned_territories(self, encoder, small_game):
        state, _ = small_game
        state.phase = Phase.DRAFT
        state.troops_to_place = 2
        mask = encoder.legal_mask(state)
        owned = set(state.territories_of(state.current_player))
        marked = {int(i) for i in np.flatnonzero(mask)}
        assert marked == {encoder.draft_index(t) for t in owned}
