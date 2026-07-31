"""
Board config loading and validation.

Every invariant checked here has produced a wrong game at some point rather than
an error: a one-way border lets an attack cross where the counter-attack cannot,
a non-contiguous id mis-indexes the ownership array, and a continent bonus with
no territories behind it is awarded to every player on every turn.

All fixtures are synthetic JSON written to tmp_path — nothing is downloaded.
"""
import json

import pytest

from engine.board import BoardConfig, BoardConfigError
from engine.state import GameState


def line_board(n: int = 4) -> dict:
    """A minimal, valid board: n territories in a row, one continent."""
    return {
        "name": "line",
        "max_players": 2,
        "continent_bonuses": {"Line": 2},
        "territories": [
            {
                "id": i,
                "name": f"T{i}",
                "continent": "Line",
                "adjacent": [j for j in (i - 1, i + 1) if 0 <= j < n],
            }
            for i in range(n)
        ],
    }


def write(tmp_path, data, name="board.json"):
    path = tmp_path / name
    path.write_text(json.dumps(data), encoding="utf-8")
    return str(path)


class TestValidConfigs:
    def test_minimal_board_loads(self, tmp_path):
        board = BoardConfig.load(write(tmp_path, line_board()))
        assert board.num_territories == 4
        assert board.territories_in_continent("Line") == [0, 1, 2, 3]

    def test_a_loaded_board_can_start_a_game(self, tmp_path):
        board = BoardConfig.load(write(tmp_path, line_board()))
        state = GameState.new_game(board, num_players=2, seed=0)
        assert sorted(set(state.owners)) == [0, 1]

    def test_shipped_configs_all_validate(self, any_board):
        any_board.validate()  # must not raise


class TestFileLevelFailures:
    def test_missing_file(self, tmp_path):
        with pytest.raises(BoardConfigError, match="not found"):
            BoardConfig.load(str(tmp_path / "nope.json"))

    def test_not_json(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("{ this is not json", encoding="utf-8")
        with pytest.raises(BoardConfigError, match="not valid JSON"):
            BoardConfig.load(str(path))

    def test_json_but_not_an_object(self, tmp_path):
        with pytest.raises(BoardConfigError, match="JSON object"):
            BoardConfig.load(write(tmp_path, [1, 2, 3]))

    def test_empty_object_reports_every_missing_key(self, tmp_path):
        with pytest.raises(BoardConfigError) as exc:
            BoardConfig.load(write(tmp_path, {}))
        message = str(exc.value)
        for key in ("name", "max_players", "territories", "continent_bonuses"):
            assert key in message

    def test_empty_territory_list(self, tmp_path):
        data = line_board()
        data["territories"] = []
        with pytest.raises(BoardConfigError, match="non-empty"):
            BoardConfig.load(write(tmp_path, data))

    def test_territory_missing_a_field(self, tmp_path):
        data = line_board()
        del data["territories"][1]["continent"]
        with pytest.raises(BoardConfigError, match="continent"):
            BoardConfig.load(write(tmp_path, data))

    def test_a_broken_config_can_still_be_inspected(self, tmp_path):
        data = line_board()
        data["territories"][0]["adjacent"] = [3]
        board = BoardConfig.load(write(tmp_path, data), validate=False)
        assert board.num_territories == 4
        with pytest.raises(BoardConfigError):
            board.validate()


class TestStructuralFailures:
    def test_duplicate_territory_id(self, tmp_path):
        data = line_board()
        data["territories"][2]["id"] = 1
        with pytest.raises(BoardConfigError, match="more than once"):
            BoardConfig.load(write(tmp_path, data))

    def test_non_contiguous_ids(self, tmp_path):
        """State arrays are indexed by id, so a gap silently mis-indexes them."""
        data = line_board()
        data["territories"][3]["id"] = 9
        data["territories"][2]["adjacent"] = [1, 9]
        data["territories"][3]["adjacent"] = [2]
        with pytest.raises(BoardConfigError, match="0..3"):
            BoardConfig.load(write(tmp_path, data))

    def test_non_integer_id(self, tmp_path):
        data = line_board()
        data["territories"][0]["id"] = "zero"
        with pytest.raises(BoardConfigError, match="not an integer"):
            BoardConfig.load(write(tmp_path, data))

    def test_adjacency_out_of_range(self, tmp_path):
        data = line_board()
        data["territories"][0]["adjacent"] = [1, 99]
        with pytest.raises(BoardConfigError, match="unknown territory 99"):
            BoardConfig.load(write(tmp_path, data))

    def test_non_integer_adjacency(self, tmp_path):
        data = line_board()
        data["territories"][0]["adjacent"] = ["T1"]
        with pytest.raises(BoardConfigError, match="non-integer"):
            BoardConfig.load(write(tmp_path, data))

    def test_self_adjacency(self, tmp_path):
        data = line_board()
        data["territories"][0]["adjacent"] = [0, 1]
        with pytest.raises(BoardConfigError, match="adjacent to itself"):
            BoardConfig.load(write(tmp_path, data))

    def test_duplicate_neighbour(self, tmp_path):
        data = line_board()
        data["territories"][0]["adjacent"] = [1, 1]
        with pytest.raises(BoardConfigError, match="twice"):
            BoardConfig.load(write(tmp_path, data))

    def test_one_way_border(self, tmp_path):
        """Exactly the defect that made 22 classic_42 borders asymmetric."""
        data = line_board()
        data["territories"][0]["adjacent"] = [1, 3]
        with pytest.raises(BoardConfigError, match="one-way border"):
            BoardConfig.load(write(tmp_path, data))


class TestContinentAndPlayerFailures:
    def test_bonus_for_a_continent_with_no_territories(self, tmp_path):
        data = line_board()
        data["continent_bonuses"]["Atlantis"] = 7
        with pytest.raises(BoardConfigError, match="Atlantis"):
            BoardConfig.load(write(tmp_path, data))

    def test_an_empty_continent_is_not_controlled_by_anyone(self, tmp_path):
        """Guards the `all([]) is True` trap behind the check above."""
        board = BoardConfig.load(write(tmp_path, line_board()))
        state = GameState.new_game(board, num_players=2, seed=0)
        assert not state.controls_continent(0, "Atlantis")
        assert not state.controls_continent(1, "Atlantis")

    def test_non_integer_bonus(self, tmp_path):
        data = line_board()
        data["continent_bonuses"]["Line"] = "two"
        with pytest.raises(BoardConfigError, match="must be an integer"):
            BoardConfig.load(write(tmp_path, data))

    @pytest.mark.parametrize("value", [1, 0, -3])
    def test_max_players_below_two(self, tmp_path, value):
        data = line_board()
        data["max_players"] = value
        with pytest.raises(BoardConfigError, match="at least 2"):
            BoardConfig.load(write(tmp_path, data))

    def test_max_players_above_territory_count(self, tmp_path):
        data = line_board(n=3)
        data["max_players"] = 6
        with pytest.raises(BoardConfigError, match="only 3 territories"):
            BoardConfig.load(write(tmp_path, data))


class TestGridFailures:
    def test_grid_needs_rows_and_cols(self, tmp_path):
        data = line_board()
        data["grid"] = {"rows": 2}
        with pytest.raises(BoardConfigError, match="rows"):
            BoardConfig.load(write(tmp_path, data))

    def test_grid_dimensions_must_be_positive(self, tmp_path):
        data = line_board()
        data["grid"] = {"rows": 0, "cols": 4}
        with pytest.raises(BoardConfigError, match="positive integer"):
            BoardConfig.load(write(tmp_path, data))

    def test_grid_must_be_big_enough(self, tmp_path):
        data = line_board()
        data["grid"] = {"rows": 1, "cols": 2}
        with pytest.raises(BoardConfigError, match="4 territories"):
            BoardConfig.load(write(tmp_path, data))

    def test_valid_grid_is_kept(self, tmp_path):
        data = line_board()
        data["grid"] = {"rows": 2, "cols": 2}
        board = BoardConfig.load(write(tmp_path, data))
        assert board.grid == {"rows": 2, "cols": 2}
