import json
from dataclasses import dataclass, field
from pathlib import Path

# The engine indexes ownership and troop arrays by territory id, so ids have to
# be exactly 0..n-1. A gap or a duplicate silently mis-indexes the whole game.
REQUIRED_TOP_LEVEL_KEYS = ("name", "max_players", "territories", "continent_bonuses")
REQUIRED_TERRITORY_KEYS = ("id", "name", "continent", "adjacent")


class BoardConfigError(ValueError):
    """
    A board config could not be read or is internally inconsistent.

    Subclasses ValueError so existing callers that catch ValueError (the API's
    create-game handler, for one) keep working.
    """


@dataclass
class Territory:
    id: int
    name: str
    continent: str
    adjacent: list[int]
    row: int | None = None   # grid row (None for non-grid boards)
    col: int | None = None   # grid col (None for non-grid boards)


@dataclass
class BoardConfig:
    name: str
    max_players: int
    territories: dict[int, Territory]
    continent_bonuses: dict[str, int]
    continent_members: dict[str, list[int]] = field(default_factory=dict)
    grid: dict | None = None  # {"rows": N, "cols": M} if grid board

    @classmethod
    def load(cls, config_path: str, validate: bool = True) -> "BoardConfig":
        """
        Read a board config from JSON.

        Raises :class:`BoardConfigError` — never a bare ``KeyError`` or
        ``JSONDecodeError`` from somewhere inside the parser — when the file is
        missing, is not JSON, or describes a board the engine cannot play. Pass
        ``validate=False`` only to inspect a board you already know is broken.
        """
        path = Path(config_path)
        try:
            with path.open(encoding="utf-8") as f:
                data = json.load(f)
        except FileNotFoundError as exc:
            raise BoardConfigError(f"Board config not found: {path}") from exc
        except OSError as exc:
            raise BoardConfigError(f"Could not read board config {path}: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise BoardConfigError(f"Board config {path} is not valid JSON: {exc}") from exc

        if not isinstance(data, dict):
            raise BoardConfigError(f"Board config {path} must hold a JSON object")
        missing = [k for k in REQUIRED_TOP_LEVEL_KEYS if k not in data]
        if missing:
            raise BoardConfigError(
                f"Board config {path} is missing required key(s): {', '.join(missing)}"
            )
        raw_territories = data["territories"]
        if not isinstance(raw_territories, list) or not raw_territories:
            raise BoardConfigError(
                f"Board config {path} needs a non-empty 'territories' list"
            )

        territories: dict[int, Territory] = {}
        continent_members: dict[str, list[int]] = {}
        for position, t in enumerate(raw_territories):
            if not isinstance(t, dict):
                raise BoardConfigError(
                    f"{path}: territory #{position} must be a JSON object"
                )
            absent = [k for k in REQUIRED_TERRITORY_KEYS if k not in t]
            if absent:
                raise BoardConfigError(
                    f"{path}: territory #{position} is missing {', '.join(absent)}"
                )
            tid = t["id"]
            if not isinstance(tid, int) or isinstance(tid, bool):
                raise BoardConfigError(f"{path}: territory id {tid!r} is not an integer")
            if tid in territories:
                raise BoardConfigError(
                    f"{path}: territory id {tid} appears more than once "
                    f"('{territories[tid].name}' and '{t['name']}')"
                )
            adjacent = t["adjacent"]
            if not isinstance(adjacent, list) or any(
                not isinstance(a, int) or isinstance(a, bool) for a in adjacent
            ):
                raise BoardConfigError(
                    f"{path}: territory {tid} has a non-integer in its adjacency list"
                )
            territories[tid] = Territory(
                id=tid,
                name=t["name"],
                continent=t["continent"],
                adjacent=list(adjacent),
                row=t.get("row"),
                col=t.get("col"),
            )
            continent_members.setdefault(t["continent"], []).append(tid)

        board = cls(
            name=data["name"],
            max_players=data["max_players"],
            territories=territories,
            continent_bonuses=data["continent_bonuses"],
            continent_members=continent_members,
            grid=data.get("grid"),
        )
        if validate:
            board.validate(source=str(path))
        return board

    def validate(self, source: str | None = None) -> None:
        """
        Check the invariants the engine relies on, raising `BoardConfigError`.

        Every one of these has produced a wrong game rather than an error at
        some point: ids that do not index the state arrays, one-way borders that
        let an attack cross where a counter-attack cannot, and continent bonuses
        for continents that hold no territory — which `all()` over an empty
        member list awards to *every* player, every turn.
        """
        where = f"{source}: " if source else ""

        def fail(message: str) -> None:
            raise BoardConfigError(where + message)

        if not isinstance(self.max_players, int) or isinstance(self.max_players, bool):
            fail(f"max_players must be an integer, got {self.max_players!r}")
        if self.max_players < 2:
            fail(f"max_players must be at least 2, got {self.max_players}")

        n = len(self.territories)
        if set(self.territories) != set(range(n)):
            missing = sorted(set(range(n)) - set(self.territories))
            extra = sorted(set(self.territories) - set(range(n)))
            fail(
                f"territory ids must be exactly 0..{n - 1}; "
                f"missing {missing}, unexpected {extra}"
            )
        if self.max_players > n:
            fail(f"max_players is {self.max_players} but the board has only {n} territories")

        for tid, terr in self.territories.items():
            if len(set(terr.adjacent)) != len(terr.adjacent):
                fail(f"territory {tid} ('{terr.name}') lists a neighbour twice")
            if tid in terr.adjacent:
                fail(f"territory {tid} ('{terr.name}') is adjacent to itself")
            for nb in terr.adjacent:
                if nb not in self.territories:
                    fail(f"territory {tid} ('{terr.name}') borders unknown territory {nb}")
                if tid not in self.territories[nb].adjacent:
                    fail(
                        f"one-way border: {tid} ('{terr.name}') lists {nb} "
                        f"('{self.territories[nb].name}') but not the reverse"
                    )

        if not isinstance(self.continent_bonuses, dict):
            fail("continent_bonuses must be an object mapping continent -> troops")
        for continent, bonus in self.continent_bonuses.items():
            if not isinstance(bonus, int) or isinstance(bonus, bool):
                fail(f"continent bonus for '{continent}' must be an integer, got {bonus!r}")
            if continent not in self.continent_members:
                fail(
                    f"continent_bonuses names '{continent}', which holds no territories; "
                    "an empty continent counts as controlled by everyone"
                )

        if self.grid is not None:
            if not isinstance(self.grid, dict) or not {"rows", "cols"} <= set(self.grid):
                fail("grid must be an object with 'rows' and 'cols'")
            rows, cols = self.grid["rows"], self.grid["cols"]
            for label, value in (("rows", rows), ("cols", cols)):
                if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                    fail(f"grid {label} must be a positive integer, got {value!r}")
            if rows * cols < n:
                fail(f"grid is {rows}x{cols} but the board has {n} territories")

    @property
    def num_territories(self) -> int:
        return len(self.territories)

    def adjacent_to(self, territory_id: int) -> list[int]:
        return self.territories[territory_id].adjacent

    def continent_of(self, territory_id: int) -> str:
        return self.territories[territory_id].continent

    def territories_in_continent(self, continent: str) -> list[int]:
        return self.continent_members.get(continent, [])

    def are_adjacent(self, a: int, b: int) -> bool:
        return b in self.territories[a].adjacent
