import json
from dataclasses import dataclass, field
from pathlib import Path


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
    def load(cls, config_path: str) -> "BoardConfig":
        path = Path(config_path)
        with path.open() as f:
            data = json.load(f)

        territories = {}
        continent_members: dict[str, list[int]] = {}
        for t in data["territories"]:
            tid = t["id"]
            territories[tid] = Territory(
                id=tid,
                name=t["name"],
                continent=t["continent"],
                adjacent=t["adjacent"],
                row=t.get("row"),
                col=t.get("col"),
            )
            continent_members.setdefault(t["continent"], []).append(tid)

        return cls(
            name=data["name"],
            max_players=data["max_players"],
            territories=territories,
            continent_bonuses=data["continent_bonuses"],
            continent_members=continent_members,
            grid=data.get("grid"),
        )

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
