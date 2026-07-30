from enum import Enum, auto


class Phase(Enum):
    DRAFT = auto()    # Place reinforcements
    ATTACK = auto()   # Attack adjacent territories
    FORTIFY = auto()  # Move troops between friendly territories


class CardType(Enum):
    INFANTRY = "infantry"
    CAVALRY = "cavalry"
    ARTILLERY = "artillery"
    WILD = "wild"


# Bonus troops for card set trades (escalating)
CARD_TRADE_BONUSES = [4, 6, 8, 10, 12, 15]  # 15 then +5 each subsequent trade

# Minimum troops to leave behind in a territory
MIN_TROOPS = 1

# Maximum dice per attack/defense roll
MAX_ATTACK_DICE = 3
MAX_DEFEND_DICE = 2

# Minimum troops needed to attack
MIN_TROOPS_TO_ATTACK = 2

# Continent member counts (used to validate continent control)
CONTINENT_SIZES = {
    "North America": 9,
    "South America": 4,
    "Europe": 7,
    "Africa": 6,
    "Asia": 12,
    "Australia": 4,
    # Small board
    "West": 5,
    "East": 5,
    "North": 5,
    "South": 5,
}
