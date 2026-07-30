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


# Bonus troops for card set trades (escalating), then +5 per subsequent trade
CARD_TRADE_BONUSES = [4, 6, 8, 10, 12, 15]

# Ceiling on the escalating trade bonus. Without a ceiling the bonus outgrows
# combat losses and long games never terminate: both sides simply stockpile.
CARD_TRADE_BONUS_CAP = 30

# Minimum troops to leave behind in a territory
MIN_TROOPS = 1

# Maximum dice per attack/defense roll
MAX_ATTACK_DICE = 3
MAX_DEFEND_DICE = 2

# Minimum troops needed to attack
MIN_TROOPS_TO_ATTACK = 2

# A hand must be traded in once it reaches this many cards (standard Risk rule)
MAX_CARDS_IN_HAND = 5

# Cards required to form a tradeable set
CARDS_PER_SET = 3
