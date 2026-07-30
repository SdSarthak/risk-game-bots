import random
from engine.constants import CARD_TRADE_BONUS_CAP, CARD_TRADE_BONUSES, CardType


class CardDeck:
    """Risk card deck: Infantry, Cavalry, Artillery + 2 Wilds."""

    def __init__(self, num_territories: int, seed: int | None = None):
        self._rng = random.Random(seed)
        self._num_territories = num_territories
        self._deck: list[CardType] = self._build_deck(num_territories)
        self._discard: list[CardType] = []

    def _build_deck(self, num_territories: int) -> list[CardType]:
        types = [CardType.INFANTRY, CardType.CAVALRY, CardType.ARTILLERY]
        cards = [types[i % 3] for i in range(num_territories)]
        cards += [CardType.WILD, CardType.WILD]
        self._rng.shuffle(cards)
        return cards

    def draw(self) -> CardType:
        if not self._deck:
            if self._discard:
                self._deck = self._discard[:]
                self._discard = []
                self._rng.shuffle(self._deck)
            else:
                # All cards are in players' hands — rebuild a fresh deck as fallback
                self._deck = self._build_deck(self._num_territories)
        return self._deck.pop()

    def discard(self, cards: list[CardType]) -> None:
        self._discard.extend(cards)

    @staticmethod
    def bonus_for_trade(trade_count: int) -> int:
        """
        Bonus troops earned for the nth card trade (0-indexed).

        Follows the standard 4/6/8/10/12/15 ladder, then +5 per trade up to
        CARD_TRADE_BONUS_CAP.
        """
        if trade_count < 0:
            raise ValueError("trade_count must be non-negative")
        if trade_count < len(CARD_TRADE_BONUSES):
            return CARD_TRADE_BONUSES[trade_count]
        return min(15 + 5 * (trade_count - 5), CARD_TRADE_BONUS_CAP)

    @staticmethod
    def find_valid_set(hand: list[CardType]) -> list[int] | None:
        """
        Find indices of a valid tradeable set of 3 cards from hand.
        Valid sets: 3 of same type, or 1 of each type, or any 2 + 1 wild.
        Returns card indices if found, else None.
        """
        if len(hand) < 3:
            return None

        # Build index lists per type
        by_type: dict[CardType, list[int]] = {}
        for i, c in enumerate(hand):
            by_type.setdefault(c, []).append(i)

        wilds = by_type.get(CardType.WILD, [])
        infantry = by_type.get(CardType.INFANTRY, [])
        cavalry = by_type.get(CardType.CAVALRY, [])
        artillery = by_type.get(CardType.ARTILLERY, [])

        # 3 of a kind (no wilds needed)
        for group in [infantry, cavalry, artillery]:
            if len(group) >= 3:
                return group[:3]

        # 1 of each type
        if infantry and cavalry and artillery:
            return [infantry[0], cavalry[0], artillery[0]]

        # 2 matching + 1 wild
        if wilds:
            for group in [infantry, cavalry, artillery]:
                if len(group) >= 2:
                    return group[:2] + wilds[:1]
            # 1 non-wild + 2 wilds... not valid in standard Risk, skip

        # 1 type + 2 wilds
        if len(wilds) >= 2:
            for group in [infantry, cavalry, artillery]:
                if group:
                    return group[:1] + wilds[:2]

        # 3 wilds (non-standard but handle it)
        if len(wilds) >= 3:
            return wilds[:3]

        return None
