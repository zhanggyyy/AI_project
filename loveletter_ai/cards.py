"""Card definitions for Love Letter.

This module defines card types, not physical card instances. For the base game,
cards with the same name are interchangeable.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class CardName(str, Enum):
    GUARD = "Guard"
    PRIEST = "Priest"
    BARON = "Baron"
    HANDMAID = "Handmaid"
    PRINCE = "Prince"
    KING = "King"
    COUNTESS = "Countess"
    PRINCESS = "Princess"


@dataclass(frozen=True, slots=True)
class CardSpec:
    name: CardName
    value: int
    count: int
    text: str


CARD_SPECS: dict[CardName, CardSpec] = {
    CardName.GUARD: CardSpec(
        CardName.GUARD,
        value=1,
        count=5,
        text="Guess another player's non-Guard hand card.",
    ),
    CardName.PRIEST: CardSpec(
        CardName.PRIEST,
        value=2,
        count=2,
        text="Privately view another player's hand.",
    ),
    CardName.BARON: CardSpec(
        CardName.BARON,
        value=3,
        count=2,
        text="Privately compare hands; lower value is eliminated.",
    ),
    CardName.HANDMAID: CardSpec(
        CardName.HANDMAID,
        value=4,
        count=2,
        text="Gain protection until your next turn.",
    ),
    CardName.PRINCE: CardSpec(
        CardName.PRINCE,
        value=5,
        count=2,
        text="Target discards hand and redraws.",
    ),
    CardName.KING: CardSpec(
        CardName.KING,
        value=6,
        count=1,
        text="Swap hands with another player.",
    ),
    CardName.COUNTESS: CardSpec(
        CardName.COUNTESS,
        value=7,
        count=1,
        text="Must be played with King or Prince; otherwise no effect.",
    ),
    CardName.PRINCESS: CardSpec(
        CardName.PRINCESS,
        value=8,
        count=1,
        text="If discarded for any reason, the player is eliminated.",
    ),
}


def build_deck() -> list[CardName]:
    """Return an ordered base-game deck before shuffling."""

    deck: list[CardName] = []
    for name, spec in CARD_SPECS.items():
        deck.extend([name] * spec.count)
    return deck
