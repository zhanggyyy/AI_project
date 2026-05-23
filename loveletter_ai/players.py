"""Player identity and per-round player state."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import NewType

from loveletter_ai.cards import CardName

PlayerId = NewType("PlayerId", int)


@dataclass(frozen=True, slots=True)
class Player:
    id: PlayerId
    name: str


@dataclass(slots=True)
class PlayerState:
    hand: list[CardName] = field(default_factory=list)
    discard_pile: list[CardName] = field(default_factory=list)
    eliminated: bool = False
    protected: bool = False

    @property
    def discard_total(self) -> int:
        from loveletter_ai.cards import CARD_SPECS

        return sum(CARD_SPECS[card].value for card in self.discard_pile)
