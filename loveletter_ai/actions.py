"""Action objects selected by agents and validated by the environment."""

from __future__ import annotations

from dataclasses import dataclass

from loveletter_ai.cards import CardName
from loveletter_ai.players import PlayerId


@dataclass(frozen=True, slots=True)
class Action:
    """A declarative player command.

    The environment owns interpretation and mutation. Optional fields are used by
    card effects such as targeted cards and Guard guesses.
    """

    actor: PlayerId
    card: CardName
    target: PlayerId | None = None
    guess: CardName | None = None

    def label(self) -> str:
        parts = [f"Play {self.card.value}"]
        if self.target is not None:
            parts.append(f"-> P{int(self.target)}")
        if self.guess is not None:
            parts.append(f"guess {self.guess.value}")
        return " ".join(parts)
