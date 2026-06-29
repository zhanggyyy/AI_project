"""Small shared helpers for agent implementations."""

from __future__ import annotations

from loveletter_ai.cards import CARD_SPECS, CardName
from loveletter_ai.players import PlayerId
from loveletter_ai.state import Observation


def card_value(card: CardName) -> int:
    return CARD_SPECS[card].value


def remaining_after_play(hand: list[CardName], played: CardName) -> list[CardName]:
    remaining = list(hand)
    if played in remaining:
        remaining.remove(played)
    return remaining


def card_from_label(label: str) -> CardName | None:
    normalized = label.strip().rstrip(".")
    for card in CardName:
        if card.value == normalized:
            return card
    return None


def vulnerable_opponents(observation: Observation) -> set[PlayerId]:
    """Return currently targetable opponents from one player's observation."""

    return {
        player.id
        for player in observation.players
        if (
            player.id != observation.viewer
            and not player.eliminated
            and not player.protected
        )
    }
