"""State and observation data structures."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from loveletter_ai.actions import Action
from loveletter_ai.cards import CardName
from loveletter_ai.players import Player, PlayerId, PlayerState


class Phase(str, Enum):
    SETUP = "setup"
    DRAW = "draw"
    ACTION = "action"
    ROUND_OVER = "round_over"


class EventVisibility(str, Enum):
    PUBLIC = "public"
    PRIVATE = "private"
    DEBUG = "debug"


@dataclass(frozen=True, slots=True)
class Event:
    message: str
    visibility: EventVisibility = EventVisibility.PUBLIC
    viewer: PlayerId | None = None


@dataclass(frozen=True, slots=True)
class HistoryEntry:
    turn_index: int
    actor: PlayerId | None
    action: Action | None
    events: tuple[Event, ...]


@dataclass(slots=True)
class GameState:
    """Complete true game state owned by the environment."""

    players: list[Player]
    player_states: dict[PlayerId, PlayerState]
    deck: list[CardName]
    hidden_card: CardName | None
    current_player: PlayerId | None
    phase: Phase = Phase.SETUP
    round_number: int = 1
    turn_index: int = 0
    winner: PlayerId | None = None
    history: list[HistoryEntry] = field(default_factory=list)

    def alive_players(self) -> list[PlayerId]:
        return [
            player.id
            for player in self.players
            if not self.player_states[player.id].eliminated
        ]


@dataclass(frozen=True, slots=True)
class PublicPlayerView:
    id: PlayerId
    name: str
    hand_size: int
    discard_pile: tuple[CardName, ...]
    eliminated: bool
    protected: bool


@dataclass(frozen=True, slots=True)
class Observation:
    """Information available to a single viewer."""

    viewer: PlayerId
    round_number: int
    turn_index: int
    phase: Phase
    current_player: PlayerId | None
    deck_size: int
    players: tuple[PublicPlayerView, ...]
    own_hand: tuple[CardName, ...]
    visible_events: tuple[Event, ...] = ()
