"""Deterministic environment boundary for Love Letter."""

from __future__ import annotations

import random
import copy
from collections.abc import Sequence

from loveletter_ai.actions import Action
from loveletter_ai.cards import CARD_SPECS, CardName, build_deck
from loveletter_ai.players import Player, PlayerId, PlayerState
from loveletter_ai.state import (
    Event,
    EventVisibility,
    GameState,
    Observation,
    Phase,
    PublicPlayerView,
)


class IllegalActionError(ValueError):
    """Raised when an action is not legal in the current state."""


class LoveLetterEnv:
    """Owns game state, legality, deterministic randomness, and transitions."""

    def __init__(self, players: Sequence[str | Player], seed: int | None = None) -> None:
        if not 2 <= len(players) <= 4:
            raise ValueError("Love Letter supports 2 to 4 players.")

        self.rng = random.Random(seed)
        self.seed = seed
        self.players = [
            player if isinstance(player, Player) else Player(PlayerId(index), str(player))
            for index, player in enumerate(players)
        ]
        self.state: GameState | None = None

    def reset(self) -> GameState:
        """Create a deterministic initial round state.

        This implements setup only: shuffle, remove one hidden card, deal one
        card to each player, and select a starting player.
        """

        deck = build_deck()
        self.rng.shuffle(deck)
        hidden_card = deck.pop()
        player_states = {player.id: PlayerState() for player in self.players}
        for player in self.players:
            player_states[player.id].hand.append(deck.pop())

        current_player = self.rng.choice([player.id for player in self.players])
        self.state = GameState(
            players=list(self.players),
            player_states=player_states,
            deck=deck,
            hidden_card=hidden_card,
            current_player=current_player,
            phase=Phase.DRAW,
        )
        self._record(
            None,
            None,
            [Event("Round setup complete.", EventVisibility.PUBLIC)],
        )
        return self.state

    def clone(self) -> LoveLetterEnv:
        """Return a deep copy suitable for search rollouts."""

        return copy.deepcopy(self)

    def legal_actions(self, player_id: PlayerId) -> tuple[Action, ...]:
        """Return all currently legal actions for a player."""

        state = self._require_state()
        if state.phase != Phase.ACTION:
            return ()
        if state.current_player != player_id:
            return ()

        player_state = state.player_states[player_id]
        if player_state.eliminated:
            return ()

        hand = list(player_state.hand)
        if CardName.COUNTESS in hand and (
            CardName.KING in hand or CardName.PRINCE in hand
        ):
            return (Action(player_id, CardName.COUNTESS),)

        actions: list[Action] = []
        for card in sorted(set(hand), key=lambda card_name: CARD_SPECS[card_name].value):
            actions.extend(self._actions_for_card(player_id, card))
        return tuple(actions)

    def step(self, action: Action) -> GameState:
        """Validate and apply one action."""

        state = self._require_state()
        legal = self.legal_actions(action.actor)
        if action not in legal:
            raise IllegalActionError(f"Illegal action: {action}")

        events = self._resolve_action(action)
        state.turn_index += 1
        self._finish_turn_or_round(events)
        self._record(action.actor, action, events)
        return state

    def begin_action_phase(self) -> GameState:
        """Draw for the current player and enter the action phase."""

        state = self._require_state()
        if state.phase == Phase.ROUND_OVER:
            return state
        if state.phase != Phase.DRAW:
            return state
        if state.current_player is None:
            raise RuntimeError("No current player is set.")

        player_state = state.player_states[state.current_player]
        player_state.protected = False
        events = [Event(f"{self._name(state.current_player)} begins their turn.")]

        if state.deck:
            drawn = state.deck.pop()
            player_state.hand.append(drawn)
            events.append(
                Event(
                    f"{self._name(state.current_player)} drew a card.",
                    EventVisibility.PUBLIC,
                )
            )
        else:
            events.append(Event("The deck is empty before draw."))
            self._end_round_by_showdown(events)

        if state.phase != Phase.ROUND_OVER:
            state.phase = Phase.ACTION
        self._record(state.current_player, None, events)
        return state

    def observe(self, viewer: PlayerId) -> Observation:
        """Return a hidden-information-safe observation for one player."""

        state = self._require_state()
        if viewer not in state.player_states:
            raise ValueError(f"Unknown viewer: {viewer}")

        public_players = []
        for player in state.players:
            player_state = state.player_states[player.id]
            public_players.append(
                PublicPlayerView(
                    id=player.id,
                    name=player.name,
                    hand_size=len(player_state.hand),
                    discard_pile=tuple(player_state.discard_pile),
                    eliminated=player_state.eliminated,
                    protected=player_state.protected,
                )
            )

        visible_events = tuple(
            event
            for entry in state.history
            for event in entry.events
            if event.visibility == EventVisibility.PUBLIC or event.viewer == viewer
        )

        return Observation(
            viewer=viewer,
            round_number=state.round_number,
            turn_index=state.turn_index,
            phase=state.phase,
            current_player=state.current_player,
            deck_size=len(state.deck),
            players=tuple(public_players),
            own_hand=tuple(state.player_states[viewer].hand),
            visible_events=visible_events,
        )

    def _record(
        self,
        actor: PlayerId | None,
        action: Action | None,
        events: Sequence[Event],
    ) -> None:
        state = self._require_state()
        from loveletter_ai.state import HistoryEntry

        state.history.append(
            HistoryEntry(
                turn_index=state.turn_index,
                actor=actor,
                action=action,
                events=tuple(events),
            )
        )

    def _require_state(self) -> GameState:
        if self.state is None:
            raise RuntimeError("Environment has not been reset.")
        return self.state

    def _actions_for_card(self, actor: PlayerId, card: CardName) -> list[Action]:
        state = self._require_state()
        targetable = self._targetable_opponents(actor)
        alive = self._alive_player_ids()

        if card == CardName.GUARD:
            guesses = [
                guess
                for guess in CardName
                if guess != CardName.GUARD
            ]
            if not targetable:
                return [Action(actor, card)]
            return [
                Action(actor, card, target, guess)
                for target in targetable
                for guess in guesses
            ]
        if card in (CardName.PRIEST, CardName.BARON, CardName.KING):
            if not targetable:
                return [Action(actor, card)]
            return [Action(actor, card, target) for target in targetable]
        if card == CardName.PRINCE:
            targets = [
                player_id
                for player_id in alive
                if player_id == actor or not state.player_states[player_id].protected
            ]
            return [Action(actor, card, target) for target in targets]
        return [Action(actor, card)]

    def _resolve_action(self, action: Action) -> list[Event]:
        state = self._require_state()
        actor_state = state.player_states[action.actor]
        actor_name = self._name(action.actor)
        actor_state.hand.remove(action.card)
        actor_state.discard_pile.append(action.card)

        events = [Event(f"{actor_name} played {action.card.value}.")]

        if action.card == CardName.PRINCESS:
            self._eliminate(action.actor, events, "discarded the Princess")
            return events

        if action.card == CardName.GUARD:
            self._resolve_guard(action, events)
        elif action.card == CardName.PRIEST:
            self._resolve_priest(action, events)
        elif action.card == CardName.BARON:
            self._resolve_baron(action, events)
        elif action.card == CardName.HANDMAID:
            actor_state.protected = True
            events.append(Event(f"{actor_name} is protected until their next turn."))
        elif action.card == CardName.PRINCE:
            self._resolve_prince(action, events)
        elif action.card == CardName.KING:
            self._resolve_king(action, events)
        elif action.card == CardName.COUNTESS:
            events.append(Event(f"{actor_name}'s Countess has no effect."))

        return events

    def _resolve_guard(self, action: Action, events: list[Event]) -> None:
        if action.target is None or action.guess is None:
            events.append(Event("Guard had no valid target."))
            return
        target_state = self._require_state().player_states[action.target]
        target_name = self._name(action.target)
        events.append(Event(f"{self._name(action.actor)} guessed {target_name}'s card."))
        if target_state.hand and target_state.hand[0] == action.guess:
            self._eliminate(action.target, events, f"was guessed as {action.guess.value}")
        else:
            events.append(Event(f"The guess was wrong. {target_name} stays alive."))

    def _resolve_priest(self, action: Action, events: list[Event]) -> None:
        if action.target is None:
            events.append(Event("Priest had no valid target."))
            return
        target_hand = self._require_state().player_states[action.target].hand
        seen = target_hand[0].value if target_hand else "nothing"
        events.append(
            Event(
                f"{self._name(action.actor)} saw {self._name(action.target)} holding {seen}.",
                EventVisibility.PRIVATE,
                viewer=action.actor,
            )
        )

    def _resolve_baron(self, action: Action, events: list[Event]) -> None:
        if action.target is None:
            events.append(Event("Baron had no valid target."))
            return
        state = self._require_state()
        actor_hand = state.player_states[action.actor].hand
        target_hand = state.player_states[action.target].hand
        if not actor_hand or not target_hand:
            events.append(Event("Baron comparison could not happen."))
            return
        actor_value = CARD_SPECS[actor_hand[0]].value
        target_value = CARD_SPECS[target_hand[0]].value
        events.append(Event("Baron comparison resolved privately."))
        if actor_value < target_value:
            self._eliminate(action.actor, events, "lost a Baron comparison")
        elif target_value < actor_value:
            self._eliminate(action.target, events, "lost a Baron comparison")
        else:
            events.append(Event("The Baron comparison was tied."))

    def _resolve_prince(self, action: Action, events: list[Event]) -> None:
        state = self._require_state()
        if action.target is None:
            events.append(Event("Prince had no valid target."))
            return

        target_state = state.player_states[action.target]
        target_name = self._name(action.target)
        if not target_state.hand:
            events.append(Event(f"{target_name} had no card to discard."))
            return

        discarded = target_state.hand.pop()
        target_state.discard_pile.append(discarded)
        events.append(Event(f"{target_name} discarded {discarded.value}."))
        if discarded == CardName.PRINCESS:
            self._eliminate(action.target, events, "discarded the Princess")
            return

        if state.deck:
            target_state.hand.append(state.deck.pop())
            events.append(Event(f"{target_name} drew a replacement card."))
        elif state.hidden_card is not None:
            target_state.hand.append(state.hidden_card)
            state.hidden_card = None
            events.append(Event(f"{target_name} drew the hidden set-aside card."))
        else:
            events.append(Event(f"{target_name} could not draw a replacement card."))

    def _resolve_king(self, action: Action, events: list[Event]) -> None:
        if action.target is None:
            events.append(Event("King had no valid target."))
            return
        state = self._require_state()
        actor_hand = state.player_states[action.actor].hand
        target_hand = state.player_states[action.target].hand
        state.player_states[action.actor].hand = target_hand
        state.player_states[action.target].hand = actor_hand
        events.append(
            Event(f"{self._name(action.actor)} swapped hands with {self._name(action.target)}.")
        )

    def _finish_turn_or_round(self, events: list[Event]) -> None:
        state = self._require_state()
        alive = self._alive_player_ids()
        if len(alive) <= 1:
            state.winner = alive[0] if alive else None
            state.phase = Phase.ROUND_OVER
            if state.winner is not None:
                events.append(Event(f"{self._name(state.winner)} wins by elimination."))
            return

        if not state.deck:
            self._end_round_by_showdown(events)
            return

        state.current_player = self._next_alive_player(state.current_player)
        state.phase = Phase.DRAW

    def _end_round_by_showdown(self, events: list[Event]) -> None:
        state = self._require_state()
        contenders = []
        for player_id in self._alive_player_ids():
            player_state = state.player_states[player_id]
            hand_value = CARD_SPECS[player_state.hand[0]].value if player_state.hand else -1
            contenders.append((hand_value, player_state.discard_total, -int(player_id), player_id))

        if not contenders:
            state.winner = None
            events.append(Event("The round ended with no surviving players."))
        else:
            _, _, _, winner = max(contenders)
            state.winner = winner
            events.append(Event(f"{self._name(winner)} wins the showdown."))
        state.phase = Phase.ROUND_OVER

    def _targetable_opponents(self, actor: PlayerId) -> list[PlayerId]:
        state = self._require_state()
        return [
            player_id
            for player_id in self._alive_player_ids()
            if player_id != actor and not state.player_states[player_id].protected
        ]

    def _alive_player_ids(self) -> list[PlayerId]:
        state = self._require_state()
        return [
            player.id
            for player in state.players
            if not state.player_states[player.id].eliminated
        ]

    def _next_alive_player(self, current: PlayerId | None) -> PlayerId:
        state = self._require_state()
        alive = set(self._alive_player_ids())
        player_ids = [player.id for player in state.players]
        if current is None:
            return player_ids[0]
        start = player_ids.index(current)
        for offset in range(1, len(player_ids) + 1):
            candidate = player_ids[(start + offset) % len(player_ids)]
            if candidate in alive:
                return candidate
        raise RuntimeError("No alive players remain.")

    def _eliminate(
        self,
        player_id: PlayerId,
        events: list[Event],
        reason: str,
    ) -> None:
        player_state = self._require_state().player_states[player_id]
        player_state.eliminated = True
        player_state.protected = False
        events.append(Event(f"{self._name(player_id)} is eliminated: {reason}."))

    def _name(self, player_id: PlayerId) -> str:
        state = self._require_state()
        for player in state.players:
            if player.id == player_id:
                return player.name
        return f"P{int(player_id)}"
