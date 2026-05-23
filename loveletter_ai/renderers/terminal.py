"""Terminal rendering for public, viewer, and reveal debug modes."""

from __future__ import annotations

from loveletter_ai.cards import CardName
from loveletter_ai.players import PlayerId
from loveletter_ai.state import EventVisibility, GameState


class TerminalRenderer:
    """Pure string renderer for deterministic terminal debugging."""

    hidden_token = "??"
    empty_token = "--"

    def render_state(
        self,
        state: GameState,
        viewer: PlayerId | None = None,
        reveal: bool = False,
        recent_events: int = 5,
    ) -> str:
        current = self._player_label(state, state.current_player)
        hidden = self._card_label(state.hidden_card) if reveal else self.hidden_token
        lines = [
            (
                f"Round {state.round_number} | {state.phase.value.upper()} | "
                f"Current: {current} | Deck: {len(state.deck)} | Hidden: {hidden}"
            ),
            "",
            "Players",
        ]

        for player in state.players:
            marker = ">" if player.id == state.current_player else " "
            player_state = state.player_states[player.id]
            status = "out" if player_state.eliminated else "alive"
            protected = "yes" if player_state.protected else "no"
            hand = self._render_hand(player_state.hand, player.id, viewer, reveal)
            discard = self._render_cards(player_state.discard_pile)
            lines.append(
                f"  {marker} P{int(player.id)} {player.name:<10} "
                f"{status:<5} protected: {protected:<3} "
                f"hand: {hand:<18} discard: {discard}"
            )

        events = list(self._visible_events(state, viewer, reveal))[-recent_events:]
        if events:
            lines.extend(["", "Recent Events"])
            for event in events:
                lines.append(f"  [{event.visibility.value}] {event.message}")

        return "\n".join(lines)

    def render_actions(self, actions: list | tuple) -> str:
        lines = ["Legal Actions"]
        for index, action in enumerate(actions):
            label = action.label() if hasattr(action, "label") else str(action)
            lines.append(f"  {index}. {label}")
        return "\n".join(lines)

    def _render_hand(
        self,
        hand: list[CardName],
        player_id: PlayerId,
        viewer: PlayerId | None,
        reveal: bool,
    ) -> str:
        if not hand:
            return self.empty_token
        if reveal or viewer == player_id:
            return self._render_cards(hand)
        return ", ".join([self.hidden_token] * len(hand))

    def _render_cards(self, cards: list[CardName]) -> str:
        if not cards:
            return self.empty_token
        return ", ".join(self._card_label(card) for card in cards)

    def _card_label(self, card: CardName | None) -> str:
        return card.value if card is not None else self.empty_token

    def _player_label(self, state: GameState, player_id: PlayerId | None) -> str:
        if player_id is None:
            return self.empty_token
        for player in state.players:
            if player.id == player_id:
                return f"P{int(player.id)} {player.name}"
        return f"P{int(player_id)}"

    def _visible_events(
        self,
        state: GameState,
        viewer: PlayerId | None,
        reveal: bool,
    ):
        for entry in state.history:
            for event in entry.events:
                if reveal:
                    yield event
                elif event.visibility == EventVisibility.PUBLIC:
                    yield event
                elif viewer is not None and event.viewer == viewer:
                    yield event
