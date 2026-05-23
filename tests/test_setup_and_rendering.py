from unittest import TestCase

from loveletter_ai.environment import LoveLetterEnv
from loveletter_ai.renderers import TerminalRenderer


class SetupAndRenderingTest(TestCase):
    def test_reset_is_deterministic_for_same_seed(self) -> None:
        first = LoveLetterEnv(["Alice", "Bob"], seed=188).reset()
        second = LoveLetterEnv(["Alice", "Bob"], seed=188).reset()

        self.assertEqual(first.deck, second.deck)
        self.assertEqual(first.hidden_card, second.hidden_card)
        self.assertEqual(first.current_player, second.current_player)
        self.assertEqual(
            [first.player_states[player.id].hand for player in first.players],
            [second.player_states[player.id].hand for player in second.players],
        )

    def test_public_renderer_masks_private_cards(self) -> None:
        state = LoveLetterEnv(["Alice", "Bob"], seed=188).reset()
        rendered = TerminalRenderer().render_state(state)

        self.assertIn("Hidden: ??", rendered)
        for player in state.players:
            for card in state.player_states[player.id].hand:
                self.assertNotIn(card.value, rendered)

    def test_viewer_renderer_shows_only_viewers_hand(self) -> None:
        state = LoveLetterEnv(["Alice", "Bob"], seed=188).reset()
        viewer = state.players[0].id
        rendered = TerminalRenderer().render_state(state, viewer=viewer)

        self.assertIn(state.player_states[viewer].hand[0].value, rendered)
        self.assertIn("P1 Bob", rendered)
        self.assertIn("hand: ??", rendered)
