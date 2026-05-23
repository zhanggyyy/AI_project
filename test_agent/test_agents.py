from random import Random
from unittest import TestCase

from loveletter_ai.actions import Action
from loveletter_ai.agents import (
    ExpectimaxAgent,
    LogicAgent,
    LogicKnowledgeBase,
    NaiveHeuristicAgent,
)
from loveletter_ai.cards import CardName
from loveletter_ai.environment import LoveLetterEnv
from loveletter_ai.players import PlayerId
from loveletter_ai.state import Event, Observation, Phase, PublicPlayerView


class AgentReasoningTest(TestCase):
    def test_naive_agent_plays_handmaid_when_holding_princess(self) -> None:
        agent = NaiveHeuristicAgent()
        observation = make_observation(
            own_hand=(CardName.PRINCESS, CardName.HANDMAID),
        )
        actions = (
            Action(PlayerId(0), CardName.PRINCESS),
            Action(PlayerId(0), CardName.HANDMAID),
        )

        self.assertEqual(
            agent.choose_action(observation, actions, Random(1)).card,
            CardName.HANDMAID,
        )

    def test_logic_kb_removes_failed_guard_guess(self) -> None:
        kb = LogicKnowledgeBase()
        kb.note_action(Action(PlayerId(0), CardName.GUARD, PlayerId(1), CardName.PRINCESS))
        observation = make_observation(
            visible_events=(
                Event("Alice guessed Bob's card."),
                Event("The guess was wrong. Bob stays alive."),
            )
        )

        kb.update(observation)

        self.assertNotIn(CardName.PRINCESS, kb.possible_cards(PlayerId(1)))

    def test_logic_agent_uses_known_card_for_guard(self) -> None:
        agent = LogicAgent()
        observation = make_observation(
            own_hand=(CardName.GUARD, CardName.PRIEST),
            visible_events=(
                Event("Alice saw Bob holding King.", viewer=PlayerId(0)),
            ),
        )
        actions = (
            Action(PlayerId(0), CardName.GUARD, PlayerId(1), CardName.PRINCESS),
            Action(PlayerId(0), CardName.GUARD, PlayerId(1), CardName.KING),
            Action(PlayerId(0), CardName.PRIEST, PlayerId(1)),
        )

        chosen = agent.choose_action(observation, actions, Random(1))

        self.assertEqual(chosen.card, CardName.GUARD)
        self.assertEqual(chosen.guess, CardName.KING)

    def test_expectimax_agent_returns_legal_action_from_env(self) -> None:
        env = LoveLetterEnv(["Alice", "Bob"], seed=188)
        state = env.reset()
        state = env.begin_action_phase()
        actor = state.current_player
        agent = ExpectimaxAgent(depth=2)

        action = agent.choose_action_from_env(env, actor, env.rng)

        self.assertIn(action, env.legal_actions(actor))


def make_observation(
    own_hand=(CardName.GUARD,),
    visible_events=(),
) -> Observation:
    return Observation(
        viewer=PlayerId(0),
        round_number=1,
        turn_index=0,
        phase=Phase.ACTION,
        current_player=PlayerId(0),
        deck_size=10,
        players=(
            PublicPlayerView(
                id=PlayerId(0),
                name="Alice",
                hand_size=len(own_hand),
                discard_pile=(),
                eliminated=False,
                protected=False,
            ),
            PublicPlayerView(
                id=PlayerId(1),
                name="Bob",
                hand_size=1,
                discard_pile=(),
                eliminated=False,
                protected=False,
            ),
        ),
        own_hand=own_hand,
        visible_events=visible_events,
    )
