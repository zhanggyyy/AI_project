from random import Random
from unittest import TestCase

from loveletter_ai.belief_expectimax import BayesianBelief, BeliefExpectimaxAgent
from loveletter_ai.cards import CardName
from loveletter_ai.environment import LoveLetterEnv
from loveletter_ai.players import PlayerId
from loveletter_ai.state import Observation, Phase, PublicPlayerView


class BeliefExpectimaxAgentTest(TestCase):
    def test_belief_distribution_excludes_visible_cards(self) -> None:
        observation = make_observation(own_hand=(CardName.PRINCESS, CardName.GUARD))
        belief = BayesianBelief.from_observation(observation)

        distribution = belief.opponent_hand_distribution(PlayerId(1))

        self.assertNotIn(CardName.PRINCESS, distribution)
        self.assertAlmostEqual(sum(distribution.values()), 1.0)

    def test_hidden_state_sample_matches_public_sizes(self) -> None:
        observation = make_observation(own_hand=(CardName.PRIEST, CardName.HANDMAID))
        belief = BayesianBelief.from_observation(
            observation,
            {PlayerId(1): {CardName.KING}},
        )

        sample = belief.sample_hidden_state(Random(7))

        self.assertEqual(sample.opponent_hands[PlayerId(1)], (CardName.KING,))
        self.assertEqual(len(sample.deck), observation.deck_size)

    def test_agent_returns_legal_action_from_env(self) -> None:
        env = LoveLetterEnv(["Alice", "Bob"], seed=188)
        state = env.reset()
        state = env.begin_action_phase()
        actor = state.current_player
        agent = BeliefExpectimaxAgent(depth=2, samples=5)

        action = agent.choose_action_from_env(env, actor, env.rng)

        self.assertIn(action, env.legal_actions(actor))


def make_observation(
    own_hand=(CardName.GUARD,),
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
        visible_events=(),
    )
