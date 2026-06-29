from unittest import TestCase

from loveletter_ai.agents import (
    BeliefMCTSAgent,
    GreedyAgent,
    ImprovedHeuristicAgent,
    MCTSAgent,
    QLearningAgent,
    RandomAgent,
)
from loveletter_ai.belief_expectimax import BeliefExpectimaxAgent
from loveletter_ai.evaluation import evaluate_pair, evaluate_pair_symmetric, run_round
from loveletter_ai.players import PlayerId


class IntegratedAgentsTest(TestCase):
    def test_rule_agents_finish_rounds_through_shared_runner(self) -> None:
        result = run_round(GreedyAgent(), RandomAgent(), seed=11)
        self.assertIn(result.winner, (PlayerId(0), PlayerId(1), None))

        result = run_round(ImprovedHeuristicAgent(), RandomAgent(), seed=12)
        self.assertIn(result.winner, (PlayerId(0), PlayerId(1), None))

    def test_search_and_rl_agents_finish_rounds_through_shared_runner(self) -> None:
        result = run_round(MCTSAgent(simulations=3, rollout_depth=3), RandomAgent(), seed=13)
        self.assertIn(result.winner, (PlayerId(0), PlayerId(1), None))

        result = run_round(
            BeliefExpectimaxAgent(depth=1, samples=2),
            RandomAgent(),
            seed=14,
        )
        self.assertIn(result.winner, (PlayerId(0), PlayerId(1), None))

        result = run_round(BeliefMCTSAgent(simulations=3, rollout_depth=3), RandomAgent(), seed=16)
        self.assertIn(result.winner, (PlayerId(0), PlayerId(1), None))

        result = run_round(QLearningAgent(epsilon=0.0), RandomAgent(), seed=15)
        self.assertIn(result.winner, (PlayerId(0), PlayerId(1), None))

    def test_evaluate_pair_aggregates_results(self) -> None:
        result = evaluate_pair(
            lambda: GreedyAgent(),
            lambda: RandomAgent(),
            games=3,
            seed=21,
        )
        self.assertEqual(result.games, 3)
        self.assertEqual(result.wins_a + result.wins_b + result.draws, 3)

    def test_symmetric_evaluation_counts_both_seats(self) -> None:
        result = evaluate_pair_symmetric(
            lambda: GreedyAgent(),
            lambda: RandomAgent(),
            games_per_seat=2,
            seed=31,
        )
        self.assertEqual(result.games, 4)
        self.assertEqual(result.wins_a + result.wins_b + result.draws, 4)
