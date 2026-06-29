"""Quick benchmark of integrated agents against RandomAgent."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from loveletter_ai.agents import (
    ApproximateQLearningAgent,
    BeliefMCTSAgent,
    ExpectimaxAgent,
    GreedyAgent,
    ImprovedHeuristicAgent,
    LogicAgent,
    MCTSAgent,
    NaiveHeuristicAgent,
    QLearningAgent,
    RandomAgent,
)
from loveletter_ai.belief_expectimax import BeliefExpectimaxAgent
from loveletter_ai.evaluation import evaluate_pair_symmetric
from loveletter_ai.training import train_q_learning_mixed


def main() -> None:
    games_per_seat = int(sys.argv[1]) if len(sys.argv) > 1 else 25
    trained_q = train_q_learning_mixed(
        episodes=1000,
        seed=188,
        agent_kind="tabular",
        epsilon_start=0.4,
        epsilon_end=0.02,
        opponent_pool=("random", "naive", "improved", "self"),
    )
    trained_approx_q = train_q_learning_mixed(
        episodes=5000,
        seed=777,
        agent_kind="approximate",
        epsilon_start=0.45,
        epsilon_end=0.02,
        alpha=0.025,
        opponent_pool=("random", "naive", "self"),
    )

    fair_factories = {
        "naive": lambda: NaiveHeuristicAgent("Naive"),
        "logic": lambda: LogicAgent("Logic"),
        "greedy": lambda: GreedyAgent("Greedy"),
        "improved": lambda: ImprovedHeuristicAgent("Improved"),
        "belief_expectimax": lambda: BeliefExpectimaxAgent(
            "BeliefExpectimax",
            depth=2,
            samples=6,
        ),
        "belief_mcts": lambda: BeliefMCTSAgent(
            "BeliefMCTS",
            simulations=16,
            rollout_depth=6,
        ),
        "tabular_q_prior": lambda: QLearningAgent(
            "TabularQPrior",
            epsilon=0.0,
            q_values=trained_q.q_values,
            visit_counts=trained_q.visit_counts,
            heuristic_prior_weight=0.35,
        ),
        "approx_q_shaping": lambda: ApproximateQLearningAgent(
            "ApproxQShaping",
            epsilon=0.0,
            weights=trained_approx_q.weights,
            initial_heuristic_weight=0.0,
        ),
    }
    oracle_factories = {
        "expectimax_oracle": lambda: ExpectimaxAgent("ExpectimaxOracle", depth=2),
        "mcts": lambda: MCTSAgent("MCTS", simulations=12, rollout_depth=6),
    }

    print(
        "Fair hidden-information benchmark vs RandomAgent "
        f"over {games_per_seat * 2} games"
    )
    print_section(fair_factories, games_per_seat)

    print(
        "\nOracle / perfect-information upper-bound benchmark vs RandomAgent "
        f"over {games_per_seat * 2} games"
    )
    print_section(oracle_factories, games_per_seat)


def print_section(factories, games_per_seat: int) -> None:
    for label, factory in factories.items():
        result = evaluate_pair_symmetric(
            factory,
            lambda: RandomAgent("Random"),
            games_per_seat=games_per_seat,
        )
        print(
            f"{label:18s} "
            f"wins={result.wins_a:3d} losses={result.wins_b:3d} "
            f"draws={result.draws:3d} win_rate={result.win_rate_a:.1%}"
        )


if __name__ == "__main__":
    main()
