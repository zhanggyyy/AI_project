"""Quick benchmark of integrated agents against RandomAgent."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from loveletter_ai.agents import (
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
from loveletter_ai.training import train_q_learning_self_play


def main() -> None:
    games_per_seat = int(sys.argv[1]) if len(sys.argv) > 1 else 25
    trained_q = train_q_learning_self_play(episodes=500, seed=188, epsilon=0.25)

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
        "qlearning": lambda: QLearningAgent(
            "QLearning",
            epsilon=0.0,
            q_values=trained_q.q_values,
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
