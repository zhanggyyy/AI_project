"""Run a full match with trained QLearningAgent on both seats."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from loveletter_ai.agents import ApproximateQLearningAgent
from loveletter_ai.training import train_q_learning_mixed, train_q_learning_self_play
from test_random_agent import run_match


def main() -> None:
    agent_a = train_q_learning_self_play(episodes=200, seed=188, epsilon=0.25)
    agent_b = train_q_learning_self_play(episodes=200, seed=189, epsilon=0.25)
    agent_a.name = "Q-Alice"
    agent_b.name = "Q-Bob"
    run_match(agent_a, agent_b)


def test_mixed_approximate_q_training_updates_weights() -> None:
    agent = train_q_learning_mixed(
        episodes=10,
        seed=188,
        agent_kind="approximate",
        epsilon_start=0.2,
        epsilon_end=0.02,
        opponent_pool=("random", "naive", "self"),
    )

    assert isinstance(agent, ApproximateQLearningAgent)
    assert agent.epsilon == 0.0
    assert len(agent.weights) > 2


if __name__ == "__main__":
    main()
