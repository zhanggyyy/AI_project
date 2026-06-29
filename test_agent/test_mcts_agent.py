"""Run a full match with MCTSAgent on both seats."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from loveletter_ai.agents import MCTSAgent
from test_random_agent import run_match


def main() -> None:
    run_match(
        MCTSAgent("MCTS-Alice", simulations=80, rollout_depth=16, debug=True),
        MCTSAgent("MCTS-Bob", simulations=80, rollout_depth=16, debug=True),
    )


if __name__ == "__main__":
    main()
