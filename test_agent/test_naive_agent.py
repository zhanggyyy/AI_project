"""Run a full match with NaiveHeuristicAgent on both seats."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from loveletter_ai.agents import NaiveHeuristicAgent
from test_random_agent import run_match


def main() -> None:
    run_match(
        NaiveHeuristicAgent("Naive-Alice"),
        NaiveHeuristicAgent("Naive-Bob"),
    )


if __name__ == "__main__":
    main()
