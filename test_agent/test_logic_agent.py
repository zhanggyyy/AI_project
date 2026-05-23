"""Run a full match with LogicAgent on both seats and print deductions."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from loveletter_ai.agents import LogicAgent
from test_random_agent import run_match


def main() -> None:
    run_match(
        LogicAgent("Logic-Alice", debug=True),
        LogicAgent("Logic-Bob", debug=True),
    )


if __name__ == "__main__":
    main()
