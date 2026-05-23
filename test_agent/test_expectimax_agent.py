"""Run a full match with ExpectimaxAgent on both seats.

This is the first determinized expectimax baseline. It searches cloned concrete
states, so it is useful for validating the expectimax control flow before we add
belief-state sampling for hidden information.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from loveletter_ai.agents import ExpectimaxAgent
from test_random_agent import run_match


def main() -> None:
    run_match(
        ExpectimaxAgent("Expectimax-Alice", depth=3, debug=True),
        ExpectimaxAgent("Expectimax-Bob", depth=3, debug=True),
    )


if __name__ == "__main__":
    main()
