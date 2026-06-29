from loveletter_ai.agents import ImprovedHeuristicAgent, RandomAgent
from loveletter_ai.evaluation import evaluate_pair, run_round


def run_game(seed: int):
    return run_round(ImprovedHeuristicAgent(), RandomAgent(), seed=seed).winner


def batch_eval(total=1000):
    result = evaluate_pair(
        lambda: ImprovedHeuristicAgent(),
        lambda: RandomAgent(),
        games=total,
    )
    print(f"\nFinal winrate: {result.win_rate_a:.2%}")


if __name__ == "__main__":
    batch_eval(1000)
