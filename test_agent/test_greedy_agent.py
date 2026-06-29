from loveletter_ai.agents import GreedyAgent, RandomAgent
from loveletter_ai.evaluation import evaluate_pair, run_round


def run_single(seed=0):
    return run_round(GreedyAgent(), RandomAgent(), seed=seed).winner


def batch_eval(total=1000):
    result = evaluate_pair(
        lambda: GreedyAgent(),
        lambda: RandomAgent(),
        games=total,
    )
    print(f"\nFinal WinRate: {result.win_rate_a:.2%}")


if __name__ == "__main__":
    batch_eval(1000)
