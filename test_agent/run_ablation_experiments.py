"""Generate report-ready ablation tables for the Love Letter agents."""

from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from loveletter_ai.agents import (  # noqa: E402
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
from loveletter_ai.belief_expectimax import BeliefExpectimaxAgent  # noqa: E402
from loveletter_ai.evaluation import evaluate_pair_symmetric  # noqa: E402
from loveletter_ai.training import train_q_learning_self_play  # noqa: E402


def main() -> None:
    games_per_seat = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    output_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("outputs/experiments")
    profile = sys.argv[3].lower() if len(sys.argv) > 3 else "quick"
    if profile not in {"quick", "full"}:
        raise ValueError("profile must be 'quick' or 'full'")
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    rows.extend(run_fair_baselines(games_per_seat, profile))
    rows.extend(run_belief_expectimax_ablation(games_per_seat, profile))
    rows.extend(run_belief_mcts_ablation(games_per_seat, profile))
    rows.extend(run_q_learning_ablation(games_per_seat, profile))
    rows.extend(run_oracle_ablation(games_per_seat, profile))

    csv_path = output_dir / "ablation_results.csv"
    md_path = output_dir / "ablation_results.md"
    write_csv(csv_path, rows)
    write_markdown(md_path, rows, games_per_seat, profile)

    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")


def run_fair_baselines(games_per_seat: int, profile: str) -> list[dict[str, str]]:
    q_episodes = 500 if profile == "quick" else 1500
    belief_samples = 6 if profile == "quick" else 12
    belief_mcts_sims = 12 if profile == "quick" else 32
    trained_q = train_q_learning_self_play(episodes=q_episodes, seed=188, epsilon=0.25)
    factories = {
        "Naive": lambda: NaiveHeuristicAgent("Naive"),
        "Logic": lambda: LogicAgent("Logic"),
        "Greedy": lambda: GreedyAgent("Greedy"),
        "Improved": lambda: ImprovedHeuristicAgent("Improved"),
        "BeliefExpectimax": lambda: BeliefExpectimaxAgent(
            "BeliefExpectimax",
            depth=2,
            samples=belief_samples,
        ),
        "BeliefMCTS": lambda: BeliefMCTSAgent(
            "BeliefMCTS",
            simulations=belief_mcts_sims,
            rollout_depth=8,
        ),
        "Q-learning": lambda: QLearningAgent(
            "QLearning",
            epsilon=0.0,
            q_values=trained_q.q_values,
        ),
    }
    rows = []
    for name, factory in factories.items():
        rows.append(
            evaluate_row(
                "fair_vs_random",
                name,
                factory,
                lambda: RandomAgent("Random"),
                games_per_seat,
                seed=10_000,
            )
        )
    return rows


def run_belief_expectimax_ablation(
    games_per_seat: int,
    profile: str,
) -> list[dict[str, str]]:
    rows = []
    sample_values = (1, 4, 8) if profile == "quick" else (1, 4, 8, 16, 32)
    depth_values = (1, 2) if profile == "quick" else (1, 2, 3)
    for samples in sample_values:
        rows.append(
            evaluate_row(
                "belief_expectimax_samples",
                f"samples={samples}",
                lambda samples=samples: BeliefExpectimaxAgent(
                    "BeliefExpectimax",
                    depth=2,
                    samples=samples,
                ),
                lambda: RandomAgent("Random"),
                games_per_seat,
                seed=20_000 + samples,
                param=str(samples),
            )
        )
    for depth in depth_values:
        rows.append(
            evaluate_row(
                "belief_expectimax_depth",
                f"depth={depth}",
                lambda depth=depth: BeliefExpectimaxAgent(
                    "BeliefExpectimax",
                    depth=depth,
                    samples=8,
                ),
                lambda: RandomAgent("Random"),
                games_per_seat,
                seed=21_000 + depth,
                param=str(depth),
            )
        )
    return rows


def run_belief_mcts_ablation(
    games_per_seat: int,
    profile: str,
) -> list[dict[str, str]]:
    rows = []
    simulation_values = (4, 12, 24) if profile == "quick" else (4, 12, 24, 48, 96)
    for simulations in simulation_values:
        rows.append(
            evaluate_row(
                "belief_mcts_simulations",
                f"simulations={simulations}",
                lambda simulations=simulations: BeliefMCTSAgent(
                    "BeliefMCTS",
                    simulations=simulations,
                    rollout_depth=8,
                ),
                lambda: RandomAgent("Random"),
                games_per_seat,
                seed=30_000 + simulations,
                param=str(simulations),
            )
        )
    return rows


def run_q_learning_ablation(games_per_seat: int, profile: str) -> list[dict[str, str]]:
    rows = []
    episode_values = (0, 100, 500) if profile == "quick" else (0, 100, 500, 1000, 3000)
    for episodes in episode_values:
        trained_q = train_q_learning_self_play(
            episodes=episodes,
            seed=188,
            epsilon=0.25,
        )
        rows.append(
            evaluate_row(
                "q_learning_episodes",
                f"episodes={episodes}",
                lambda trained_q=trained_q: QLearningAgent(
                    "QLearning",
                    epsilon=0.0,
                    q_values=trained_q.q_values,
                ),
                lambda: RandomAgent("Random"),
                games_per_seat,
                seed=40_000 + episodes,
                param=str(episodes),
            )
        )
    return rows


def run_oracle_ablation(games_per_seat: int, profile: str) -> list[dict[str, str]]:
    rows = []
    depth_values = (1, 2) if profile == "quick" else (1, 2, 3)
    mcts_values = (4, 12) if profile == "quick" else (4, 12, 24, 48)
    for depth in depth_values:
        rows.append(
            evaluate_row(
                "oracle_expectimax_depth",
                f"depth={depth}",
                lambda depth=depth: ExpectimaxAgent("ExpectimaxOracle", depth=depth),
                lambda: RandomAgent("Random"),
                games_per_seat,
                seed=50_000 + depth,
                param=str(depth),
            )
        )
    for simulations in mcts_values:
        rows.append(
            evaluate_row(
                "oracle_mcts_simulations",
                f"simulations={simulations}",
                lambda simulations=simulations: MCTSAgent(
                    "MCTSOracle",
                    simulations=simulations,
                    rollout_depth=8,
                ),
                lambda: RandomAgent("Random"),
                games_per_seat,
                seed=51_000 + simulations,
                param=str(simulations),
            )
        )
    return rows


def evaluate_row(
    section: str,
    label: str,
    agent_factory,
    opponent_factory,
    games_per_seat: int,
    seed: int,
    param: str = "",
) -> dict[str, str]:
    result = evaluate_pair_symmetric(
        agent_factory,
        opponent_factory,
        games_per_seat=games_per_seat,
        seed=seed,
    )
    rate = result.win_rate_a
    stderr = math.sqrt(rate * (1.0 - rate) / result.games) if result.games else 0.0
    ci_low, ci_high = wilson_interval(result.wins_a, result.games)
    return {
        "section": section,
        "label": label,
        "param": param,
        "games": str(result.games),
        "wins": str(result.wins_a),
        "losses": str(result.wins_b),
        "draws": str(result.draws),
        "win_rate": f"{rate:.4f}",
        "stderr": f"{stderr:.4f}",
        "ci95_low": f"{ci_low:.4f}",
        "ci95_high": f"{ci_high:.4f}",
    }


def wilson_interval(wins: int, games: int, z: float = 1.96) -> tuple[float, float]:
    if games <= 0:
        return 0.0, 0.0
    p_hat = wins / games
    denominator = 1.0 + z * z / games
    center = (p_hat + z * z / (2 * games)) / denominator
    margin = (
        z
        * math.sqrt((p_hat * (1.0 - p_hat) + z * z / (4 * games)) / games)
        / denominator
    )
    return max(0.0, center - margin), min(1.0, center + margin)


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = [
        "section",
        "label",
        "param",
        "games",
        "wins",
        "losses",
        "draws",
        "win_rate",
        "stderr",
        "ci95_low",
        "ci95_high",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(
    path: Path,
    rows: list[dict[str, str]],
    games_per_seat: int,
    profile: str,
) -> None:
    sections = [
        ("fair_vs_random", "Fair Hidden-Information Agents vs Random"),
        ("belief_expectimax_samples", "Ablation: BeliefExpectimax Samples"),
        ("belief_expectimax_depth", "Ablation: BeliefExpectimax Depth"),
        ("belief_mcts_simulations", "Ablation: BeliefMCTS Simulations"),
        ("q_learning_episodes", "Ablation: Q-learning Episodes"),
        ("oracle_expectimax_depth", "Oracle Baseline: Expectimax Depth"),
        ("oracle_mcts_simulations", "Oracle Baseline: MCTS Simulations"),
    ]
    lines = [
        "# Love Letter Agent Ablation Results",
        "",
        f"Each row uses symmetric evaluation with {games_per_seat} games per seat.",
        f"Profile: `{profile}`.",
        "Win rate is for the first listed agent against RandomAgent unless stated otherwise.",
        "",
    ]
    for key, title in sections:
        section_rows = [row for row in rows if row["section"] == key]
        if not section_rows:
            continue
        lines.extend([f"## {title}", "", "| Setting | Games | W-L-D | Win Rate | 95% CI |", "| --- | ---: | ---: | ---: | ---: |"])
        for row in section_rows:
            ci = f"{percent(row['ci95_low'])}-{percent(row['ci95_high'])}"
            record = f"{row['wins']}-{row['losses']}-{row['draws']}"
            lines.append(
                f"| {row['label']} | {row['games']} | {record} | "
                f"{percent(row['win_rate'])} | {ci} |"
            )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def percent(value: str) -> str:
    return f"{float(value) * 100:.1f}%"


if __name__ == "__main__":
    main()
