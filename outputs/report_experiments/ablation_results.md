# Love Letter Agent Ablation Results

Each row uses symmetric evaluation with 20 games per seat.
Profile: `quick`.
Win rate is for the first listed agent against RandomAgent unless stated otherwise.

## Fair Hidden-Information Agents vs Random

| Setting | Games | W-L-D | Win Rate | 95% CI |
| --- | ---: | ---: | ---: | ---: |
| Naive | 40 | 27-13-0 | 67.5% | 52.0%-79.9% |
| Logic | 40 | 31-9-0 | 77.5% | 62.5%-87.7% |
| Greedy | 40 | 31-9-0 | 77.5% | 62.5%-87.7% |
| Improved | 40 | 31-9-0 | 77.5% | 62.5%-87.7% |
| BeliefExpectimax | 40 | 26-14-0 | 65.0% | 49.5%-77.9% |
| BeliefMCTS | 40 | 30-10-0 | 75.0% | 59.8%-85.8% |
| Tabular Q+prior | 40 | 31-9-0 | 77.5% | 62.5%-87.7% |
| Approx Q+shaping | 40 | 31-9-0 | 77.5% | 62.5%-87.7% |

## Ablation: BeliefExpectimax Samples

| Setting | Games | W-L-D | Win Rate | 95% CI |
| --- | ---: | ---: | ---: | ---: |
| samples=1 | 40 | 30-10-0 | 75.0% | 59.8%-85.8% |
| samples=4 | 40 | 29-11-0 | 72.5% | 57.2%-83.9% |
| samples=8 | 40 | 30-10-0 | 75.0% | 59.8%-85.8% |

## Ablation: BeliefExpectimax Depth

| Setting | Games | W-L-D | Win Rate | 95% CI |
| --- | ---: | ---: | ---: | ---: |
| depth=1 | 40 | 27-13-0 | 67.5% | 52.0%-79.9% |
| depth=2 | 40 | 27-13-0 | 67.5% | 52.0%-79.9% |

## Ablation: BeliefMCTS Simulations

| Setting | Games | W-L-D | Win Rate | 95% CI |
| --- | ---: | ---: | ---: | ---: |
| simulations=4 | 40 | 26-14-0 | 65.0% | 49.5%-77.9% |
| simulations=12 | 40 | 22-18-0 | 55.0% | 39.8%-69.3% |
| simulations=24 | 40 | 27-13-0 | 67.5% | 52.0%-79.9% |

## Ablation: Q-learning Variants

| Setting | Games | W-L-D | Win Rate | 95% CI |
| --- | ---: | ---: | ---: | ---: |
| heuristic only | 40 | 30-10-0 | 75.0% | 59.8%-85.8% |
| pure tabular Q | 40 | 24-16-0 | 60.0% | 44.6%-73.7% |
| tabular Q + prior | 40 | 30-10-0 | 75.0% | 59.8%-85.8% |
| approx Q + shaping | 40 | 31-9-0 | 77.5% | 62.5%-87.7% |

## Ablation: Approximate Q-learning Episodes

| Setting | Games | W-L-D | Win Rate | 95% CI |
| --- | ---: | ---: | ---: | ---: |
| episodes=0 | 40 | 29-11-0 | 72.5% | 57.2%-83.9% |
| episodes=500 | 40 | 30-10-0 | 75.0% | 59.8%-85.8% |
| episodes=1000 | 40 | 30-10-0 | 75.0% | 59.8%-85.8% |
| episodes=5000 | 40 | 31-9-0 | 77.5% | 62.5%-87.7% |

## Oracle Baseline: Expectimax Depth

| Setting | Games | W-L-D | Win Rate | 95% CI |
| --- | ---: | ---: | ---: | ---: |
| depth=1 | 40 | 30-10-0 | 75.0% | 59.8%-85.8% |
| depth=2 | 40 | 33-7-0 | 82.5% | 68.0%-91.2% |

## Oracle Baseline: MCTS Simulations

| Setting | Games | W-L-D | Win Rate | 95% CI |
| --- | ---: | ---: | ---: | ---: |
| simulations=4 | 40 | 32-8-0 | 80.0% | 65.2%-89.5% |
| simulations=12 | 40 | 32-8-0 | 80.0% | 65.2%-89.5% |
