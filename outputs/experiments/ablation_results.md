# Love Letter Agent Ablation Results

Each row uses symmetric evaluation with 5 games per seat.
Profile: `quick`.
Win rate is for the first listed agent against RandomAgent unless stated otherwise.

## Fair Hidden-Information Agents vs Random

| Setting | Games | W-L-D | Win Rate | 95% CI |
| --- | ---: | ---: | ---: | ---: |
| Naive | 10 | 7-3-0 | 70.0% | 39.7%-89.2% |
| Logic | 10 | 8-2-0 | 80.0% | 49.0%-94.3% |
| Greedy | 10 | 8-2-0 | 80.0% | 49.0%-94.3% |
| Improved | 10 | 8-2-0 | 80.0% | 49.0%-94.3% |
| BeliefExpectimax | 10 | 7-3-0 | 70.0% | 39.7%-89.2% |
| BeliefMCTS | 10 | 7-3-0 | 70.0% | 39.7%-89.2% |
| Tabular Q+prior | 10 | 7-3-0 | 70.0% | 39.7%-89.2% |
| Approx Q+shaping | 10 | 8-2-0 | 80.0% | 49.0%-94.3% |

## Ablation: BeliefExpectimax Samples

| Setting | Games | W-L-D | Win Rate | 95% CI |
| --- | ---: | ---: | ---: | ---: |
| samples=1 | 10 | 7-3-0 | 70.0% | 39.7%-89.2% |
| samples=4 | 10 | 9-1-0 | 90.0% | 59.6%-98.2% |
| samples=8 | 10 | 7-3-0 | 70.0% | 39.7%-89.2% |

## Ablation: BeliefExpectimax Depth

| Setting | Games | W-L-D | Win Rate | 95% CI |
| --- | ---: | ---: | ---: | ---: |
| depth=1 | 10 | 9-1-0 | 90.0% | 59.6%-98.2% |
| depth=2 | 10 | 9-1-0 | 90.0% | 59.6%-98.2% |

## Ablation: BeliefMCTS Simulations

| Setting | Games | W-L-D | Win Rate | 95% CI |
| --- | ---: | ---: | ---: | ---: |
| simulations=4 | 10 | 7-3-0 | 70.0% | 39.7%-89.2% |
| simulations=12 | 10 | 7-3-0 | 70.0% | 39.7%-89.2% |
| simulations=24 | 10 | 9-1-0 | 90.0% | 59.6%-98.2% |

## Ablation: Q-learning Variants

| Setting | Games | W-L-D | Win Rate | 95% CI |
| --- | ---: | ---: | ---: | ---: |
| heuristic only | 10 | 7-3-0 | 70.0% | 39.7%-89.2% |
| pure tabular Q | 10 | 5-5-0 | 50.0% | 23.7%-76.3% |
| tabular Q + prior | 10 | 7-3-0 | 70.0% | 39.7%-89.2% |
| approx Q + shaping | 10 | 6-4-0 | 60.0% | 31.3%-83.2% |

## Ablation: Approximate Q-learning Episodes

| Setting | Games | W-L-D | Win Rate | 95% CI |
| --- | ---: | ---: | ---: | ---: |
| episodes=0 | 10 | 7-3-0 | 70.0% | 39.7%-89.2% |
| episodes=500 | 10 | 7-3-0 | 70.0% | 39.7%-89.2% |
| episodes=1000 | 10 | 6-4-0 | 60.0% | 31.3%-83.2% |
| episodes=5000 | 10 | 6-4-0 | 60.0% | 31.3%-83.2% |

## Oracle Baseline: Expectimax Depth

| Setting | Games | W-L-D | Win Rate | 95% CI |
| --- | ---: | ---: | ---: | ---: |
| depth=1 | 10 | 8-2-0 | 80.0% | 49.0%-94.3% |
| depth=2 | 10 | 8-2-0 | 80.0% | 49.0%-94.3% |

## Oracle Baseline: MCTS Simulations

| Setting | Games | W-L-D | Win Rate | 95% CI |
| --- | ---: | ---: | ---: | ---: |
| simulations=4 | 10 | 9-1-0 | 90.0% | 59.6%-98.2% |
| simulations=12 | 10 | 6-4-0 | 60.0% | 31.3%-83.2% |
