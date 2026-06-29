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
| BeliefExpectimax | 10 | 5-5-0 | 50.0% | 23.7%-76.3% |
| BeliefMCTS | 10 | 7-3-0 | 70.0% | 39.7%-89.2% |
| Q-learning | 10 | 8-2-0 | 80.0% | 49.0%-94.3% |

## Ablation: BeliefExpectimax Samples

| Setting | Games | W-L-D | Win Rate | 95% CI |
| --- | ---: | ---: | ---: | ---: |
| samples=1 | 10 | 7-3-0 | 70.0% | 39.7%-89.2% |
| samples=4 | 10 | 8-2-0 | 80.0% | 49.0%-94.3% |
| samples=8 | 10 | 7-3-0 | 70.0% | 39.7%-89.2% |

## Ablation: BeliefExpectimax Depth

| Setting | Games | W-L-D | Win Rate | 95% CI |
| --- | ---: | ---: | ---: | ---: |
| depth=1 | 10 | 8-2-0 | 80.0% | 49.0%-94.3% |
| depth=2 | 10 | 9-1-0 | 90.0% | 59.6%-98.2% |

## Ablation: BeliefMCTS Simulations

| Setting | Games | W-L-D | Win Rate | 95% CI |
| --- | ---: | ---: | ---: | ---: |
| simulations=4 | 10 | 8-2-0 | 80.0% | 49.0%-94.3% |
| simulations=12 | 10 | 5-5-0 | 50.0% | 23.7%-76.3% |
| simulations=24 | 10 | 8-2-0 | 80.0% | 49.0%-94.3% |

## Ablation: Q-learning Episodes

| Setting | Games | W-L-D | Win Rate | 95% CI |
| --- | ---: | ---: | ---: | ---: |
| episodes=0 | 10 | 5-5-0 | 50.0% | 23.7%-76.3% |
| episodes=100 | 10 | 6-4-0 | 60.0% | 31.3%-83.2% |
| episodes=500 | 10 | 5-5-0 | 50.0% | 23.7%-76.3% |

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
