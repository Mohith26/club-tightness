# How tight is CLUB on the Gaussian benchmark?

CLUB (Cheng, Hao, Dai, Liu, Gan, Carin, ICML 2020, [paper](https://arxiv.org/abs/2006.12013), [code](https://github.com/Linear95/CLUB)) estimates an upper bound on mutual information using a learned conditional q(y|x). I wanted to see how the estimate depends on how well q fits, so I reran the paper's correlated-Gaussian setup at a few q capacities and compared against a closed form.

## Closed form

On the paper's benchmark each coordinate pair (x_i, y_i) is bivariate normal with correlation rho, d = 20. If q is the exact conditional N(rho x, 1 - rho^2), the CLUB objective works out to

    CLUB_exact = d * rho^2 / (1 - rho^2)    while    I(X;Y) = -(d/2) * log(1 - rho^2)

So at the benchmark's MI levels the bound with a perfect q is 4.43, 9.84, 16.44, 24.51 and 34.37 nats for true MI 2, 4, 6, 8 and 10. It is a valid upper bound everywhere, but a loose one, and the gap grows with correlation. `test_club.py` checks this formula against a Monte Carlo estimate.

## Experiment

Same schedule as `mi_estimation.ipynb` in the upstream repo: d = 20, batch 64, Adam with lr 0.005, 4000 steps per MI level, levels trained in sequence, estimate = mean of the last 500 per-step values. The only thing I vary is the size of q. The notebook uses `hidden_size = 15`, which gives each head (mean and log-variance) 15 // 2 = 7 hidden units. My `club.py` reimplements the upstream `CLUB` and `CLUBSample` classes (I checked they give identical outputs and losses for the same weights). 3 seeds, CPU.

Estimated MI (mean +/- std over seeds):

| estimator | q hidden units per head | MI=2 | MI=4 | MI=6 | MI=8 | MI=10 |
|---|---|---|---|---|---|---|
| CLUB | 7 (notebook default) | 1.45 +/- 0.02 | 3.44 +/- 0.03 | 5.83 +/- 0.08 | 8.73 +/- 0.03 | 12.24 +/- 0.09 |
| CLUB | 32 | 4.11 +/- 0.01 | 9.26 +/- 0.05 | 15.56 +/- 0.10 | 23.17 +/- 0.04 | 32.45 +/- 0.03 |
| CLUB | 128 | 4.12 +/- 0.02 | 9.34 +/- 0.02 | 15.67 +/- 0.10 | 23.31 +/- 0.03 | 32.92 +/- 0.02 |
| CLUB | linear q | 4.36 +/- 0.01 | 9.67 +/- 0.03 | 16.21 +/- 0.10 | 24.18 +/- 0.03 | 33.25 +/- 0.05 |
| CLUBSample | 7 (notebook default) | 1.45 +/- 0.02 | 3.45 +/- 0.04 | 5.81 +/- 0.06 | 8.71 +/- 0.01 | 12.24 +/- 0.08 |
| CLUBSample | 128 | 4.12 +/- 0.01 | 9.35 +/- 0.04 | 15.68 +/- 0.13 | 23.34 +/- 0.07 | 32.94 +/- 0.03 |
| CLUB with exact q (closed form) | n/a | 4.43 | 9.84 | 16.44 | 24.51 | 34.37 |

How far each learned q is from the true conditional, as the average log-likelihood gap on 4096 fresh samples (nats per sample):

| estimator | q hidden units per head | MI=2 | MI=4 | MI=6 | MI=8 | MI=10 |
|---|---|---|---|---|---|---|
| CLUB | 7 | 1.37 | 2.57 | 3.82 | 5.01 | 6.20 |
| CLUB | 32 | 0.18 | 0.20 | 0.24 | 0.26 | 0.30 |
| CLUB | 128 | 0.18 | 0.19 | 0.21 | 0.22 | 0.27 |
| CLUB | linear q | 0.12 | 0.12 | 0.12 | 0.11 | 0.11 |

## What I take from it

- With the notebook's q, the estimate lands close to the true MI (1.45 to 12.24 against 2 to 10). That q is far from the true conditional, though: 7 ReLU units cannot represent the 20-dimensional map y ~ rho x, and its log-likelihood is 1.4 to 6.2 nats per sample below the truth.
- Once q actually fits (32+ hidden units, or a linear q, gap under 0.3 nats), both CLUB and CLUBSample climb to within about 7% of the closed-form exact-q value, which is 3.4x the true MI at MI = 10.
- So on this benchmark the close tracking comes from an underfit q, and a better q makes the number more faithful to the bound but further from the MI. For MI minimization that may not matter much, since the bound is still minimized at independence, but it matters if the number is read as an MI estimate.

## Limits

Only the Gaussian setting, not the cubic one or the downstream MI-minimization tasks. Three seeds, one learning rate. I did not rerun the other estimators from the paper (MINE, NWJ, InfoNCE, L1Out).

## Run it

    python -m venv .venv && .venv/bin/pip install -r requirements.txt
    .venv/bin/python -m pytest -q          # closed form and implementation checks
    .venv/bin/python bench.py --seeds 3    # about 3.5 minutes on a laptop CPU
    .venv/bin/python report.py results.json
