# Learning to Bid in Ad Auctions

A published ad-auction simulation stack that no longer installed on any current Python. I found the dependency deadlock, cut it out without moving the numerics, and proved the port by running 600,000 impression opportunities of the second-price baseline end to end.

## What I did

The stack ships pinned to a 2023 snapshot: `numpy==1.22.0`, `numba==0.55.1`, `llvmlite==0.38.0`, `torch==2.6.0`. That set is now unresolvable, and the failure is structural rather than cosmetic. Numba requires an exactly-matching `llvmlite` build and trails NumPy's ABI by a release or two, so `numba==0.55.1` will not compile against any NumPy modern enough to satisfy a current `torch`. Loosening the pins does not help either — the resolver just walks into the same wall from the other side. Pip dies before a single line of simulation runs.

Chasing the pin forward is the obvious move and it is the wrong one. It buys a working install for a few months and re-breaks on the next NumPy ABI bump. I went after the dependency itself.

Numba turns out to earn its place in exactly one line of this codebase: a `@jit(nopython=True)` decorator on the scalar `sigmoid` in `src/Models.py`. That single function is genuinely load-bearing. `Auction.simulate_opportunity` calls it once per impression opportunity to turn the context/embedding dot product into a true CTR, and `OracleAllocator.estimate_CTR` calls it again on every bid. Under the baseline configuration that is on the order of a million invocations, which is presumably why it was compiled in the first place.

So the question was whether the JIT could come out without changing what the simulator computes. It can, and the reason is specific. The body is `1.0 / (1.0 + np.exp(-x))` — pure NumPy, no Python-level loop, and it is already handed the whole embedding matrix at once. Numba was eliminating per-call interpreter overhead on a function that is called with arrays, so that overhead was amortised across the batch before numba ever saw it. NumPy's ufunc performs the same arithmetic in the same order on the same dtypes. Removing the decorator therefore costs wall-clock and nothing else; the outputs are bit-comparable.

I replaced the import with a no-op decorator that passes the function through untouched, handling both the bare `@jit` and parameterised `@jit(nopython=True)` call forms so the existing call site needed no edit. `requirements-loose.txt` carries the same package set with the pins stripped, letting the resolver assemble a coherent modern stack instead of reconstructing a 2023 one that no longer exists in compatible form.

Then I verified it, because a port that installs is not the same as a port that is correct:

```
python src/main.py config/SP_Oracle.json
```

Six truthful oracle bidders holding 12 items each, two participants per round, second-price allocation, 5-dimensional item embeddings observed at rank 4, 10,000 rounds per iteration across 20 iterations and 3 independent seeds — 600,000 impression opportunities. It completes and writes five metric CSVs (gross utility, net utility, overbid regret, underbid regret, combined results) plus fifteen plots covering allocation regret, estimation regret, auction revenue, CTR bias and CTR RMSE.

I want the scope of that claim to be exact: this establishes that the environment reproduces under the port, not that I re-derived the original paper's findings. What this repository is, concretely, is an ad-auction simulator that installs and runs today, with its numerics intact and its baseline experiment demonstrably completing.

**Not yet built:** the budget-constrained bidding layer. `Agent.charge()` accumulates spend, but nothing reads it back as a constraint — there is no per-agent budget, no pacing signal, no spend trajectory. Bidding under a hard budget is the problem I want this harness pointed at, and it is not in this tree yet. I would rather say so than imply otherwise.

## How the simulator works

`main.py` parses a JSON config through `parse_config`, drawing item embeddings and per-agent item values, then `instantiate_agents` and `instantiate_auction` construct the population. `simulation_run` drives the outer loop: run `rounds_per_iter` opportunities, call each agent's `update` to retrain on what it logged, repeat for `num_iter` iterations, all inside `num_runs` independent seeds.

A single round is `Auction.simulate_opportunity`. It samples a user context, draws `num_participants_per_round` agents to compete, and computes each one's true CTR through the `sigmoid` call described above. Each participating `Agent` runs `select_item` to choose what to bid on using its allocator's CTR estimate, then `bid` to convert estimated value into a bid. `AuctionAllocation.py` supplies `FirstPrice` and `SecondPrice`, both implementing `allocate` to return winners and clearing prices; the winner is charged via `Agent.charge`, and the round is recorded as an `ImpressionOpportunity` holding the true CTR, the estimated CTR, the bid, the price and the realised click. Every regret metric is derived from those records afterwards.

The two learned halves are deliberately separate. `BidderAllocation.py` answers *what is this impression worth* — `OracleAllocator` reads true embeddings, `PyTorchLogisticRegressionAllocator` learns them via Bayesian logistic regression with a Laplace approximation, following Chapelle & Li's Thompson sampling algorithm. `Bidder.py` answers *what should I bid given that value* — `TruthfulBidder` bids value outright, `EmpiricalShadedBidder` shades by a factor tuned against observed win rates, and `ValueLearningBidder`, `PolicyLearningBidder` and `DoublyRobustBidder` learn the shading factor from logged bandit feedback.

Those learned bidders share `BidShadingContextualBandit` in `src/Models.py`, which parameterises a Gaussian over the shading factor γ conditioned on `(P(click), value)`. `initialise_policy` first fits it to imitate the logging policy so propensities do not collapse to zero, after which its `loss` implements the off-policy objectives under comparison: `REINFORCE`, `REINFORCE_offpolicy`, `TRPO` with a KL penalty, `PPO` with clipped importance weights, and `Doubly Robust`, which pairs a clipped-IPS correction with a direct-method term from `PyTorchWinRateEstimator`. Keeping value estimation apart from the shading policy is what makes allocation regret and estimation regret separately measurable — and it is the seam a budget controller would slot into.

## Repo layout

| Path | Origin | Contents |
| --- | --- | --- |
| `requirements-loose.txt` | mine | Unpinned dependency set that resolves on a current stack |
| `README.md` | mine | This file |
| `src/Models.py` | modified by me | Numba `jit` import replaced with a pass-through decorator |
| `src/` (rest) | upstream | Auction loop, agents, bidders, allocators, notebooks |
| `config/` | upstream | Six experiment configs, first- and second-price |
| `results/` | generated | Run outputs; git-ignored |
| `requirements.txt` | upstream | Original 2023 pins, kept for reference |
| `LICENSE`, `NOTICE`, `CITATION.cff` | upstream | Apache-2.0 terms and copyright |
| `CONFIG.md`, `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md` | upstream | Project docs |

`src/Models.py` is the only file under `src/` I have touched.

## Running it

```
python -m venv .venv
.venv\Scripts\activate          # Windows; source .venv/bin/activate elsewhere
pip install -r requirements-loose.txt
```

Install from `requirements-loose.txt`, not `requirements.txt` — the pinned file is the 2023 set and will fail to resolve, which is the whole problem this repo solves.

Run any config from the repo root:

```
python src/main.py config/SP_Oracle.json
```

`config/` holds six. `SP_Oracle.json` and `SP_Truthful_TS.json` are second-price; `FP_DM_Oracle.json`, `FP_DM_TS.json`, `FP_IPS_TS.json` and `FP_DR_TS.json` are first-price across the different off-policy estimators. Each writes to the subdirectory named in its `output_dir`. `CONFIG.md` documents the schema.

`SP_Oracle.json` is the configuration I have verified end to end under this port. The first-price learned-bidder configs exercise the PyTorch policies and run considerably longer; I have not put them through it yet.

Two exploratory notebooks also run — `jupyter notebook`, then open either *Getting Started* notebook under `src/`.

## Built on

Extends [AuctionGym](https://github.com/amzn/auction-gym) (© Amazon.com, Inc., Apache-2.0 — see `LICENSE` and `NOTICE`), from Jeunen, Murphy & Allison, *Off-Policy Learning-to-Bid with AuctionGym*, KDD '23.

```bibtex
@inproceedings{jeunen2023offpolicy,
    author    = {Jeunen, Olivier and Murphy, Sean and Allison, Ben},
    title     = {Off-Policy Learning-to-Bid with AuctionGym},
    booktitle = {Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
    year      = {2023},
    pages     = {4219--4228},
    doi       = {10.1145/3580305.3599877}
}
```
