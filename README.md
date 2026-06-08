# HyAR-CHO v2 — Modular Package

HyAR reinforcement learning applied to CHO-cell IgG media optimization with a
constraint-based genome-scale model (iCHO2291), plus the full priority-1–10
FBA analysis suite and 19 ablation flags recommended by the research audit.

## Layout

```
cho_hyar_v2/
├── environment.yml
├── README.md
└── cho_hyar/
    ├── config.py              ← CLI args + 19 ablation flags
    ├── environment.py         ← CHOEnvironment (reward-norm, PBRS, penalties,
    │                            curriculum, infeasibility, action-param)
    ├── representation.py      ← HyAR VAE: embedding table + encoder + decoder
    ├── policy.py              ← FactoredBernoulliActor + double critic
    ├── buffer.py              ← ReplayBuffer + LSC + RSC
    ├── trainer.py             ← HyAR-TD3 training loop
    ├── run.py                 ← top-level entry point
    └── analysis/
        ├── helpers.py
        ├── basic_plots.py     ← reward / state / key-flux / discrete heatmap
        ├── discrete.py        ← per-obj + cross-obj add/remove analyses
        ├── fva.py             ← Priority 1: Flux Variability Analysis
        ├── baselines.py       ← Priority 2: default / FBA / random / HyAR
        ├── flux_sampling.py   ← Priority 3: ACHR / optgp sampling
        ├── pareto.py          ← Priority 4: Pareto frontier + hypervolume
        ├── shadow_prices.py   ← Priority 5: shadow prices + reduced costs
        ├── subsystem_enrichment.py  ← Priority 6: hypergeometric + BH-FDR
        ├── sensitivity.py     ← Priority 7: Morris elementary effects
        ├── essentiality.py    ← Priority 8: single-reaction knockout
        ├── convergence.py     ← Priority 9: return / entropy / toggle rate
        ├── clustering.py      ← Priority 10: hierarchical + Jaccard
        └── pca.py             ← PCA / biplot / loadings (from v1)
```

## Setup

```bash
conda env create -f environment.yml
conda activate cho_hyar
```

For GPU, replace the pip torch line with the CUDA wheel for your setup, e.g.:
```yaml
- torch --extra-index-url https://download.pytorch.org/whl/cu118
```

## Run

From the directory that contains the `cho_hyar/` package:

```bash
# full default run — factored-bernoulli action encoding, all analyses
python -m cho_hyar.run --output-dir results_default

# analyses only (reuse saved histories)
python -m cho_hyar.run --skip-training --output-dir results_default

# with expensive flux sampling enabled
python -m cho_hyar.run --run-flux-sampling --n-flux-samples 5000
```

## Ablation flags (19 total)

| # | Flag | Options (default bold) | Purpose |
|---|---|---|---|
| 1 | `--action-encoding`     | hyar-k-entry / **factored-bernoulli** | Core action space design |
| 2 | `--reward-norm`         | none / **fba-max** / welford | Prevent DM_igg dominance |
| 3 | `--action-param`        | absolute / **fractional** / log-signed | 4-OoM uptake range |
| 4 | `--reward-shape`        | **linear** / chebyshev | Pareto coverage |
| 5 | `--pbrs`                | none / **growth** / pfba | Dense per-step signal |
| 6 | `--smoothness`          | none / **caps** | Anti-thrashing |
| 7 | `--infeas-handling`     | terminate / **penalty** / relax-fba | FBA infeasibility |
| 8 | `--embedding-init`      | **random** / hand-features | Biological priors |
| 9 | `--curriculum`          | **none** / rich-to-lean | Progressive restriction |
| 10 | `--switch-penalty`     | 0 / 0.001 / **0.01** / 0.1 | λ·|a_t − a_{t−1}| |
| 11 | `--policy-dist`        | **tanh-gaussian** / beta | Continuous head dist |
| 12 | `--share-embed`        | shared / per-reaction / **shared+bias** | Parameter sharing |
| 13 | `--algo`               | **td3** / sac / ppo | Latent-policy algorithm |
| 14 | `--lsc-percent`        | 80 / **96** / 100 | Paper-verified optimum |
| 15 | `--dyn-weight-beta`    | 0 / 1 / 5 / **10** / 20 | Dynamics loss weight |
| 16 | `--latent-dim`         | 3 / **6** / 12 | Latent size |
| 17 | `--rsc`                | **on** / off | Representation Shift Correction |
| 18 | `--encoder-fusion`     | **elementwise** / concat | VAE conditioning |
| 19 | `--dynamics-head`      | **cascaded** / parallel | Prediction head structure |

### Example ablation runs

```bash
# Pure paper HyAR (no audit improvements)
python -m cho_hyar.run \
    --action-encoding hyar-k-entry \
    --reward-norm none \
    --action-param absolute \
    --pbrs none \
    --smoothness none \
    --switch-penalty 0 \
    --output-dir results_paper_original

# Audit-recommended configuration (default; shown for clarity)
python -m cho_hyar.run --output-dir results_audit_default

# Ablation: disable dynamics prediction
python -m cho_hyar.run \
    --dynamics-prediction off \
    --output-dir results_no_dyn

# Ablation: compare latent dimensions
for d in 3 6 12; do
    python -m cho_hyar.run --latent-dim $d --output-dir results_dim_$d
done
```

## Priority-1–10 analyses

Defaults **on** (cheap):
- FVA, baselines, Pareto, shadow prices, subsystem enrichment, Morris
  sensitivity, essentiality, convergence diagnostics, hierarchical
  clustering + Jaccard, PCA.

Default **off** (expensive):
- Flux sampling (~5 min/objective). Enable with `--run-flux-sampling`.

Disable any individual analysis with `--no-run-<name>` (argparse auto-generates
these when using `action="store_true"` with `default=True`).

## Runtime on iCHO2291

Approximate timing (40 episodes × 30 steps × 4 objectives):
- Training: **45 min CPU / 20 min GPU**
- FVA (exchanges only): **2 min**
- Baselines: **3 min** (dominated by 50-sample random search × 4)
- Pareto: **30 s**
- Shadow prices: **10 s**
- Subsystem enrichment: **10 s**
- Morris: **5 min** (10 trajectories × ~30 reactions × 4 objectives)
- Essentiality: **2 min**
- Convergence + clustering + PCA: **30 s**
- Flux sampling (if enabled): **20 min**

Total default: **~60 min** CPU.

## Headless safety

Verified no `plt.show()` calls; `matplotlib.use("Agg")` set in every plot
module; every `plt.savefig` paired with `plt.close()`. The script never
blocks on user input.

## Architecture notes

Per the research audit, the **factored-Bernoulli actor is the default** because
the CHO media problem is a binary-vector discrete action per step (one
on/off decision per exchange reaction), not the single K-way categorical of
HyAR's Platform/Goal benchmarks. The HyAR VAE still provides the continuous
parameter representation conditioned on each reaction's on/off embedding.
Switch to `--action-encoding hyar-k-entry` to reproduce the original paper
formulation as an ablation comparison.
