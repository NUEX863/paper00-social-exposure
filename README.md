# Social Exposure

Reproducible exposure-response measurements and network adoption simulations with local LLMs.

## Contents

- `src/`: collection, response parsing, statistical analysis and contagion simulation.
- `configs/`: experimental settings, model identities, request seeds and network specifications.
- `prompts/`: frozen prompts, synthetic personas and item definitions.
- `runs/`: selected observation tables, interpreted model records and computed results.
- `figures/`: four result figures.
- `scripts/`: integrity validation and analysis entry points.

## Setup

Python 3.11 is required. Install the direct dependencies:

```sh
python -m pip install -r requirements.txt
python scripts/validate_release.py
```

The validation command makes no model calls. `requirements-lock.txt` records the broader original environment. File hashes are preserved across platforms by `.gitattributes`.

## Existing results

| Data | Location |
|---|---|
| ERA: 42,630 extracted records and statistics | `runs/era_analysis/20261019T015255Z_d5b87f8c44c6/` |
| Amount comparison: aligned token scores | `runs/era_readout_alignment/20261019T080301Z_ec8ffdefaa6c/` |
| Network comparison: 432 support records, 477 decisions, eight trajectories | `runs/sprint_v4/20261019_authorized/` |
| Classical simulation summary | `runs/a20_batches/20261012T070248Z_5cc5734979444df7a023a20fcc96022f/summary.json` |

The complete original raw ERA and amount-response archives are not included. Extracted observations and interpreted macro records support numerical replay, but do not replace a raw-response audit. New collections save raw responses and source snapshots.

## Offline reproduction

Use new output directories:

```sh
python scripts/replay_core_statistics.py --output reproductions/core_stats
python src/analyze_sprint_v4.py --run-dir runs/sprint_v4/20261019_authorized --analysis-dir reproductions/macro_analysis
python src/run_a20_batch.py --output-root reproductions/classical
```

These commands run local statistical calculations and simulations, with no LLM requests. Core analysis uses 2,000 bootstrap resamples. Macro plots are written to `outputs/figures/`; released figures in `figures/` are unchanged.

## New LLM collection

The recorded runtime is Ollama 0.40.6. Model and template digests are in `configs/era_submission_runtime_v3.json`; collectors reject mismatches. Start Ollama separately. These commands consume local GPU time:

```sh
ollama pull qwen3:8b-q4_K_M
ollama pull mistral:7b-instruct-v0.3-q4_K_M
python src/run_era_formal.py --validate-design
python src/run_era_formal.py --run-dir reproductions/core_new
python src/analyze_era_formal.py --run-dir reproductions/core_new --analysis-dir reproductions/core_new_analysis
python src/run_era_heldout.py --run-dir reproductions/amount_new
```

The amount collector runs 2,400 requests and analyzes them against the released historical core controls. Current model tags can differ from recorded digests; record any runtime amendment explicitly. Fixed seeds do not ensure bitwise-identical inference across hardware or backends. Use fresh directories rather than resuming original private-run source snapshots.

For the bounded network collection:

```sh
python scripts/authorize_macro.py --approve --output reproductions/macro_authorization.json
python src/run_sprint_v4.py --run-dir reproductions/macro_new --authorization reproductions/macro_authorization.json
python src/analyze_sprint_v4.py --run-dir reproductions/macro_new
```

The network collector has a cap of 1,800 calls and 2,700 seconds. The saved collection used 909 calls. Its network size is 64, versus 128 for the classical comparison.

## Readout and scope

Sampling at temperature 0.7 and temperature-one conditional token-pair scores are distinct readouts. Unavailable scores are not zero. Synthetic persona intervals do not describe human-population uncertainty. Macro simulation ranges condition on the measured responses and two paired graph configurations. The R2 network condition is exploratory. The explicit reinforcement policy uses investigator-fixed parameters and is not calibrated against matched human data.
