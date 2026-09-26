# Cold-inputs research baseline

This project supports GPTRec with Sequential semantic IDs and full-softmax SASRec
with ordinary item IDs through one training entry point. It includes interaction
preprocessing, temporal and leave-one-out splitting, metadata embedding preparation,
FAISS residual quantization, training, and evaluation. Hydra configs live in
`runs/config/`; entry points live in `runs/`.
DiskTracker and ClearML are available, with DiskTracker selected by default.

The research plan is in [codex_plan.md](../codex_plan.md); the implemented split
is described in [codex_splitting_pipeline.md](codex_splitting_pipeline.md).
The split and embedding launchers cover Beauty2023 and Yambda500m.

## Environment and entry points

Use the existing Python 3.11 environment from this project root:

```bash
export PYTHON_BIN="/home/jovyan/shares/SR003.nfs2/volodkevich/25_10_semantic/sem_venv_311_av/bin/python"
export PYTHONPATH=.
export SEQ_REC_DATA_PATH=/path/to/your/data
```

`requirements.txt` lists the inherited dependencies. The baseline manifest records
the versions used for validation; no packages are installed by the tests.

| Entry point | Purpose |
|---|---|
| `runs/get_data/setup_amazon2014.py` | Download Amazon 2014 interactions and metadata. |
| `runs/preprocess.py` | Read raw events and save cleaned interactions and metadata. |
| `runs/split.py` | Construct splits, encode IDs, and save mappings/statistics. |
| `runs/build_semantic_embeddings.py` | Encode text or prepare existing item vectors; requires `name` and `split_name`. |
| `runs/quantize_and_train.py` | Train and evaluate either recipe; quantize vectors and append Sequential collision codes for SID. |
| `runs/quantize.py`, `runs/train.py` | Helpers used by the combined entry point. |

Inspect a resolved config without training:

```bash
"$PYTHON_BIN" -B runs/quantize_and_train.py --cfg job --resolve dataset=Beauty2014
"$PYTHON_BIN" -B runs/quantize_and_train.py --cfg job --resolve dataset=Beauty2014 recipe=SASRec
```

Select the recipe using the same entry point, after preparing the chosen split:

```bash
"$PYTHON_BIN" -B runs/quantize_and_train.py dataset=Beauty2014 recipe=GPTRec_SID
"$PYTHON_BIN" -B runs/quantize_and_train.py dataset=Beauty2014 recipe=SASRec
```

Both commands use the dataset and trainer settings in the shared configuration,
including GPU training by default. Supply the prepared `split_name` when it differs
from the default. GPTRec requires embeddings for that split. The SASRec recipe sets
`quantization.enabled=false`, so it needs no embeddings and runs no SID diagnostics.
Incompatible recipe/quantization settings fail before creating a tracker run.

SASRec defaults to 128 hidden units, three blocks, two heads, dropout 0.3, and the
shared AdamW/cosine scheduler configuration. `dataset.seq_length_items=L` limits
training to L events and gives SASRec at most L−1 input events with shifted labels.
Prediction also uses at most L−1 input events. The model allocates one embedding
row per item plus padding row 0. Its full-softmax loss and ranking preserve the
source behavior: when `filter_seen=false`, padding class 0 remains eligible for
prediction. SASRec supports `dataset.filter_seen`; all retained dataset defaults
set it to false. Its tracker names include SASRec settings without quantizer fields.

`sh/prepare_data.sh` contains the current cleaning commands. Test preparation
uses a local data root. No historical experiment presets, generated runs, or
source Git history are included.

## Baseline limitations

- Dataset names and numerical settings describe the source SID pipeline; they
  are not finalized settings for the cold-input study.
- Temporal splitting fixes training suffixes before defining seen items and
  saves three input variants with shared targets and history statistics.
- Model-specific representations of unseen inputs remain open. Constructing
  a retained input does not make every model capable of representing its items.
- Validation users are sampled when dataloaders are constructed. A saved shared
  validation sample is not implemented.
- SID evaluation does not apply `filter_seen`; SASRec applies it when enabled.
- Model-training exposure logs retain the source definitions. The split audit
  separately counts selected training events and distinct training users per item.
- Sequential assignment preserves insertion order. Stable extension of a fitted
  SID mapping to later cold items is not implemented.

## Data preprocessing

Shared settings live in `runs/config/preprocess.yaml`. `sh/prepare_data.sh` has
separate blocks for Beauty 2014, Sports 2014, Toys 2014, Beauty 2023, Yambda 50M,
and Yambda 500M. Set its environment variables, then copy individual commands
or run the script from the project root. Inspect Beauty 2014 without running it:

```bash
"$PYTHON_BIN" -B runs/preprocess.py dataset=Beauty2014 dataset.name=Beauty2014 --cfg job --resolve
```

Main preprocessing enables iterative user/item 5-core filtering
(`n_core_filtering: true`, `user_min_count: 5`, `item_min_count: 5`), with no user
sampling or sequence truncation. Amazon has no rating threshold; Yambda uses `played_ratio_pct >= 50`,
with the final 30-day window applied first for 500M. Exact source-record duplicates
keep their last occurrence. Stable ordering preserves timestamp ties, and
consecutive same-item runs keep their first event. Then 5-core filtering removes
users and items with fewer than five retained events, repeating consecutive-repeat
removal and count filtering until stable. Counts cover the whole cleaned dataset
window before splitting and count events, not distinct user-item pairs. Five
global events do not imply five training events or training-warm membership.
Existing filter helpers perform these operations; `dataset_stats` supplies the audit.
User minimum count 5 with no item-frequency filter is an alternative, not the
current preprocessing setting.

Raw interactions and optional metadata are read from `SEQ_REC_DATA_PATH/raw/`.
Output uses `preprocessed/<dataset>_cold_inputs/`: interactions, optional metadata,
`config.yaml`, and `statistics.json`. `_source_row` identifies the original row.
Statistics contain per-step removals, timestamp ties, and cleaned history-length
and item-frequency distributions. Missing IDs/timestamps and existing output
directories raise errors. Beauty 2023 requires the default Polars engine for
deduplication of its nested `images` field. `dataset.timestamp_unit` is `ms` for
Beauty 2023 and defaults to `s` elsewhere. Day-window filters and day-range
statistics respect this setting; saved timestamps retain their original values.
Splitting and training remain separate.

## Cold-input splitting

The [preprocessing experiment](experiments/2026-09-20_preprocessing/README.md)
contains the generated CSV and Markdown summary of all six datasets. Experiments
follow the dated layout and are excluded from Git; see the
[experiment index](experiments/README.md).

The study focuses on Beauty 2023 and Yambda 500M. Primary input comparisons require
targets with all three variants available; exclusions are reported separately.
The [consolidated audit](experiments/2026-09-20_preprocessing/findings/codex_split_variants_audit.md)
includes cropped-training frequency histograms, matched-input bar plots, and
catalogue activity/stability plots over the full cleaned timelines.

`GlobalTimeSplitter` uses temporal validation only, with required
`time_threshold`, `val_time_threshold`, and `train_max_events` parameters.
`remove_unseen_targets=true` selects the last training-seen holdout event;
`remove_unseen_users=false` retains users outside the cropped training-user set
subject to the inherited temporal eligibility. `LeaveOneOutSplitter` remains
available through `splitting_strategy=loo`.

`gts.yaml` is the default splitting config. Both time thresholds default to `0.9`;
`splitting_strategy.train_max_events` defaults to `${dataset.seq_length_items}`.
The dataset configs set this length to 20 for Beauty and 100 for Yambda;
Sports2014 and Toys2014 retain their baseline length of 16.

`sh/prepare_data.sh` cleans all six datasets. `sh/split.sh` reads only the two
selected `_cold_inputs` datasets; `sh/generate_embeddings.sh` uses the same names
and split directories in a separate step. Each launcher exports an absolute
`PYTHON_BIN` path. Inspect both split configs with `bash sh/split.sh --cfg job --resolve`.

| Dataset | Training / input limit | Test / validation threshold | Split name |
|---|---|---|---|
| Beauty2023 | 20 / 19 | 0.9 / 0.9, nested timestamp quantiles | `gts_q09_val_by_time` |
| Yambda500m, final 30 days | 100 / 99 | 25913600 / 25827200, seconds | `gts_1d_val_by_time` |

GTS saves retain, remove-then-crop, and crop-then-remove inputs, alongside
separate validation/test targets. Targets include original-history frequency
histograms and each variant's statistics and availability. Empty variants are
excluded from model evaluation without deleting shared targets. Select inputs
with `input_variant=retain`, `remove_then_crop` (default), or `crop_then_remove`.
LOO also saves separate validation inputs and targets. Use
`splitting_strategy=loo split_name=loo` for a separate LOO destination.

The splitter has no rare threshold. Exact item counts and per-target histograms
support later rarity analysis, including the audit's thresholds 1 through 5.
Dataset/model length settings must accommodate the saved sequences; the loader
rejects additional truncation or `train_last_days` changes to a temporal split.
See the [pipeline description](codex_splitting_pipeline.md) and
[figure prompt](codex_splitting_figure_prompt.md).

Terminal output reports event, user, and item totals per subset. Detailed counts
are saved as `seq_len_distribution.csv`, `item_occurrence_distribution.csv`, and
`frequency_histogram.csv` (GTS training-item frequencies), each with a `subset`
column. `statistics.json` holds scalar summaries and resolved protocol settings.
Per-target history histograms remain in the target Parquet files.

The loader aligns each selected variant with its available targets. Enforcing the
all-three-available population across model runs and saving one shared validation
sample (cap 20,000, dedicated seed) remain evaluation work. Unseen-input
representations and output-catalog scope also remain open. The audit and
`runs/split.py` refuse existing outputs; combined-validation artifacts require a
new split name for the separate-target loader.

## Validation

Run the preprocessing regression checks with synthetic data on CPU:

```bash
PYTHONPATH=. "$PYTHON_BIN" -B codex_tests/codex_check_preprocessing.py
```

These cover Pandas/Polars agreement, source-record identity, ordering, repeats,
singletons, inclusive window boundaries, relevance filtering, empty results,
missing fields, saved audits, and overwrite protection. Temporary fixtures stay
inside `codex_validation/` and are removed after the checks.

Check the temporal splitter extension and LOO on CPU:

```bash
PYTHONPATH=. "$PYTHON_BIN" -B codex_tests/codex_check_splits.py
CUDA_VISIBLE_DEVICES='' PYTHONPATH=. "$PYTHON_BIN" -B codex_tests/codex_check_split_pipeline.py
```

These checks cover fixed training membership, temporal eligibility, tied target
identity, three variants, empty histories, deferred rarity, Parquet roundtrips,
CSV count consistency, overwrite protection, explicit-target dataset parity, per-user metric
aggregation, GTS/LOO baseline parity, and split configs.

Run the approved CPU checks from this directory:

```bash
bash codex_tests/codex_validate.sh
```

The equivalence check reads `../../sid_scaling` and compares the retained model
and split behavior using Sequential explicitly. Cleaning follows the separate
preprocessing checks above. It includes manual split and ranking-metric checks.
SASRec parity checks read `../../semantic-ids` and compare dataset windows, model
outputs, loss, gradients, ranking, and validation metrics with the source.
Both recipes have a synthetic two-epoch CPU test with saved validation each epoch
and checkpoint reload. The SID test uses existing vectors and the constrained
decoder to exercise valid catalog predictions; this is not a research protocol
decision. The SASRec test runs with no embedding directory. Text embedding
downloads and GPU/full-data runs are excluded.

Tests hide CUDA and keep caches, fixtures, and outputs in ignored
`codex_validation/`. Each invocation creates new output directories. The wrapper
stops on a failed command. See [codex_sasrec_validation.md](codex_sasrec_validation.md)
for the current integration checks and file inventory.
[codex_baseline_validation.md](codex_baseline_validation.md) and
[codex_baseline_manifest.json](codex_baseline_manifest.json) describe the initial
SID snapshot; their hashes do not describe the complete SASRec-enabled tree.
