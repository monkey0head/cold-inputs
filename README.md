# Cold-inputs research baseline

This project supports GPTRec with Sequential semantic IDs and full-softmax SASRec
with ordinary item IDs through one training entry point. It includes interaction
preprocessing, temporal and leave-one-out splitting, metadata embedding preparation,
FAISS residual quantization, training, and evaluation. Hydra configs live in
`runs/config/`; entry points live in `runs/`.
DiskTracker and ClearML are available, with DiskTracker selected by default.

The research protocol and decisions are in [codex_plan.md](../codex_plan.md).
Beauty 2014 is the first dataset for the cleaning and audit stage. Sequence
lengths and Amazon temporal boundaries remain undecided; model and splitting
code still use the baseline protocol.

## Environment and entry points

Use the existing Python 3.11 environment from this project root:

```bash
source ../../sem_venv_311_av/bin/activate
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
python -B runs/quantize_and_train.py --cfg job --resolve dataset=Beauty2014
python -B runs/quantize_and_train.py --cfg job --resolve dataset=Beauty2014 recipe=SASRec
```

Select the recipe using the same entry point, after preparing the chosen split:

```bash
python -B runs/quantize_and_train.py dataset=Beauty2014 recipe=GPTRec_SID
python -B runs/quantize_and_train.py dataset=Beauty2014 recipe=SASRec
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
- Training suffixes are chosen during dataset access, after the split vocabulary
  has been defined. Fixed suffixes therefore do not yet define item warmth.
- Cold-only histories are removed by source split alignment. Cold-input
  reconstruction and the additional evaluation group are not implemented.
- Validation users are sampled when dataloaders are constructed. A saved shared
  validation sample is not implemented.
- SID evaluation does not apply `filter_seen`; SASRec applies it when enabled.
- Training-exposure diagnostics use the source definitions, which differ from
  the study's selected-training-event definition of unseen items.
- Sequential assignment preserves insertion order. Stable extension of a fitted
  SID mapping to later cold items is not implemented.

## Data preprocessing

Shared settings live in `runs/config/preprocess.yaml`. `sh/prepare_data.sh` has
separate blocks for Beauty 2014, Sports 2014, Toys 2014, Beauty 2023, Yambda 50M,
and Yambda 500M. Set its environment variables, then copy individual commands
or run the script from the project root. Inspect Beauty 2014 without running it:

```bash
python -B runs/preprocess.py dataset=Beauty2014 dataset.name=Beauty2014 --cfg job --resolve
```

Main preprocessing has no user/item frequency filter, user sampling, or sequence
truncation. Amazon has no rating threshold; Yambda uses `played_ratio_pct >= 50`,
with the final 30-day window applied first for 500M. Exact source-record duplicates
keep their last occurrence. Stable ordering preserves timestamp ties, and
consecutive same-item runs keep their first event. Existing filter helpers perform
these operations; `dataset_stats` supplies the audit.

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

## Validation

Run the preprocessing regression checks with synthetic data on CPU:

```bash
PYTHONPATH=. ../../sem_venv_311_av/bin/python -B codex_tests/codex_check_preprocessing.py
```

These cover Pandas/Polars agreement, source-record identity, ordering, repeats,
singletons, inclusive window boundaries, relevance filtering, empty results,
missing fields, saved audits, and overwrite protection. Temporary fixtures stay
inside `codex_validation/` and are removed after the checks.

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
