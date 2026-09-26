#!/usr/bin/env bash
set -euo pipefail
trap 'status=$?; echo "FAILED (exit ${status})" >&2; exit "$status"' ERR
trap 'exit 130' INT
trap 'exit 143' TERM

export PYTHON_BIN="/home/jovyan/shares/SR003.nfs2/volodkevich/25_10_semantic/sem_venv_311_av/bin/python"
export PYTHONPATH="./"
export CUDA_VISIBLE_DEVICES=""
export SEQ_REC_DATA_PATH=/home/jovyan/shares/SR003.nfs2/recsys_data/semantic_ids

echo "START Beauty2023"
"$PYTHON_BIN" -B runs/split.py \
    dataset=Beauty2023 dataset.name=Beauty2023_cold_inputs \
    split_name=gts_q09_val_by_time "$@"
echo "DONE Beauty2023 (exit 0)"

echo "START Yambda500m (final 30 days)"
# Absolute cutoffs in seconds: test is the final day; validation is the preceding day.
"$PYTHON_BIN" -B runs/split.py \
    dataset=Yambda500m dataset.name=Yambda500m_listens_cold_inputs \
    splitting_strategy.time_threshold=25913600 splitting_strategy.val_time_threshold=25827200 \
    split_name=gts_1d_val_by_time "$@"
echo "DONE Yambda500m (exit 0)"
