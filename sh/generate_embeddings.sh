#!/usr/bin/env bash
set -euo pipefail
trap 'status=$?; echo "FAILED (exit ${status})" >&2; exit "$status"' ERR
trap 'exit 130' INT
trap 'exit 143' TERM

export PYTHON_BIN="/home/jovyan/shares/SR003.nfs2/volodkevich/25_10_semantic/sem_venv_311_av/bin/python"
export PYTHONPATH="./"
export HF_HUB_DISABLE_XET=1
export SEQ_REC_DATA_PATH=/home/jovyan/shares/SR003.nfs2/recsys_data/semantic_ids

echo "START Beauty2023 embeddings"
"$PYTHON_BIN" -B runs/build_semantic_embeddings.py \
    dataset=Beauty2023 dataset.name=Beauty2023_cold_inputs name=all-MiniLM-L6-v2 \
    split_name=gts_q09_val_by_time cuda_visible_devices=2 "$@"
echo "DONE Beauty2023 embeddings (exit 0)"

echo "START Yambda500m embeddings"
"$PYTHON_BIN" -B runs/build_semantic_embeddings.py \
    dataset=Yambda500m dataset.name=Yambda500m_listens_cold_inputs name=normalized_embeds \
    split_name=gts_1d_val_by_time "$@"
echo "DONE Yambda500m embeddings (exit 0)"
