#!/usr/bin/env bash
# Run from the project root with the project environment activated.
# Shared cleaning settings are in runs/config/preprocess.yaml.
# Split construction follows the sequence-length and temporal-cutoff decisions.

set -euo pipefail

export PYTHONPATH="./"
export SEQ_REC_DATA_PATH=/home/jovyan/shares/SR003.nfs2/recsys_data/semantic_ids

echo "=== Beauty2014 ==="
python runs/preprocess.py dataset=Beauty2014 dataset.name=Beauty2014

echo "=== Sports2014 ==="
python runs/preprocess.py dataset=Sports2014 dataset.name=Sports2014

echo "=== Toys2014 ==="
python runs/preprocess.py dataset=Toys2014 dataset.name=Toys2014

echo "=== Beauty2023 ==="
python runs/preprocess.py dataset=Beauty2023 dataset.name=Beauty2023

echo "=== Yambda50m ==="
python runs/preprocess.py dataset=Yambda50m dataset.name=Yambda50m_listens \
    preprocessing_params.min_relevance=50

echo "=== Yambda500m (last 30 days) ==="
python runs/preprocess.py dataset=Yambda500m dataset.name=Yambda500m_listens \
    preprocessing_params.min_relevance=50 \
    preprocessing_params.last_n_days=30
