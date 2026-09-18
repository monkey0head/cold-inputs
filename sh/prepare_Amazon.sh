#!/usr/bin/env bash
# Prepare the Amazon datasets for the SID scaling study: preprocess, split, embeddings.
#
# Run from the project root:
#   bash sh/prepare_Amazon.sh
#
# Writes preprocessed/<Dataset>_scaling/ and
# split/<Dataset>_scaling/gts_q09_val_by_time/, leaving the shared
# preprocessed/<Dataset>/ and split/<Dataset>/ data of the collision
# experiments untouched.
#
# The dataset configs name the preprocessed data (`name: <Dataset>_scaling`), so
# only the preprocess step needs `dataset.name=<Dataset>` to read the raw directory.
# 5-core filtering with consecutive-duplicate removal (runs/config/preprocess.yaml)
# and the quantile-0.9 global temporal split (runs/config/split.yaml) are the config
# defaults.

set -euo pipefail

export PYTHONPATH="./"
export CLEARML_SUPPRESS_UPDATE_MESSAGE=1
export HF_HUB_DISABLE_XET=1   # the xet protocol hangs on this cluster
export SEQ_REC_DATA_PATH=/home/jovyan/shares/SR003.nfs2/recsys_data/semantic_ids

echo "=== Beauty2014 ==="
# python runs/preprocess.py dataset=Beauty2014 dataset.name=Beauty2014 suffix=scaling
# python runs/split.py dataset=Beauty2014
python runs/build_semantic_embeddings.py dataset=Beauty2014 name=all-MiniLM-L6-v2 \
    split_name=gts_q09_val_by_time cuda_visible_devices=2

echo "=== Sports2014 ==="
# python runs/preprocess.py dataset=Sports2014 dataset.name=Sports2014 suffix=scaling
# python runs/split.py dataset=Sports2014
python runs/build_semantic_embeddings.py dataset=Sports2014 name=all-MiniLM-L6-v2 \
    split_name=gts_q09_val_by_time cuda_visible_devices=2

echo "=== Toys2014 ==="
# python runs/preprocess.py dataset=Toys2014 dataset.name=Toys2014 suffix=scaling
# python runs/split.py dataset=Toys2014
python runs/build_semantic_embeddings.py dataset=Toys2014 name=all-MiniLM-L6-v2 \
    split_name=gts_q09_val_by_time cuda_visible_devices=2

echo "=== Beauty2023 ==="
# python runs/preprocess.py dataset=Beauty2023 dataset.name=Beauty2023 suffix=scaling
# python runs/split.py dataset=Beauty2023
python runs/build_semantic_embeddings.py dataset=Beauty2023 name=all-MiniLM-L6-v2 \
    split_name=gts_q09_val_by_time cuda_visible_devices=2
