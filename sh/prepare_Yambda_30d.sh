#!/usr/bin/env bash
# Prepare the last-30-days Yambda500m listens dataset for the SID scaling study:
# preprocess, split, embeddings.
#
# Run from the project root:
#   bash sh/prepare_Yambda_30d.sh
#
# Writes preprocessed/Yambda500m_listens_30d_u5_i20_scaling/ and
# split/Yambda500m_listens_30d_u5_i20_scaling/gts_1d_val_by_time/, leaving the
# uncropped Yambda500m_listens_u5_i20_scaling data that sh/prepare_Yambda.sh built
# untouched.
#
# last_n_days=30 crops before every other filter, so played_ratio_pct >= 50,
# item_min_count=20 and user_min_count=5 all count inside the 30-day window. The crop
# keeps the tail, so max_timestamp stays 26000000 and the absolute gts_1d thresholds
# need no adjustment: test is still the last day and validation the day before.
#
# The embeddings step stacks the precomputed `normalized_embed` column, so it needs no
# GPU; the resulting embs dir is `normalized_embeds`, which training must be told about
# with `quantization.embs_name=normalized_embeds`.

set -euo pipefail

export PYTHONPATH="./"
export CLEARML_SUPPRESS_UPDATE_MESSAGE=1
export SEQ_REC_DATA_PATH=/home/jovyan/shares/SR003.nfs2/recsys_data/semantic_ids

echo "=== preprocess ==="
python runs/preprocess.py dataset=Yambda500m dataset.name=Yambda500m_listens \
    suffix=30d_u5_i20_scaling \
    preprocessing_params.last_n_days=30 \
    preprocessing_params.min_relevance=50 \
    preprocessing_params.item_min_count=20

echo "=== split ==="
python runs/split.py dataset=Yambda500m_30d splitting_strategy=gts_1d \
    split_name=gts_1d_val_by_time

echo "=== embeddings ==="
python runs/build_semantic_embeddings.py dataset=Yambda500m_30d name=normalized_embeds \
    split_name=gts_1d_val_by_time cuda_visible_devices=0

echo "=== done ==="
