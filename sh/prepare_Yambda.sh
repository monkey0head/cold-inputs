#!/usr/bin/env bash
# Prepare the Yambda listens datasets for the SID scaling study: preprocess, split,
# embeddings.
#
# Run from the project root:
#   bash sh/prepare_Yambda.sh
#
# Writes preprocessed/<Raw>_u5_i20_scaling/ and
# split/<Raw>_u5_i20_scaling/gts_1d_val_by_time/, leaving the shared
# preprocessed/<Raw>_u5_i20/ and split/<Raw>_u5_i20/ data untouched. The embeddings
# step stacks the precomputed `normalized_embed` column, so it needs no GPU; the
# resulting embs dir is `normalized_embeds`, which training must be told about with
# `quantization.embs_name=normalized_embeds`.
#
# min_relevance=50 drops skips (played_ratio_pct < 50) before N-core filtering and
# matches the other Yambda listens datasets; without it the result is not
# comparable to them. Together with item_min_count=20 it is the only part of the
# recipe that differs from the runs/config/preprocess.yaml defaults.
#
# The Yambda dataset configs hold the preprocessed name, suffix included, so the
# split step and everything after it need no name override. The raw name is given
# here for the preprocess read. The 1-day cutoffs live in
# runs/config/splitting_strategy/gts_1d.yaml.

set -euo pipefail

export PYTHONPATH="./"
export CLEARML_SUPPRESS_UPDATE_MESSAGE=1
export SEQ_REC_DATA_PATH=/home/jovyan/shares/SR003.nfs2/recsys_data/semantic_ids

echo "=== Yambda50m ==="
# python runs/preprocess.py dataset=Yambda50m dataset.name=Yambda50m_listens suffix=u5_i20_scaling \
#     preprocessing_params.min_relevance=50 preprocessing_params.item_min_count=20
# python runs/split.py dataset=Yambda50m splitting_strategy=gts_1d split_name=gts_1d_val_by_time
python runs/build_semantic_embeddings.py dataset=Yambda50m name=normalized_embeds \
    split_name=gts_1d_val_by_time  cuda_visible_devices=2

echo "=== Yambda500m ==="
# python runs/preprocess.py dataset=Yambda500m dataset.name=Yambda500m_listens suffix=u5_i20_scaling \
#     preprocessing_params.min_relevance=50 preprocessing_params.item_min_count=20
# python runs/split.py dataset=Yambda500m splitting_strategy=gts_1d split_name=gts_1d_val_by_time
python runs/build_semantic_embeddings.py dataset=Yambda500m name=normalized_embeds \
    split_name=gts_1d_val_by_time  cuda_visible_devices=2
