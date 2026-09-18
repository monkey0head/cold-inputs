import os
import pickle

import hydra
import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

from src.run_utils import pandas_reader

# Some sentence-transformers models expect task-specific prefixes (see model cards).
MODEL_PREFIXES = {
    "nomic-ai/nomic-embed-text-v1.5": "clustering: ",
    "intfloat/multilingual-e5-large": "passage: ",
    "intfloat/multilingual-e5-base": "passage: ",
}


def generate_embeds_from_text(data, config):

    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.cuda_visible_devices)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    data = add_item_text(data, config.dataset.meta_columns_to_embed)

    print(f"Loading model: {config.model_name}")
    model = load_model(
        model_name=config.model_name,
        device=device
    )
    print("Model loaded")

    print("Generating embeddings...")
    prefix = get_prefix_for_model(config)
    if prefix:
        print(f"Using input prefix for this model: {prefix!r}")
    embeddings = encode_text(
        model,
        data["_full_description"],
        batch_size=config.batch_size,
        show_progress=True,
        prefix=prefix,
    )

    print(f"Generated embeddings shape: {embeddings.shape}")
    return embeddings


def add_item_text(data, columns):
    data["_full_description"] = ""
    for column in columns:
        data["_full_description"] += data[column].apply(lambda x: f"{column}: {x}; ").astype(str)
    return data


def load_model(model_name="sentence-transformers/all-MiniLM-L6-v2", device=None):
    # imported here, not at module level: sentence_transformers enumerates the CUDA
    # devices on import, which freezes the CUDA_VISIBLE_DEVICES mapping for the whole
    # process. Importing it before `cuda_visible_devices` is applied would send the
    # work to GPU 0 whatever the config says.
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device=device, trust_remote_code=True)
    return model


def get_prefix_for_model(config):

    prefix = OmegaConf.select(config, "text_prefix", default=None)
    if prefix is not None:
        return str(prefix)

    return MODEL_PREFIXES.get(config.model_name)


def encode_text(model, texts, batch_size=32, show_progress=True, prefix=""):
    if isinstance(texts, pd.Series):
        texts = texts.tolist()
    texts = [str(text) if pd.notna(text) else "" for text in texts]
    if prefix:
        texts = [prefix + text for text in texts]

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
        normalize_embeddings=True  # Note: embeddings are L2-normalized before quantization.
    )
    return embeddings


@hydra.main(config_path="config", config_name="build_semantic_embeddings")
def main(config):
    # Load data
    dataset_name = config.dataset.name

    emb_path = os.path.join(
        os.environ["SEQ_REC_DATA_PATH"],
        "preprocessed",
        dataset_name,
        f"{dataset_name}_meta.parquet",
    )

    mapping_path = os.path.join(
        os.environ["SEQ_REC_DATA_PATH"],
        "split",
        dataset_name,
        config.split_name,
        "item_mapping.pkl",
    )
    with open(mapping_path, mode="rb") as file:
        item_mapping = pickle.load(file)

    print(f"Reading metadata from {emb_path}")
    data = pandas_reader(emb_path)
    data["item_id"] = data["item_id"].map(item_mapping)

    # One row per item_id so embedding row index aligns with item_id (row i = item_id i+1 when consecutive)
    data = data.drop_duplicates(subset=["item_id"], keep="first")
    data.sort_values(by="item_id", inplace=True)
    print(f"Data shape: {data.shape}")

    if OmegaConf.select(config, "dataset.meta_columns_to_embed", default=None) is not None:
        print("Generating embeddings from text")
        embeddings = generate_embeds_from_text(data, config)
    elif OmegaConf.select(config, "dataset.meta_embed_col", default=None) is not None:
        print("Converting existing embeddings")
        embeddings = np.vstack(data[config.dataset.meta_embed_col].values)

    else:
        raise ValueError("define meta_columns_to_embed or meta_embed_col in dataset config")

    path_to_save = os.path.join(
        os.environ["SEQ_REC_DATA_PATH"],
        "split",
        dataset_name,
        config.split_name,
        "embs",
        config.name,
    )
    os.makedirs(path_to_save, exist_ok=True)

    OmegaConf.save(config, os.path.join(path_to_save, "config.yaml"))

    np.save(os.path.join(path_to_save, "embs.npy"), embeddings)
    print(f"Saved embeddings to {path_to_save}")


if __name__ == "__main__":

    main()
