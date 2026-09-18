import os
import json
import torch
import polars as pl
import pandas as pd
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything


def reset_peak_memory_stats():
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            with torch.cuda.device(i):
                torch.cuda.reset_peak_memory_stats()


def log_peak_memory_stats(experiment):
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            with torch.cuda.device(i):
                max_alloc = torch.cuda.max_memory_allocated() / (1024 ** 3)      # GB
                max_reserved = torch.cuda.max_memory_reserved() / (1024 ** 3)    # GB
                print(f"[GPU {i}] peak allocated: {max_alloc:.4f} GB | peak reserved: {max_reserved:.4f} GB")
                experiment.log_scalar(f'gpu{i}_peak_allocated_gb', max_alloc)
                experiment.log_scalar(f'gpu{i}_peak_reserved_gb', max_reserved)


def fix_seeds(random_state):
    """Set up random seeds."""

    seed_everything(random_state, workers=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_parameters(module):
    """Return (total_params, trainable_params)."""
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable

def crop_embeddings_to_num_items(embeddings, config):

    dataset_config = OmegaConf.load(os.path.join(config.base_path, "config.yaml"))

    num_items = dataset_config["num_items"]
    shift = dataset_config["shift"]

    # embeddings are for real items, preprocessed item_ids are shifted by shift
    embeddings = embeddings[:num_items + 1 - shift, :]
    print(f"Cropped embeddings to max id: {num_items}")

    return embeddings


def dataframe_reader(filepath: str, engine: str = "polars"):
    if engine == "polars":
        return polars_reader(filepath)
    elif engine == "pandas":
        return pandas_reader(filepath)
    else:
        raise ValueError(f"invalid reader engine: {engine}")


def pandas_reader(data_path):
    if str(data_path).split(".")[-1] == 'parquet':
        return pd.read_parquet(data_path)
    return pd.read_csv(data_path)


def polars_reader(filepath: str) -> pl.DataFrame:
    _, ext = os.path.splitext(filepath)

    if ext == ".parquet":
        return pl.read_parquet(filepath)
    elif ext == ".csv":
        return pl.read_csv(filepath)
    else:
        raise ValueError(f"unsupported file extension: {ext!r}")


def dataframe_writer(data, filepath: str, engine="polars"):
    if engine == "polars":
        return polars_writer(data, filepath)
    elif engine == "pandas":
        return pandas_writer(data, filepath)
    else:
        raise ValueError(f"invalid writer engine: {engine}")


def pandas_writer(df, data_path):
    if str(data_path).split(".")[-1] == 'parquet':
        df.to_parquet(data_path, index=False)
    else:
        df.to_csv(data_path, index=False)


def polars_writer(data: pl.DataFrame, filepath: str) -> None:
    _, ext = os.path.splitext(filepath)

    if ext == ".parquet":
        data.write_parquet(filepath)
    elif ext == ".csv":
        data.write_csv(filepath)
    else:
        raise ValueError(f"unsupported file extension: {ext!r}")
