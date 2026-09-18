"""
Make user and item indexes.
"""

import numpy as np
import pandas as pd
import polars as pl


# === ENCODE ===
def encode(
    data: pd.DataFrame | pl.DataFrame,
    col: str,
    shift: int = 0,
    mapping: dict | None = None,
    expand_mapping: bool = False,
) -> tuple[pd.DataFrame | pl.DataFrame, dict]:

    if isinstance(data, pd.DataFrame):
        return _pandas_encode(data, col, shift, mapping, expand_mapping)
    elif isinstance(data, pl.DataFrame):
        return _polars_encode(data, col, shift, mapping, expand_mapping)
    else:
        raise TypeError


def _pandas_encode(
    data: pd.DataFrame,
    col: str,
    shift: int = 0,
    mapping: dict | None = None,
    expand_mapping: bool = False,
) -> tuple[pd.DataFrame, dict]:

    unique_values = np.unique(data[col].values)

    if mapping is None:
        mapping = {value: idx for idx, value in enumerate(unique_values, start=shift)}
    else:
        if expand_mapping:
            mapping = _expand_mapping(unique_values, mapping)

    data[col] = data[col].map(mapping)

    assert data[col].notna().all(), "some keys are missing but `expand_mapping` set to False"

    return data, mapping


def _polars_encode(
    data: pl.DataFrame,
    col: str,
    shift: int = 0,
    mapping: dict | None = None,
    expand_mapping: bool = False,
) -> tuple[pl.DataFrame, dict]:

    unique_values = np.unique(data[col].to_numpy())

    if mapping is None:
        mapping = {value: idx for idx, value in enumerate(unique_values, start=shift)}
    else:
        if expand_mapping:
            mapping = _expand_mapping(unique_values, mapping)

    return data.with_columns(pl.col(col).replace_strict(mapping)), mapping


def _expand_mapping(unique_values: np.ndarray, mapping: dict) -> dict:
    next_idx = max(mapping.values()) + 1

    for value in unique_values:
        if value not in mapping:
            mapping[value] = next_idx
            next_idx += 1

    return mapping
