"""Make split to train, validation and test."""

import json
import os
import pickle

import hydra
import pandas as pd
from omegaconf import DictConfig, OmegaConf, open_dict

from src.preprocess.indexes import encode
from src.preprocess.utils import dataset_stats
from src.run_utils import pandas_reader


@hydra.main(version_base=None, config_path="config", config_name="split")
def main(config: DictConfig) -> None:

    print(OmegaConf.to_yaml(config, resolve=True))

    dataset_name = config.dataset.name
    base_path = os.path.join(os.environ["SEQ_REC_DATA_PATH"], "preprocessed", dataset_name)
    output_dir = os.path.join(os.environ["SEQ_REC_DATA_PATH"], "split", dataset_name, config.split_name)
    if os.path.exists(output_dir):
        raise FileExistsError(f"Refusing to overwrite existing split: {output_dir}")

    splitting_strategy = hydra.utils.instantiate(config.splitting_strategy)

    data = pandas_reader(os.path.join(base_path, f"{dataset_name}.parquet"))
    metadata = pandas_reader(os.path.join(base_path, f"{dataset_name}_meta.parquet"))

    split = splitting_strategy.split(data)

    # encode item and user IDs
    for subset in split:
        if subset == "train":
            split[subset], item_mapping = encode(split[subset], col="item_id", shift=1)
            split[subset], user_mapping = encode(split[subset], col="user_id", shift=0)

            with open_dict(config):
                config.num_items = max(item_mapping.values())
                config.shift = 1
        else:
            split[subset], item_mapping = encode(
                split[subset],
                col="item_id",
                mapping=item_mapping,
                expand_mapping=True,
            )
            if "user_id" in split[subset]:
                split[subset], user_mapping = encode(
                    split[subset],
                    col="user_id",
                    mapping=user_mapping,
                    expand_mapping=True,
                )

    metadata, item_mapping = encode(metadata, col="item_id", mapping=item_mapping, expand_mapping=True)

    split_stats = {}
    timestamp_unit = OmegaConf.select(config, "dataset.timestamp_unit", default="s")
    for dataset in split:
        if dataset == "item_stats":
            frequencies = split[dataset].groupby("training_events").size()
            split_stats[dataset] = {
                "n_items": len(split[dataset]),
                "frequency_histogram": [{"training_events": int(frequency), "n_items": int(count)}
                                        for frequency, count in frequencies.items()],
            }
            continue
        split_stats[dataset] = dataset_stats(split[dataset], extended=True, timestamp_unit=timestamp_unit)
        stats = split_stats[dataset]
        print(f"{dataset}: {stats['n_interactions']:,} events, "
              f"{stats['n_users']:,} users, {stats['n_items']:,} items")

    split_stats["protocol"] = getattr(splitting_strategy, "statistics", {})
    os.makedirs(output_dir, exist_ok=False)

    # save config
    OmegaConf.save(config, os.path.join(output_dir, "config.yaml"))
    # save mappings
    with open(os.path.join(output_dir, "item_mapping.pkl"), mode="wb") as file:
        pickle.dump(item_mapping, file)
    with open(os.path.join(output_dir, "user_mapping.pkl"), mode="wb") as file:
        pickle.dump(user_mapping, file)
    # save data
    for dataset in split:
        split[dataset].to_parquet(os.path.join(output_dir, f'{dataset}.parquet'), index=False)
    # save stats
    for key in ("seq_len_distribution", "item_occurrence_distribution", "frequency_histogram"):
        distributions = [pd.DataFrame(stats.pop(key)).assign(subset=subset)
                         for subset, stats in split_stats.items() if key in stats]
        if distributions:
            pd.concat(distributions, ignore_index=True).to_csv(
                os.path.join(output_dir, f"{key}.csv"), index=False)
    with open(os.path.join(output_dir, "statistics.json"), mode="w", encoding="utf-8") as file:
        json.dump(split_stats, file, indent=2, default=str)


if __name__ == "__main__":

    main()
