"""Make split to train, validation and test."""

import json
import os
import pickle

import hydra
from omegaconf import DictConfig, OmegaConf, open_dict

from src.preprocess.indexes import encode
from src.preprocess.utils import dataset_stats
from src.run_utils import pandas_reader


@hydra.main(version_base=None, config_path="config", config_name="split")
def main(config: DictConfig) -> None:

    print(OmegaConf.to_yaml(config, resolve=True))

    dataset_name = config.dataset.name
    base_path = os.path.join(os.environ["SEQ_REC_DATA_PATH"], "preprocessed", dataset_name)

    data = pandas_reader(os.path.join(base_path, f"{dataset_name}.parquet"))
    metadata = pandas_reader(os.path.join(base_path, f"{dataset_name}_meta.parquet"))

    splitting_strategy = hydra.utils.instantiate(config.splitting_strategy)
    # split() returns train / validation (combined input+target) / test_input / test_target.
    # Input/target separation, cold filtering and alignment happen inside the splitter.
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
        split_stats[dataset] = dataset_stats(split[dataset], extended=True, timestamp_unit=timestamp_unit)
        print(f"{dataset} statistics")
        print(split_stats[dataset])

    output_dir = os.path.join(os.environ["SEQ_REC_DATA_PATH"], 'split', dataset_name, config.split_name)
    os.makedirs(output_dir, exist_ok=True)

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
    with open(os.path.join(output_dir, "statistics.json"), mode="w", encoding="utf-8") as file:
        json.dump(split_stats, file, indent=2, default=str)


if __name__ == "__main__":

    main()
