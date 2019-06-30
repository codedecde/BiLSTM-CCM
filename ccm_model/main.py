from __future__ import absolute_import
from typing import List, Optional, Tuple
import argparse
from copy import deepcopy
import os
import tqdm
import re
import torch
import random

from allennlp.data import DatasetReader, Instance, Vocabulary
from allennlp.data.iterators import DataIterator
from allennlp.common import Params
from allennlp.models import Model
from allennlp.training import Trainer
from allennlp.common.util import JsonDict

from utils import read_from_config_file
from allennlp.common.util import sanitize


def get_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser("ccm/crf tagger")
    parser.add_argument("-cf", "--config_file", type=str, required=True,
                        help="The configuration file")
    parser.add_argument("--base_dir", type=str, required=True,
                        help="The Output directory file")
    parser.add_argument("-d", "--devices", type=int,
                        default=[-1], nargs="+")

    parser.add_argument("-six", "--start_index", type=int, required=False, default=None)
    parser.add_argument("-eix", "--end_index", type=int, required=False, default=None)
    args = parser.parse_args()
    return args


def get_trainer_from_config(config: Params,
                            train_instances: List[Instance],
                            val_instances: List[Instance],
                            device: int) -> Trainer:
    trainer_params = config.pop("trainer")
    trainer_params["cuda_device"] = device
    model_params = config.pop("model")
    vocab = Vocabulary.from_instances(train_instances)
    model = Model.from_params(model_params, vocab=vocab)
    iterator = DataIterator.from_params(config.pop("iterator"))
    iterator.index_with(vocab)
    trainer = Trainer.from_params(
        model=model,
        iterator=iterator,
        train_data=train_instances,
        validation_data=val_instances,
        serialization_dir=None,
        params=trainer_params)
    return trainer


def post_process_prediction(prediction: JsonDict) -> List[str]:
    tags = prediction["tags"]
    tags = [re.sub(r"^.*-", "", tag) for tag in tags]
    return tags


def train_single(config: Params,
                 instances: List[Instance],
                 index: int,
                 cuda_device: int) -> Tuple[List[str], Model]:
    instances = deepcopy(instances)
    config = deepcopy(config)
    test_instance = instances[index]
    train_val_instances = instances[:index] + instances[index + 1:]
    random.shuffle(train_val_instances)
    num_train_instances = int(0.9 * len(train_val_instances))
    train_instances = train_val_instances[:num_train_instances]
    val_instances = train_val_instances[num_train_instances:]
    trainer = get_trainer_from_config(config, train_instances, val_instances, cuda_device)
    trainer.train()
    model = trainer.model
    model.eval()
    prediction = sanitize(model.forward_on_instance(test_instance))
    return prediction, model.cpu()


def serial_processing(instances: List[Instance], config: Params, device: int,
                      serialization_dir: str, start_index: Optional[int] = None,
                      end_index: Optional[int] = None) -> None:
    start_index = start_index or 0
    end_index = end_index or len(instances)
    for index in tqdm.tqdm(range(start_index, end_index)):
        prediction, model = train_single(config, instances, index, device)
        with open(os.path.join(serialization_dir, f"prediction_{index}.txt"), "w") as f:
            f.write("\n".join(prediction))
        if index == end_index - 1:
            model_save_path = os.path.join(
                serialization_dir, f"best_model_start_index_{start_index}_end_index_{end_index}.th"
            )
            torch.save(model.state_dict(), model_save_path)


def main(args) -> None:
    config = read_from_config_file(args.config_file)
    serialization_dir = args.base_dir
    os.makedirs(serialization_dir, exist_ok=True)
    reader_params = config.pop("dataset_reader")
    reader = DatasetReader.from_params(reader_params)
    data_file_path = config.pop("all_data_path")
    instances = reader.read(data_file_path)
    serial_processing(instances, config, args.devices[0],
                      serialization_dir, args.start_index, args.end_index)


if __name__ == "__main__":
    args = get_arguments()
    main(args)
