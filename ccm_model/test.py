from __future__ import absolute_import
from typing import Dict, Any, List
import os
import json
import logging
import tqdm
import argparse

from allennlp.data import DatasetReader
from allennlp.models import Model

from utils import setup_logger, read_from_config_file, bool_flag

logger = logging.getLogger(__name__)


def get_arguments() -> Dict[str, Any]:
    parser = argparse.ArgumentParser("Testing a model")
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--config_file", type=str, required=True)
    parser.add_argument("--cuda", type=bool_flag, default=False)
    parser.add_argument("--test_folder", type=str, required=True)
    parser.add_argument("--outfile", type=str, required=False, default=None)
    args = parser.parse_args()
    return vars(args)


def test(args: Dict[str, Any]):
    """This is the testing function. Usually, different models
    may have different testing behaviour, depending on what they
    are trying to achieve

    Parameters:
        args (Dict[str, Any]): A dictionary consisting of the following
            keys:
                * base_dir (str): The base model directory to load
                    the model from
                * infile (str): The file to load the test data from
    """
    setup_logger()
    model_dir = args["model_dir"]
    config_file = args["config_file"]
    config = read_from_config_file(config_file)

    reader = DatasetReader.from_params(config.pop("dataset_reader"))
    reader.eval()
    device = 0 if args["cuda"] else -1
    model = Model.load(
        config=config,
        serialization_dir=os.path.join(model_dir),
        cuda_device=device)
    instances = []
    index_to_file_map = {}
    for index, file in enumerate(os.listdir(args["test_folder"])):
        start_index = len(instances)
        instances += reader.read(os.path.join(args["test_folder"], file))
        end_index = len(instances)
        for ix in range(start_index, end_index):
            index_to_file_map[ix] = file

    batch_size = 32
    predictions = []
    for ix in tqdm.tqdm(range(0, len(instances), batch_size)):
        batched_instances = instances[ix: ix + batch_size]
        predictions += model.forward_on_instances(batched_instances)

    outfile = args["outfile"]
    if outfile is not None:
        predictions_by_file: Dict[str, List[List[str]]] = {}
        for index, file_name in index_to_file_map.items():
            if file_name not in predictions_by_file:
                predictions_by_file[file_name] = []
            predictions_by_file[file_name].append(predictions[index])

        with open(outfile, "w") as f:
            json.dump(predictions_by_file, f, indent=4)

    # metrics = model.get_metrics(reset=True)
    # for metric in metrics:
    #     metric_val = metrics[metric]
    #     logger.info(f"{metric}: {metric_val:0.2f}")


if __name__ == "__main__":
    args = get_arguments()
    test(args)
