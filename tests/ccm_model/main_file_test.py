from __future__ import absolute_import
import pytest
from copy import deepcopy

from allennlp.data import DatasetReader
from allennlp.models import CrfTagger

from utils import read_from_config_file
from ccm_model import get_trainer_from_config, train_single, post_process_prediction


@pytest.fixture(scope="module")
def inits():
    config_file = "./training_configs/basic/ner_test.jsonnet"
    config = read_from_config_file(config_file)

    reader_params = config.pop("dataset_reader")
    reader = DatasetReader.from_params(reader_params)
    data_file_path = config.pop("all_data_path")
    instances = reader.read(data_file_path)
    return {"instances": instances, "config": config}


class TestMainFile(object):
    def test_get_trainer_from_configs(self, inits):
        config = inits["config"]
        instances = inits["instances"]
        index = 5

        train_val_instances = instances[:index] + instances[index + 1:]
        assert len(train_val_instances) == len(instances) - 1
        num_train_instances = int(0.9 * len(train_val_instances))
        train_instances = train_val_instances[:num_train_instances]
        val_instances = train_val_instances[num_train_instances:]
        assert len(train_instances) + len(val_instances) == len(train_val_instances)
        assert len(val_instances) == 14

        trainer = get_trainer_from_config(
            deepcopy(config),
            train_instances, val_instances, -1
        )
        assert isinstance(trainer.model, CrfTagger)

    def test_train_single(self, inits):
        config = inits["config"]
        instances = inits["instances"]
        index = 5
        out_pred = train_single(deepcopy(config), instances, index, -1)
        assert isinstance(out_pred, list)
        assert all([isinstance(x, str) for x in out_pred])
