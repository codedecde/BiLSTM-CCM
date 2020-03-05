from __future__ import absolute_import
import pytest
from copy import deepcopy

from allennlp.data import DatasetReader, Vocabulary
from allennlp.models import CrfTagger, Model

from utils import read_from_config_file
from ccm_model.codl_main import train_single, get_training_data


@pytest.fixture(scope="module")
def inits():
    config_file = "./training_configs/basic/ner_test.jsonnet"
    config = read_from_config_file(config_file)

    reader_params = config.pop("dataset_reader")
    reader = DatasetReader.from_params(reader_params)
    data_file_path = config.pop("all_data_path")
    instances = reader.read(data_file_path)
    partial_data_path = config.pop("partial_data_path")
    partial_instances = reader.read(partial_data_path)
    return {"instances": instances, "config": config, "reader": reader, "partial_instances": partial_instances}


class TestCoDLFile(object):
    def test_get_training_data(self, inits):
        vocab = Vocabulary.from_instances(inits["instances"])
        model = Model.from_params(inits["config"].pop("model"), vocab=vocab)
        instances = deepcopy(inits["partial_instances"])
        new_instances = get_training_data(inits["partial_instances"], model, inits["reader"])
        assert len(new_instances) == len(instances)
        for instance, new_instance in zip(instances, new_instances):
            for prev_label, new_label in zip(instance.fields["tags"].labels, new_instance.fields["tags"].labels):
                if prev_label != "I-<UNK>":
                    assert prev_label == new_label
                else:
                    assert new_label != prev_label

    # def test_train_single(self, inits):
    #     config = inits["config"]
    #     instances = inits["instances"]
    #     index = 5
    #     out_pred = train_single(deepcopy(config), instances, index, -1)
    #     assert isinstance(out_pred, list)
    #     assert all([isinstance(x, str) for x in out_pred])
