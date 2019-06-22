# pylint: disable=no-self-use
import pytest
from typing import List

from allennlp.common import Params
from allennlp.data import Vocabulary
from allennlp.data.dataset import Batch
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.token_indexers import SingleIdTokenIndexer


@pytest.fixture(scope="module")
def data_path() -> str:
    return "./data/Data_136_with_feats.txt"


@pytest.fixture(scope="function")
def params() -> Params:
    params = Params({
        "type": "handcrafted_feature_reader",
        "token_indexers": {
            "tokens": "single_id"
        },
        "features_index_map": "./data/features.txt"
    })
    return params


class TestHandCraftedFeatureReader(object):
    def test_can_load_from_params(self, params: Params) -> None:
        reader = DatasetReader.from_params(params)
        assert isinstance(reader._token_indexers["tokens"], SingleIdTokenIndexer)

    def test_can_read_instances(self, data_path: str, params: Params) -> None:
        reader = DatasetReader.from_params(params)
        instances = reader.read(data_path)
        assert len(instances) == 136
        vocab = Vocabulary.from_instances(instances)
        batch_size = 32
        for ix in range(0, len(instances), batch_size):
            batch = Batch(instances[ix: ix + batch_size])
            batch.index_instances(vocab)
            tensor_dict = batch.as_tensor_dict()
            assert tensor_dict["tokens"]["tokens"].size() == \
                (tensor_dict["features"].size(0), tensor_dict["features"].size(1))
