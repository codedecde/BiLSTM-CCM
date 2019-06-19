# pylint: disable=no-self-use
import pytest
from typing import List
import numpy as np

from allennlp.common import Params
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.token_indexers import SingleIdTokenIndexer


@pytest.fixture(scope="module")
def data_path() -> str:
    return "./data/Data_136_with_feats.txt"


class TestFeatureReader(object):
    def test_can_load_from_params(self) -> None:
        params = Params({
            "type": "feature_reader",
            "token_indexers": {
                "tokens": "single_id"
            }
        })
        reader = DatasetReader.from_params(params)
        assert isinstance(reader._token_indexers["tokens"], SingleIdTokenIndexer)

    def test_can_read_instances(self, data_path: str) -> None:
        params = Params({
            "type": "feature_reader",
            "token_indexers": {
                "tokens": "single_id"
            }
        })
        reader = DatasetReader.from_params(params)
        instances = reader.read(data_path)
        assert len(instances) == 136
        # compute attribute penalty here
        num_satisfied = 0
        for instance in instances:
            labels: List[str] = instance.fields["tags"].labels
            if any(x == "attr" for x in labels):
                num_satisfied += 1
        total = len(instances)
        num_unsatisfied = total - num_satisfied
        num_satisfied += 0.1
        num_unsatisfied += 0.1
        penalty = np.log(num_satisfied) - np.log(num_unsatisfied)
