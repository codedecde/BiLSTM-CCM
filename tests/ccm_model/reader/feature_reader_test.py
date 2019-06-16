# pylint: disable=no-self-use
import pytest

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
        import pdb; pdb.set_trace()
