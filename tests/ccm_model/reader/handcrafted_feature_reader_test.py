# pylint: disable=no-self-use
import pytest
from typing import List
import re

from allennlp.common import Params
from allennlp.data import Vocabulary
from allennlp.data.dataset import Batch
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.token_indexers import SingleIdTokenIndexer

from ccm_model.reader.utils import get_sentences_from_markers


@pytest.fixture(scope="module")
def data_path() -> str:
    yield "./data/Data_136_with_feats.txt"


@pytest.fixture(scope="module")
def partial_data_path() -> str:
    yield "./data/partially_labeled_data_with_going_features.txt"


@pytest.fixture(scope="function")
def params() -> Params:
    params = Params({
        "type": "handcrafted_feature_reader",
        "token_indexers": {
            "tokens": "single_id"
        },
        "features_index_map": "./data/features.txt"
    })
    yield params


@pytest.fixture(scope="function")
def sentence_marker_params() -> Params:
    params = Params({
        "type": "handcrafted_feature_reader",
        "token_indexers": {
            "tokens": "single_id"
        },
        "features_index_map": "./data/features.txt",
        "use_sentence_markers": True
    })
    yield params


class TestHandCraftedFeatureReader(object):
    def test_can_load_from_params(self, params: Params) -> None:
        reader = DatasetReader.from_params(params)
        assert isinstance(reader._token_indexers["tokens"], SingleIdTokenIndexer)

    def test_can_read_instances(self, data_path: str, params: Params) -> None:
        reader = DatasetReader.from_params(params)
        instances = reader.read(data_path)
        instance_number = 34
        tmp_path = f"./data/instance_{instance_number}.txt"
        with open(tmp_path, "w") as fil:
            tokens = instances[instance_number]["tokens"].tokens
            labels = instances[instance_number]["tags"].labels
            labels = [re.sub(r"^.*-", "", tag) for tag in labels]
            assert len(tokens) == 86
            write_text_buf: List[str] = []
            for token, label in zip(tokens, labels):
                write_text_buf.append(f"{token.text} {label}")
            fil.write("\n".join(write_text_buf))
        assert len(instances) == 136
        vocab = Vocabulary.from_instances(instances)
        batch_size = 32
        for ix in range(0, len(instances), batch_size):
            batch = Batch(instances[ix: ix + batch_size])
            batch.index_instances(vocab)
            tensor_dict = batch.as_tensor_dict()
            assert tensor_dict["tokens"]["tokens"].size() == \
                (tensor_dict["features"].size(0), tensor_dict["features"].size(1))

    def test_can_read_partial_instances(self, partial_data_path, params: Params) -> None:
        reader = DatasetReader.from_params(params)
        instances = reader.read_partial(partial_data_path)
        vocab = Vocabulary.from_instances(instances)
        assert "partial_labels" in vocab._token_to_index
        assert set(vocab.get_token_to_index_vocabulary("partial_labels").keys()) == \
            set(["I-<UNK>", "O", "I-type", "I-attr", "I-location"])

    def test_sentence_markers(self, data_path: str, sentence_marker_params: Params) -> None:
        reader = DatasetReader.from_params(sentence_marker_params)
        instances = reader.read(data_path)
        # vocab = Vocabulary.from_instances(instances)
        for instance in instances:
            tokens = instance["tokens"]
            sentence_markers = instance["metadata"]["sentence_markers"]
            sentences = get_sentences_from_markers(tokens, sentence_markers)
            assert sum(len(x) for x in sentences) == len(tokens) == sentence_markers[-1]
