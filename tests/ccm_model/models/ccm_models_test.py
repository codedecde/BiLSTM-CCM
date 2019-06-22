from __future__ import absolute_import
import pytest
from typing import Dict
import torch
from copy import deepcopy
import os

from allennlp.common import Params
from allennlp.data import Token, Vocabulary, Instance, DatasetReader
from allennlp.data.dataset import Batch
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.fields import SequenceLabelField, TextField, Field
from allennlp.models import Model


@pytest.fixture(scope="module")
def instance():
    text = ["Nice", "pleasant", "beach", "in", "Tahiti"]
    tokens = [Token(x) for x in text]
    tags = ["I-ATTR", "B-ATTR", "I-TYPE", "O", "I-LOC"]
    token_indexers = {"tokens": SingleIdTokenIndexer()}
    instance_field: Dict[str, Field] = {}
    sequence = TextField(tokens, token_indexers)
    instance_field["tokens"] = sequence
    instance_field["tags"] = SequenceLabelField(tags, sequence, "labels")
    instance = Instance(instance_field)
    yield instance


@pytest.fixture(scope="function")
def params():
    params = Params({
        "type": "ccm_model",
        "label_encoding": "IOB1",
        "constrain_crf_decoding": True,
        "calculate_span_f1": True,
        "dropout": 0.5,
        "text_field_embedder": {
            "token_embedders": {
                "tokens": {
                    "type": "embedding",
                    "embedding_dim": 50
                }
            }
        },
        "encoder": {
            "type": "lstm",
            "input_size": 50,
            "hidden_size": 200,
            "num_layers": 2,
            "dropout": 0.5,
            "bidirectional": True
        },
        "ccm_decoder": {
            "label_encoding": "IOB1",
            "constrain_crf_decoding": True,
            "label_namespace": "labels",
            "hard_constraints": ["TYPE"],
            "soft_constraints": {"ATTR": 0.5}
        }
    })
    yield params


@pytest.fixture(scope="module")
def handcrafted_features_params():
    params = Params({
        "type": "ccm_model",
        "label_encoding": "IOB1",
        "num_features": 282,
        "constrain_crf_decoding": True,
        "calculate_span_f1": True,
        "dropout": 0.5,
        "text_field_embedder": {
            "token_embedders": {
                "tokens": {
                    "type": "embedding",
                    "embedding_dim": 50
                }
            }
        },
        "encoder": {
            "type": "lstm",
            "input_size": 50 + 282,
            "hidden_size": 200,
            "num_layers": 2,
            "dropout": 0.5,
            "bidirectional": True
        },
        "ccm_decoder": {
            "label_encoding": "IOB1",
            "constrain_crf_decoding": True,
            "label_namespace": "labels",
            "hard_constraints": ["TYPE"],
            "soft_constraints": {"ATTR": 0.5}
        }
    })
    yield params


@pytest.fixture(scope="module")
def handcrafted_features_reader_params():
    params = Params({
        "type": "handcrafted_feature_reader",
        "token_indexers": {
            "tokens": "single_id"
        },
        "features_index_map": "./data/features.txt"
    })
    yield params


@pytest.fixture(scope="function")
def params_without_start_end_transitions():
    params = Params({
        "type": "ccm_model",
        "label_encoding": "IOB1",
        "constrain_crf_decoding": True,
        "calculate_span_f1": True,
        "include_start_end_transitions": False,
        "dropout": 0.5,
        "text_field_embedder": {
            "token_embedders": {
                "tokens": {
                    "type": "embedding",
                    "embedding_dim": 50
                }
            }
        },
        "encoder": {
            "type": "lstm",
            "input_size": 50,
            "hidden_size": 200,
            "num_layers": 2,
            "dropout": 0.5,
            "bidirectional": True
        },
        "ccm_decoder": {
            "label_encoding": "IOB1",
            "constrain_crf_decoding": True,
            "label_namespace": "labels",
            "hard_constraints": ["TYPE"],
            "soft_constraints": {"ATTR": 0.5}
        }
    })
    yield params


@pytest.fixture(scope="module")
def logits():
    yield torch.rand(1, 5, 5)


@pytest.fixture(scope="module")
def mask():
    yield torch.ByteTensor([[1, 1, 1, 1, 1]])


class TestCcmModel(object):
    def test_from_params(self, instance: Instance, params: Params) -> None:
        # pylint: disable=protected-access
        vocab = Vocabulary.from_instances([instance])
        model = Model.from_params(vocab=vocab, params=params)
        assert model._ccm_decoder._num_tags == vocab.get_vocab_size("labels")
        itype_index = vocab.get_token_index("I-TYPE", namespace="labels")
        iattr_index = vocab.get_token_index("I-ATTR", namespace="labels")
        battr_index = vocab.get_token_index("B-ATTR", namespace="labels")
        assert model._ccm_decoder._hard_constraints == {"TYPE": [itype_index]}
        assert model._ccm_decoder._soft_constraints["ATTR"][1] == 0.5
        assert set(model._ccm_decoder._soft_constraints["ATTR"][0]) == set([iattr_index, battr_index])

    def test_decode(self, instance: Instance, params: Params,
                    logits: torch.Tensor, mask: torch.ByteTensor) -> None:
        vocab = Vocabulary.from_instances([instance])
        model = Model.from_params(vocab=vocab, params=params)
        output_dict = {
            "loss": 0.124,
            "logits": logits,
            "mask": mask
        }
        output_dict = model.decode(output_dict)
        assert len(output_dict["tags"]) == 1
        assert len(output_dict["tags"][0]) == 5

    def test_decode_without_start_end_transitions(
        self, instance: Instance, params_without_start_end_transitions: Params,
        logits: torch.Tensor, mask: torch.ByteTensor
    ) -> None:
        vocab = Vocabulary.from_instances([instance])
        model = Model.from_params(vocab=vocab, params=params_without_start_end_transitions)
        output_dict = {
            "loss": 0.124,
            "logits": logits,
            "mask": mask
        }
        output_dict = model.decode(output_dict)
        assert len(output_dict["tags"]) == 1
        assert len(output_dict["tags"][0]) == 5

    @pytest.mark.skipif(not os.path.exists("./data/Data_136_with_feats.txt"), reason="Data File not found")
    def test_handcrafted_module(
        self, handcrafted_features_params, handcrafted_features_reader_params
    ) -> None:
        reader = DatasetReader.from_params(handcrafted_features_reader_params)
        instances = reader.read("./data/Data_136_with_feats.txt")
        vocab = Vocabulary.from_instances(instances)
        model = Model.from_params(vocab=vocab, params=handcrafted_features_params)
        test_instances = instances[:10]
        batch = Batch(test_instances)
        batch.index_instances(vocab)
        output = model.forward(**batch.as_tensor_dict())
        assert "loss" in output
