from __future__ import absolute_import
import pytest
from typing import List, Dict
import numpy as np

from allennlp.common import Params
from allennlp.data import DatasetReader, Vocabulary
from ccm_model.modules.constrained_conditional_module import ConstrainedConditionalModule


@pytest.fixture(scope="module")
def logits():
    yield np.array([
        [[0, 0, .5, .5, .2], [0, 0, .3, .3, .1], [0, 0, .9, 10, 1]],
        [[0, 0, .2, .5, .2], [0, 0, 3, .3, .1], [0, 0, .9, 1, 1]],
    ])


@pytest.fixture(scope="module")
def mask():
    yield np.array([
        [1, 1, 1],
        [1, 1, 0]
    ])


@pytest.fixture(scope="module")
def transitions():
    yield np.array([
        [0.1, 0.2, 0.3, 0.4, 0.5],
        [0.8, 0.3, 0.1, 0.7, 0.9],
        [-0.3, 2.1, -5.6, 3.4, 4.0],
        [0.2, 0.4, 0.6, -0.3, -0.4],
        [1.0, 1.0, 1.0, 1.0, 1.0]
    ])


@pytest.fixture(scope="module")
def constraints():
    yield [(0, 0), (0, 1),
           (1, 1), (1, 2),
           (2, 2), (2, 3),
           (3, 3), (3, 4),
           (4, 4), (4, 0)]


@pytest.fixture(scope="module")
def start_transitions():
    yield np.array([0.1, 0.2, 0.3, 0.4, 0.6])


@pytest.fixture(scope="module")
def end_transitions():
    yield np.array([-0.1, -0.2, 0.3, -0.4, -0.4])


@pytest.fixture(scope="module")
def data_path() -> str:
    yield "./data/Data_136_with_feats.txt"


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


@pytest.fixture(scope="function")
def ccm_params() -> Params:
    params = Params({
        "hard_constraints": ["type"],
        "soft_constraints": {"attr": 7.86},
        "sentence_penalty_map": {"I-type": 50.},
        "constrain_crf_decoding": True,
        "label_encoding": "IOB1"
    })
    yield params


class TestCcmModule(object):
    def test_ccm_decode(self, logits, mask, transitions,
                        start_transitions, end_transitions) -> None:

        ccm_module = ConstrainedConditionalModule(5)
        predicted_tags = ccm_module.ccm_tags(logits, mask, transitions, start_transitions, end_transitions)
        viterbi_tags = [[2, 4, 3],
                        [4, 2]]
        assert predicted_tags == viterbi_tags

    def test_ccm_constrained_decode(
        self, logits, mask, transitions,
        start_transitions, end_transitions, constraints
    ) -> None:

        ccm_module = ConstrainedConditionalModule(5, constraints)
        predicted_tags = ccm_module.ccm_tags(logits, mask, transitions, start_transitions, end_transitions)
        viterbi_tags = [[2, 3, 3],
                        [2, 3]]
        assert predicted_tags == viterbi_tags

    def test_hard_constrained_decode(
        self, logits, mask, transitions,
        start_transitions, end_transitions
    ) -> None:
        ccm_module = ConstrainedConditionalModule(5, hard_constraints={"PER": [0, 1]})
        predicted_tags = ccm_module.ccm_tags(logits, mask, transitions, start_transitions, end_transitions)
        assert all(any(x in [0, 1] for x in y) for y in predicted_tags)

    def test_soft_constrained_decode(
        self, logits, mask, transitions,
        start_transitions, end_transitions
    ) -> None:
        ccm_module = ConstrainedConditionalModule(5, soft_constraints={"PER": ([0, 1], 0.0001)})
        predicted_tags = ccm_module.ccm_tags(logits, mask, transitions, start_transitions, end_transitions)
        viterbi_tags = [[2, 4, 3],
                        [4, 2]]
        assert predicted_tags == viterbi_tags
        ccm_module = ConstrainedConditionalModule(5, soft_constraints={"PER": ([0, 1], 1000)})
        soft_tags = ccm_module.ccm_tags(logits, mask, transitions, start_transitions, end_transitions)
        ccm_module = ConstrainedConditionalModule(5, hard_constraints={"PER": [0, 1]})
        hard_tags = ccm_module.ccm_tags(logits, mask, transitions, start_transitions, end_transitions)
        assert soft_tags == hard_tags

    def test_partial_labels(
        self, logits, mask, transitions, start_transitions, end_transitions
    ) -> None:
        ccm_module = ConstrainedConditionalModule(5, soft_constraints={"PER": ([0, 1], 0.0001)})
        partial_labels = [[(1, 0)], [(0, 1), (1, 2)]]
        predicted_tags = ccm_module.ccm_tags(
            logits, mask, transitions, start_transitions, end_transitions, partial_labels
        )
        assert predicted_tags[0][1] == 0
        assert predicted_tags[1] == [1, 2]

    def test_sentence_markers(self, logits, mask, transitions, start_transitions, end_transitions) -> None:
        ccm_module = ConstrainedConditionalModule(
            5, hard_constraints={"PER": [1]}, sentence_penalty_map=(1, 0.))
        # test no penalty

        sentence_boundaries: List[List[int]] = [[2, 3], [1, 2]]
        p1 = ccm_module.ccm_tags(logits, mask, transitions, start_transitions, end_transitions,
                                 sentence_boundaries=sentence_boundaries)
        p2 = ccm_module.ccm_tags(logits, mask, transitions, start_transitions, end_transitions)
        assert p1 == p2
        # test with penalty
        ccm_module = ConstrainedConditionalModule(
            5, hard_constraints={"PER": [1]}, sentence_penalty_map=(1, 50.))
        p3 = ccm_module.ccm_tags(logits, mask, transitions, start_transitions, end_transitions,
                                 sentence_boundaries=sentence_boundaries)
        assert p3 == p1

    def test_from_params(self, data_path: str, sentence_marker_params: Params, ccm_params: Params) -> None:
        reader = DatasetReader.from_params(sentence_marker_params)
        instances = reader.read(data_path)
        vocab = Vocabulary.from_instances(instances)
        ccm_module = ConstrainedConditionalModule.from_params(vocab=vocab, params=ccm_params)
        index = vocab.get_token_index("I-type", "labels")
        assert ccm_module._sentence_penalty_map == (index, 50.)
