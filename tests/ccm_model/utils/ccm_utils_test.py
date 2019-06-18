import pytest

import numpy as np

from ccm_model.utils.ccm_utils import ccm_decode


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


class TestCCMDecode(object):
    def test_ccm_decode(self, logits, mask, transitions,
                        start_transitions, end_transitions):
        lengths = mask.sum(-1)
        viterbi_tags = [[2, 4, 3],
                        [4, 2]]
        predicted_viterbi_tags = []
        for index in range(logits.shape[0]):
            single_logits = logits[index, :lengths[index], :]
            predictions = ccm_decode(single_logits, transitions, start_transitions, end_transitions)
            predicted_viterbi_tags.append(predictions)
        assert predicted_viterbi_tags == viterbi_tags

    def test_ccm_decode_with_constraints(self, logits, mask, transitions,
                                         start_transitions, end_transitions,
                                         constraints):
        lengths = mask.sum(-1)
        viterbi_tags = [[2, 3, 3],
                        [2, 3]]
        predicted_viterbi_tags = []
        for index in range(logits.shape[0]):
            single_logits = logits[index, :lengths[index], :]
            predictions = ccm_decode(
                single_logits, transitions, start_transitions, end_transitions, constraints)
            predicted_viterbi_tags.append(predictions)
        assert predicted_viterbi_tags == viterbi_tags
