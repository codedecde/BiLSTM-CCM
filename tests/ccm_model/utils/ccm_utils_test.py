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
def viterbi_tags():
    yield [[2, 4, 3],
           [4, 2]]


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
def start_transitions():
    yield np.array([0.1, 0.2, 0.3, 0.4, 0.6])


@pytest.fixture(scope="module")
def end_transitions():
    yield np.array([-0.1, -0.2, 0.3, -0.4, -0.4])


class TestCCMDecode(object):
    def test_ccm_decode(self, logits, mask, transitions,
                        start_transitions, end_transitions, viterbi_tags):
        lengths = mask.sum(-1)
        predicted_viterbi_tags = []
        for index in range(logits.shape[0]):
            single_logits = logits[index, :lengths[index], :]
            predictions = ccm_decode(single_logits, transitions, start_transitions, end_transitions)
            predicted_viterbi_tags.append(predictions)
        assert predicted_viterbi_tags == viterbi_tags
