from __future__ import absolute_import
import pytest
import numpy as np

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
