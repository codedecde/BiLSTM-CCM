from __future__ import absolute_import
from typing import List, Dict, Optional, Tuple
import numpy as np
import cvxpy as cp
import re

from allennlp.data import Vocabulary
from allennlp.common import Registrable, Params
from allennlp.common.checks import ConfigurationError
from allennlp.modules.conditional_random_field import allowed_transitions


class ConstrainedConditionalModule(Registrable):
    def __init__(self,
                 num_tags: int,
                 transition_constraints: Optional[List[Tuple[int, int]]] = None,
                 hard_constraints: Optional[Dict[str, List[int]]] = None,
                 soft_constraints: Optional[Dict[str, Tuple[List[int], float]]] = None) -> None:
        self._transition_constraints = transition_constraints or []
        self._hard_constraints = hard_constraints or {}
        self._soft_constraints = soft_constraints or {}
        self._num_tags = num_tags

    def get_masked_transitions(
        self, transitions: np.array,
        start_transitions: Optional[np.array] = None,
        end_transitions: Optional[np.array] = None
    ) -> Tuple[np.array, np.array, np.array]:
        num_tags = self._num_tags
        constraints = self._transition_constraints
        if start_transitions is None:
            start_transitions = np.zeros(num_tags)

        if end_transitions is None:
            end_transitions = np.zeros(num_tags)
        constraint_mask = np.ones((num_tags + 2, num_tags + 2))
        if constraints:
            constraint_mask *= 0.
            for i, j in constraints:
                constraint_mask[i, j] = 1.
        constrained_transitions = (constraint_mask[:num_tags, :num_tags] * transitions) + \
            ((1. - constraint_mask[:num_tags, :num_tags]) * -10000.)
        start_tag = num_tags
        end_tag = num_tags + 1
        constrained_start_transitions = (constraint_mask[start_tag, :num_tags] * start_transitions) + \
            ((1. - constraint_mask[start_tag, :num_tags]) * -10000.)
        constrained_end_transitions = (constraint_mask[:num_tags, end_tag] * end_transitions) + \
            ((1. - constraint_mask[:num_tags, end_tag]) * -10000.)
        return constrained_transitions, constrained_start_transitions, constrained_end_transitions

    def ccm_decode(
        self, logits: np.array, transitions: np.array,
        start_transitions: Optional[np.array] = None,
        end_transitions: Optional[np.array] = None,
        partial_labels: Optional[List[Tuple[int, int]]] = None
    ) -> List[int]:
        transitions, start_transitions, end_transitions = self.get_masked_transitions(
            transitions, start_transitions, end_transitions)
        seq_len, num_tags = logits.shape

        init_weights = np.zeros((num_tags))
        weights = np.zeros((seq_len - 1, num_tags, num_tags))
        for index in range(logits.shape[0]):
            if index == 0:
                init_weights = logits[index] + start_transitions
            else:
                weights[index - 1] = transitions + logits[index].reshape(1, num_tags)
            if index == seq_len - 1:
                weights[index - 1] += end_transitions.reshape(1, num_tags)
        # defining the variable
        variables = [cp.Variable((num_tags), integer=True)]
        variables += [cp.Variable((num_tags, num_tags), integer=True) for _ in range(seq_len - 1)]
        dummy = None
        if self._soft_constraints:
            dummy = cp.Variable((len(self._soft_constraints)), integer=True)
        # defining the objective function
        expression = cp.sum(cp.multiply(init_weights, variables[0]))
        for index in range(1, len(variables)):
            expression += cp.sum(cp.multiply(weights[index - 1], variables[index]))
        for index, tag in enumerate(self._soft_constraints):
            _, penalty = self._soft_constraints[tag]
            expression -= (penalty * dummy[index])
        objective = cp.Maximize(expression)
        # defining the constraints
        constraints = []
        # first the lower and upper bounds for the integers
        for variable in variables:
            constraints.append(variable <= 1)
            constraints.append(variable >= 0)
        if dummy is not None:
            constraints.append(dummy >= 0)
            constraints.append(dummy <= 1)
        # define constraints for y_0
        constraints.append(cp.sum(variables[0]) == 1)
        # # define constraints for y_0 -> y_1
        for index, y0 in enumerate(variables[0]):
            constraints.append(y0 == cp.sum(variables[1][index]))
        # # define constraints for y_i -> y_(i+1)
        for index in range(1, len(variables) - 1):
            for common_tag in range(num_tags):
                constraints.append(
                    cp.sum(variables[index][:, common_tag]) ==
                    cp.sum(variables[index + 1][common_tag, :])
                )
        # now the hard constraints
        for var, indices in self._hard_constraints.items():
            expr = 0
            for index in indices:
                expr += variables[0][index]
            for six in range(1, len(variables)):
                for index in indices:
                    expr += cp.sum(variables[six][:, index])
            constraints.append(expr >= 1)
        # now the soft constraints
        for dix, tag in enumerate(self._soft_constraints):
            indices, _ = self._soft_constraints[tag]
            expr = dummy[dix]
            for index in indices:
                expr += variables[0][index]
            for six in range(1, len(variables)):
                for index in indices:
                    expr += cp.sum(variables[six][:, index])
            constraints.append(expr >= 1)

        # now the partially labeled examples
        if partial_labels:
            for pos, label in partial_labels:
                expr = 0
                expr += variables[pos][label] if pos == 0 else cp.sum(variables[pos][:, label])
                constraints.append(expr >= 1)

        # the final optimization problem
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.GLPK_MI)
        # now we decode the values
        tag_list: List[int] = []
        start_tag = np.argmax(variables[0].value)
        assert variables[0].value.sum() == 1
        assert variables[0].value[start_tag] == 1.
        tag_list.append(start_tag)
        prev_tag = start_tag
        for index in range(1, seq_len):
            curr_tag = np.argmax(variables[index].value[prev_tag, :])
            assert np.sum(variables[index].value) == 1
            assert variables[index].value[prev_tag, curr_tag] == 1
            tag_list.append(curr_tag)
            prev_tag = curr_tag

        return tag_list

    def ccm_tags(self, logits: np.ndarray, mask: np.ndarray,
                 transitions: np.ndarray,
                 start_transitions: Optional[np.array] = None,
                 end_transitions: Optional[np.array] = None,
                 partial_labels: Optional[List[List[Tuple[int, int]]]] = None) -> List[List[int]]:
        lengths = mask.astype(int).sum(-1)
        predicted_tags: List[List[int]] = []
        for ix in range(logits.shape[0]):
            example_partial_labels = partial_labels[ix] if partial_labels else None
            pred_val = self.ccm_decode(logits[ix, :lengths[ix], :], transitions,
                                       start_transitions, end_transitions, example_partial_labels)
            predicted_tags.append(pred_val)
            assert len(pred_val) == lengths[ix]
        return predicted_tags

    @classmethod
    def from_params(cls, vocab: Vocabulary, params: Params) -> 'ConstrainedConditionalModule':
        hard_constraints = params.pop("hard_constraints", [])
        soft_constraints = params.pop("soft_constraints", {})
        label_namespace = params.pop("label_namespace", "labels")
        constrain_crf_decoding = params.pop("constrain_crf_decoding", False)
        label_encoding = params.pop("label_encoding", None)
        hard_constraints_to_indices: Dict[str, List[int]] = {}
        for tag in hard_constraints:
            hard_constraints_to_indices[tag] = []
            for label, index in vocab.get_token_to_index_vocabulary(label_namespace).items():
                if re.match(rf"^.*-{tag}", label):
                    hard_constraints_to_indices[tag].append(index)
        soft_constraints = soft_constraints or {}
        soft_constraints_to_indices: Dict[str, Tuple[List[int], float]] = {}
        for tag, penalty in soft_constraints.items():
            indices = []
            for label, index in vocab.get_token_to_index_vocabulary(label_namespace).items():
                if re.match(rf"^.*-{tag}", label):
                    indices.append(index)
            soft_constraints_to_indices[tag] = (indices, penalty)
        num_tags = vocab.get_vocab_size(label_namespace)
        if constrain_crf_decoding:
            if not label_encoding:
                raise ConfigurationError("constrain_crf_decoding is True, but "
                                         "no label_encoding was specified.")
            labels = vocab.get_index_to_token_vocabulary(label_namespace)
            constraints = allowed_transitions(label_encoding, labels)
        else:
            constraints = None
        params.assert_empty(cls.__name__)
        return ConstrainedConditionalModule(num_tags, constraints,
                                            hard_constraints_to_indices,
                                            soft_constraints_to_indices)
