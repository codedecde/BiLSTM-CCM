from __future__ import absolute_import
from typing import List

import numpy as np
import cvxpy as cp


def ccm_decode(logits: np.array, transitions: np.array,
               start_transitions: np.array, end_transitions: np.array) -> List[int]:
    """Models the viterbi decoding as a Mixed Integer Problem. This allows for
    incorporating global constraints
    Parameters:
        logits (``Tensor``): seq_len x num_tags
        transitions (``Tensor``): num_tags x num_tags: current_tag -> next_tag logits
        start_transitions (``Tensor``): num_tags -> <START_TAG> -> tag logits
        end_transitions (``Tensor``): num_tags -> current_tag -> <END TAG> logits
    Returns:
        tags: List[int]: The ccm based decoded outputs
    """
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
    # defining the objective function
    variables = [cp.Variable((num_tags), integer=True)]
    variables += [cp.Variable((num_tags, num_tags), integer=True) for _ in range(seq_len - 1)]
    expression = cp.sum(cp.multiply(init_weights, variables[0]))
    for index in range(1, len(variables)):
        expression += cp.sum(cp.multiply(weights[index - 1], variables[index]))
    objective = cp.Maximize(expression)
    # defining the constraints
    constraints = []
    # first the lower and upper bounds for the integers
    for variable in variables:
        constraints.append(variable <= 1)
        constraints.append(variable >= 0)
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
                cp.sum(variables[index + 1][common_tag, :]))
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
