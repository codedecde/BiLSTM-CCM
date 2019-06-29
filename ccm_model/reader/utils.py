from __future__ import absolute_import
from typing import List

from allennlp.data import Token


def get_sentence_markers_from_tokens(tokens: List[Token]) -> List[int]:
    punctuation_set = set([".", "?", "!"])
    sentence_markers: List[int] = []
    for ix in range(1, len(tokens)):
        token = tokens[ix]
        if token.text in punctuation_set:
            continue
        if tokens[ix - 1].text in punctuation_set:
            sentence_markers.append(ix)
    sentence_markers.append(len(tokens))
    return sentence_markers


def get_sentences_from_markers(tokens: List[Token], markers: List[int]) -> List[List[Token]]:
    sentences: List[List[Token]] = []
    start = 0
    for end in markers:
        sentences.append(tokens[start: end])
        start = end
    return sentences
