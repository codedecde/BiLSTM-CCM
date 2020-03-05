# pylint: disable=no-self-use
from __future__ import absolute_import
from typing import List

from allennlp.data import Token

from ccm_model.reader.utils import get_sentence_markers_from_tokens


def _convert_to_tokens(sentence: List[str]) -> List[Token]:
    return [Token(x) for x in sentence]


class TestUtils(object):
    def test_get_sentence_markers_from_tokens(self):
        # test 1: no punctuations
        sentence = ["Hi", "Let's", "meet", "next", "week"]
        assert get_sentence_markers_from_tokens(
            _convert_to_tokens(sentence)) == [5]
        # test 2: punctuation in the end
        sentence = ["Hi", "Let's", "meet", "next", "week", "."]
        assert get_sentence_markers_from_tokens(
            _convert_to_tokens(sentence)) == [6]
        # test 3: punctuation in the middle
        sentence = ["Hi", "!", "Let's", "meet", "next", "week"]
        assert get_sentence_markers_from_tokens(
            _convert_to_tokens(sentence)) == [2, 6]
        # test 4: punctuation in the middle and end
        sentence = ["Hi", "!", "Let's", "meet", "next", "week", "."]
        assert get_sentence_markers_from_tokens(
            _convert_to_tokens(sentence)) == [2, 7]
        # test 5: punctuation is continuation 1
        sentence = ["Hi", "!", "Let's", "meet", "next", "week", ".", "!"]
        assert get_sentence_markers_from_tokens(
            _convert_to_tokens(sentence)) == [2, len(sentence)]
        # test 6: punctuation is continuation 2
        sentence = ["Hi", "?", "!", "Let's", "meet", "next", "week", ".", "!"]
        assert get_sentence_markers_from_tokens(
            _convert_to_tokens(sentence)) == [3, len(sentence)]
        # test 7: single utterance
        sentence = ["hi"]
        assert get_sentence_markers_from_tokens(
            _convert_to_tokens(sentence)) == [len(sentence)]
        # test 8: start with punct 1:
        sentence = ["."]
        assert get_sentence_markers_from_tokens(
            _convert_to_tokens(sentence)) == [len(sentence)]
        # test 8: start with punct 1:
        sentence = [".", "hi"]
        assert get_sentence_markers_from_tokens(
            _convert_to_tokens(sentence)) == [1, len(sentence)]
