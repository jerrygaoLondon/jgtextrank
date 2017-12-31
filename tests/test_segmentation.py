import types

from jgtextrank.preprocessing.segmentation import word_2_tokenised_sentences

import unittest


class TestSegmentation(unittest.TestCase):

    def test_word_2_tokenised_sentences(self):
        example_abstract = "Compatibility of systems of linear constraints over the set of natural numbers. " \
                           "Criteria of compatibility of a system of linear Diophantine equations, strict inequations, " \
                           "and nonstrict inequations are considered. Upper bounds for components of a minimal set of " \
                           "solutions and algorithms of construction of minimal generating sets of solutions for all " \
                           "types of systems are given. These criteria and the corresponding algorithms for " \
                           "constructing a minimal supporting set of solutions can be used in solving all the " \
                           "considered types systems and systems of mixed types."
        tokenised_sentences = word_2_tokenised_sentences(example_abstract)
        assert isinstance(tokenised_sentences, types.GeneratorType)
        sentence_size = 0
        for tokenised_sentence in tokenised_sentences:
            sentence_size += 1
            assert isinstance(tokenised_sentence, types.GeneratorType)

        assert len(list(tokenised_sentences)) == 0
        assert sentence_size == 4