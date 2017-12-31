# -*- coding: utf-8 -*-
# ==============================================================================
#
# Authors: Jie Gao <j.gao@sheffield.ac.uk>
#
# Copyright (c) 2017 JIE GAO . All Rights Reserved.
#
# Permission is hereby granted, free of charge, to any person
# obtaining a copy of this software and associated documentation
# files (the "Software"), to deal in the Software without
# restriction, including without limitation the rights to use,
# copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following
# conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
# OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
# HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
# WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
# OTHER DEALINGS IN THE SOFTWARE.
#
# ==============================================================================
"""
Text segmentation utility divides text into meaningful units, such as tokens, sentences etc.

Abstracts from wikipedia:

The process of text segmentation is non-trivial task. While some written languages have explicit word boundary markers,
such as the word spaces of written English and the distinctive initial, medial and final letter shapes of Arabic,
such signals are sometimes ambiguous and not present in all written languages.

In English and many other languages using some form of the Latin alphabet, the space is a good approximation of a word
divider (word delimiter). (Some examples where the space character alone may not be sufficient include contractions
like can't for can not.). For sentence segmentation, using punctuation, particularly the full stop character is a
reasonable approximation in English and some other languages.  However even in English this problem is not trivial
due to the use of the full stop character for abbreviations, which may or may not also terminate a sentence.
For example Mr. is not its own sentence in "Mr. Smith went to the shops in Jones Street."
When processing plain text, tables of abbreviations that contain periods can help prevent incorrect assignment of
sentence boundaries.

Some segmentation process may need to analyse the structure of single linguistic unit, which is known as morphological
analysis (https://en.wikipedia.org/wiki/Morphology_(linguistics)).
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__)))

import nltk
import re
import string
from itertools import chain
from abc import ABCMeta, abstractmethod
from jgtextrank.decorators import requires_nltk_corpus

PUNCTUATION_REGEX = re.compile('[{0}]'.format(re.escape(string.punctuation)))

from nltk import pos_tag
__author__ = 'Jie Gao <j.gao@sheffield.ac.uk>'

#from nltk.data import load
#_POS_TAGGER = 'taggers/maxent_treebank_pos_tagger/english.pickle'
#tagger = load(_POS_TAGGER)

def pos_tagging():
    """
    Use NLTK's currently recommended part of speech tagger to tag the given list of tokens.

    Usage:    pos_tag(word_tokenize(sent_content))
    :return:
    """

    return  pos_tag


def strip_punc(s, all=False):
    """Removes punctuation from a string.

    :param s: The string.
    :param all: Remove all punctuation. If False, only removes punctuation from
        the ends of the string.
    """
    if all:
        return PUNCTUATION_REGEX.sub('', s.strip())
    else:
        return s.strip().strip(string.punctuation)


def with_metaclass(meta, *bases):
    """
    This method is from TextBlob library.
    refer to <textblob.compat.py>

    Defines a metaclass.

    Creates a dummy class with a dummy metaclass. When subclassed, the dummy
    metaclass is used, which has a constructor that instantiates a
    new class from the original parent. This ensures that the dummy class and
    dummy metaclass are not in the inheritance tree.

    Credit to Armin Ronacher.
    """
    class metaclass(meta):
        __call__ = type.__call__
        __init__ = type.__init__
        def __new__(cls, name, this_bases, d):
            if this_bases is None:
                return type.__new__(cls, name, (), d)
            return meta(name, bases, d)
    return metaclass('temporary_class', None, {})


class BaseTokenizer(with_metaclass(ABCMeta), nltk.tokenize.api.TokenizerI):
    """
    This class is from TextBlob library.
    refer to <textblob.tokenizers.py>

    Abstract base class from which all Tokenizer classes inherit.
    Descendant classes must implement a ``tokenize(text)`` method
    that returns a list of noun phrases as strings.
    """
    @abstractmethod
    def tokenize(self, text):
        """Return a list of tokens (strings) for a body of text.

        :rtype: list
        """
        return

    def itokenize(self, text, *args, **kwargs):
        """Return a generator that generates tokens "on-demand" for lower memory usage.

        .. versionadded:: 0.6.0

        :rtype: generator
        """
        return (t for t in self.tokenize(text, *args, **kwargs))


class WordTokenizer(BaseTokenizer):
    """
    This class is from TextBlob library.
    refer to <textblob.tokenizers.py>

    NLTK's recommended word tokenizer (currently the TreeBankTokenizer).
    Uses regular expressions to tokenize text. Assumes text has already been
    segmented into sentences.

    Performs the following steps:

    * split standard contractions, e.g. don't -> do n't
    * split commas and single quotes
    * separate periods that appear at the end of line
    """

    def tokenize(self, text, include_punc=True):
        """Return a list of word tokens.

        :param text: string of text.
        :param include_punc: (optional) whether to include punctuation as separate tokens. Default to True.
        """
        tokens = nltk.tokenize.word_tokenize(text)
        if include_punc:
            return tokens
        else:
            # Return each word token
            # Strips punctuation unless the word comes from a contraction
            # e.g. "Let's" => ["Let", "'s"]
            # e.g. "Can't" => ["Ca", "n't"]
            # e.g. "home." => ['home']
            return [word if word.startswith("'") else strip_punc(word, all=False)
                    for word in tokens if strip_punc(word, all=False)]


class SentenceTokenizer(BaseTokenizer):
    """
    This class is from TextBlob library.
    refer to <textblob.tokenizers.py>

    NLTK's sentence tokenizer (currently PunkSentenceTokenizer).
    Uses an unsupervised algorithm to build a model for abbreviation words,
    collocations, and words that start sentences,
    then uses that to find sentence boundaries.
    """

    @requires_nltk_corpus
    def tokenize(self, text):
        """Return a list of sentences."""
        return nltk.tokenize.sent_tokenize(text)

#: Convenience function for tokenizing sentences
sent_tokenize = SentenceTokenizer().itokenize

_word_tokenizer = WordTokenizer()  # Singleton word tokenizer


def word_tokenize(text, include_punc=True, *args, **kwargs):
    """Convenience function for tokenizing text into words.
    sentence splitting -> tokenization

    NOTE: NLTK's word tokenizer expects sentences as input, so the text will be
    tokenized to sentences before being tokenized to words.
    :rtype: generator[of tokens]
    :return tokens
    """
    words = chain.from_iterable(
        _word_tokenizer.itokenize(sentence, include_punc=include_punc,
                                  *args, **kwargs)
        for sentence in sent_tokenize(text))
    return words


def word_2_tokenised_sentences(text, include_punc=True, *args, **kwargs):
    """

    :param text:
    :param include_punc:
    :param args:
    :param kwargs:
    :type generatorType
    :return:
    """
    sentences = sent_tokenize(text)
    for sentence in sentences:
        yield _word_tokenizer.itokenize(sentence, include_punc=include_punc,
                                        *args, **kwargs)

if __name__ == '__main__':
    tokenised_text = ['Compatibility', 'of', 'systems', 'of', 'linear', 'constraints', 'over', 'the', 'set', 'of', 'natural', 'numbers', '.', 'Criteria', 'of', 'compatibility', 'of', 'a', 'system', 'of', 'linear', 'Diophantine', 'equations', ',', 'strict', 'inequations', ',', 'and', 'nonstrict', 'inequations', 'are', 'considered', '.', 'Upper', 'bounds', 'for', 'components', 'of', 'a', 'minimal', 'set', 'of', 'solutions', 'and', 'algorithms', 'of', 'construction', 'of', 'minimal', 'generating', 'sets', 'of', 'solutions', 'for', 'all', 'types', 'of', 'systems', 'are', 'given', '.', 'These', 'criteria', 'and', 'the', 'corresponding', 'algorithms', 'for', 'constructing', 'a', 'minimal', 'supporting', 'set', 'of', 'solutions', 'can', 'be', 'used', 'in', 'solving', 'all', 'the', 'considered', 'types', 'systems', 'and', 'systems', 'of', 'mixed', 'types', '.']
    print(sent_tokenize(tokenised_text))
