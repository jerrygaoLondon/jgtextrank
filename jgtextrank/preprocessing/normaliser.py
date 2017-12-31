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
Text normalization utility that transforms text into a single canonical form.

Normalising text before processing it can reduce noise and improve accuracy of term extraction algorithms
that relies on various frequency metrics.

Text normalization is frequently used when converting the numbers, punctuations, acronyms, abbreviations, accents,
irregular and ungrammatic constructs in specialised domain that are non-standard "words", which need to be standardized
depending on the context.

"""
import string
import six
import unicodedata

from nltk.stem.porter import *
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

__author__ = 'Jie Gao <j.gao@sheffield.ac.uk>'
__all__ = ["wordnet_lemmatizer", "punctutations", "punctuation_filter", "stemmer", "stem",
           "lemmatize", "normalize", "punctuation_filter_for_list", "remove_punctuations",
           "remove_digits", "punctuation_filter_for_word_level"]

# global variables
WS = ' '
COLLAPSE = re.compile(r'\s+')

# define unicode category here to filter out characters, such as symbols, punctuations, etc.
CATEGORY_DEFAULTS = {
    # 'C': WS,
    'M': '',
    # 'Z': WS,
    # 'P': WS,
    # 'S': WS
}

# convenient function
wordnet_lemmatizer = WordNetLemmatizer()
punctutations = string.punctuation
#remove punctuations from a string
punctuation_filter = lambda t : filter(lambda a: a not in string.punctuation, t)
stemmer = PorterStemmer()


def stem(word):
    return stemmer.stem(word)


def lemmatize(word, pos='n'):
    """
    lemmatize word by using NLTK wordnet package

    if empty pos is passed, it will skip the lemmatization

    :type word: string
    :param word: word to be lemmatised
    :type pos: char
    :param pos: four part-of-speech categories supported in NLTK wordnet
                SEE options in nltk.corpus.reader.wordnet.POS_LIST
                SEE
    :rtype: string
    :return: lemmatized word or original word if pos=''
    """
    if not pos.strip():
        return word

    return wordnet_lemmatizer.lemmatize(word, pos=pos)


def get_wordnet_pos(nltk_pos_tag):
    """
    map nltk PoS tags to WordNet PoS names

    :type nltk_pos_tag: string
    :param nltk_pos_tag: pos tag
    :type char
    :return: wordnet PoS name
    """
    if nltk_pos_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_pos_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_pos_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_pos_tag.startswith('R'):
        return wordnet.ADV
    else:
        return ''


def normalize(text, lowercase=True, collapse=True, decompose=True,
              replace_categories=CATEGORY_DEFAULTS, lemma=False, pos_tag="n"):
    """
     Taken from Normality library

    The main normalization function for text. This will take a string
    and apply a set of transformations to it so that it can be processed
    more easily afterwards.
    :type text: string
    :param text: word
    :type lowercase: bool
    :param lowercase: lower case the text
    :type collapse: bool
    :param collapse: replace multiple whitespace-like characters with a single whitespace.
            This is especially useful with category replacement which can lead to a lot of whitespace.
    :type decompose: bool
    :param decompose: apply a unicode normalization (NFKD) to separate
      simple characters and their diacritics.
    :param replace_categories: This will perform a replacement of whole classes of unicode characters
                (e.g. symbols, marks, numbers) with a given character.
                It is used to replace any non-text elements of the input string.
    :type lemma: bool
    :param lemma: if lemmatize the word. Default to false
    :type pos_tag: string
    :param pos_tag: PoS tag used to lemmatize the word.  the method will automatically map the tag to wordnet tag
    :rtype: string
    :return: normalised word
    """
    if not isinstance(text, six.string_types):
        return

    # Python 3?
    if six.PY2 and not isinstance(text, six.text_type):
        text = text.decode('utf-8')

    if lowercase:
        # Yeah I made a Python package for this.
        text = text.lower()

    # if transliterate:
    #    # Perform unicode-based transliteration, e.g. of cyricllic
    #    # or CJK scripts into latin.
    #    text = unidecode(text)
    #    if six.PY2:
    #        text = unicode(text)

    if decompose:
        # Apply a canonical unicoe normalization form, e.g.
        # transform all composite characters with diacritics
        # into a series of characters followed by their
        # diacritics as separate unicode codepoints.
        text = unicodedata.normalize('NFKD', text)

    # Perform unicode category-based character replacement. This is
    # used to filter out whole classes of characters, such as symbols,
    # punctuation, or whitespace-like characters.
    characters = []
    for character in text:
        category = unicodedata.category(character)[0]
        character = replace_categories.get(category, character)
        characters.append(character)
    text = u''.join(characters)
    # print(text)
    if collapse:
        # Remove consecutive whitespace.
        text = COLLAPSE.sub(WS, text).strip(WS)

    if lemma:
        text = lemmatize(text, pos=get_wordnet_pos(pos_tag))

    return text


def punctuation_filter_for_list(tokens=list()):
    return [remove_punctuations(token) for token in tokens]


def remove_punctuations(raw_term):
    replace_all_punc_term_trans = raw_term.maketrans(string.punctuation, ' '*len(string.punctuation))
    space_replace_punc_all_term = raw_term.translate(replace_all_punc_term_trans)
    #remove multiple spaces and trailing spaces
    space_replace_punc_all_term = ' '.join(space_replace_punc_all_term.split())
    return space_replace_punc_all_term


def remove_digits(raw_term):
    remove_all_digits_term_trans = raw_term.maketrans(string.digits, ' '*len(string.digits))
    remove_punc_raw_term = raw_term.translate(remove_all_digits_term_trans)
    remove_punc_raw_term = remove_punc_raw_term.strip()
    remove_punc_raw_term = ' '.join(remove_punc_raw_term.split())
    return remove_punc_raw_term


def punctuation_filter_for_word_level(tokens=list()):
    """
    to effectively avoid errors caused by tokenisation
    e.g., "\'lysis genes\'" -> ["\'lysis", 'genes', "\'"] -> output: ["lysis", 'genes']
    e.g., ['highli', 'purifi', 'monocytes/macrophag'] -> ['highli', 'purifi', 'monocytes', 'macrophag']
    :param tokens:
    :return: list, token lists
    """
    filtered_tokens = punctuation_filter(tokens)
    punc_filtered_tokens = list()
    for token in filtered_tokens:
        punc_filtered_token = remove_punctuations(token)
        if ' ' in punc_filtered_token:
            punc_filtered_tokens.extend(punc_filtered_token.split(' '))
        else:
            punc_filtered_tokens.append(''.join(punc_filtered_token))

    return punc_filtered_tokens
    #return [''.join(punctuation_filter(token)) for token in filtered_tokens]


if __name__ == '__main__':
    print(normalize("supporting", lemma=True, pos_tag="VBG"))

    print(normalize("inequations", lemma=True, pos_tag="n"))

    print(normalize("corresponding", lemma=True, pos_tag="JJ"))

