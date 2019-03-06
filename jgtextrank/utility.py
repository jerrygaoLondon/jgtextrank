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
import multiprocessing.pool

__author__ = 'jieg'

import sys
import os
package_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(package_dir)

import os
import warnings
import functools
import string
import operator
import itertools
from collections import OrderedDict
from collections import Counter
import csv

from typing import Dict, List, Tuple, Generator
from .preprocessing.segmentation import word_tokenize
from .preprocessing.segmentation import SentenceTokenizer
from nltk.corpus import stopwords


_stop_words = stopwords.words('english')

#: Convenience functions
sent_tokenize = SentenceTokenizer().itokenize
stop_words_filter = lambda t : filter(lambda a: a not in _stop_words, t)
punctuation_filter = lambda t : filter(lambda a: a not in string.punctuation, t)


class CorpusContent2RawSentences(object):
    """
    This class can be used for textual corpus where sentences are not pre-processed.

    Raw lowercased tokenised sentences are returned.

    """
    def __init__(self, dirname: str, encoding: str = "utf-8", remove_stop_words: bool = False, remove_punc: bool = False):
        self.dirname = dirname
        # encoding: iso-8859-1 for ttc corpus
        self.encoding = encoding
        self.remove_stop_words = remove_stop_words
        self.remove_punc = remove_punc

    def __iter__(self):
        for fname in os.listdir(self.dirname):
            doc_content = ""

            with open(os.path.join(self.dirname, fname), encoding=self.encoding) as f:
            #for line in open(os.path.join(self.dirname, fname), encoding=self.encoding):
                for line in f:
                    doc_content += line

            for sentence in sent_tokenize(doc_content):
                original_tokens = list(word_tokenize(sentence.lower()))

                filtered_context = original_tokens

                if self.remove_stop_words:
                    filtered_context = list(stop_words_filter(original_tokens))

                # remove punctuation can improve the model a little bit
                if self.remove_punc:
                    filtered_context = list(punctuation_filter(filtered_context))

                yield filtered_context


def get_top_n_from_dict(dictionary: Dict, top_n: int):
    dict_counter = Counter(dictionary)
    top_n_items = dict_counter.most_common(top_n)
    return top_n_items


def concat(ngram_tuple: Tuple[str, str]):
    ngram_str=""
    for i in range(len(ngram_tuple)):
        ngram_str +=" "+ngram_tuple[i]
    return ngram_str.strip()


def avg_dicts(dict1, dict2):
    """
    merge two dictionaries and avg their values
    :param dict1:
    :param dict2:
    :return:
    """
    from collections import Counter
    sums = Counter()
    counters = Counter()
    for itemset in [dict1, dict2]:
        sums.update(itemset)
        counters.update(itemset.keys())

    return {x: float(sums[x])/counters[x] for x in sums.keys()}


def isSubStringOf(term1, term2):
    """
    check if term1 is a substring of term2 on token-level
    :param term1:
    :param term2:
    :return:
    """
    if len(term1.strip()) == 0 or len(term2.strip()) == 0:
        return False

    term1_token_set = set(term1.split(' '))
    term2_token_set = set(term2.split(' '))

    matched_tokens = set(term1_token_set).intersection(set(term2_token_set))
    if len(matched_tokens) == len(term1_token_set):
        return True
    else:
        return False


def read_by_line(filePath):
    """
    load file content
    return file content in list of lines
    """
    DELIMITER = "\n"
    with open(filePath, encoding="utf-8") as f:
        content = [line.rstrip(DELIMITER) for line in f.readlines()]
    return content


def deprecated(func):
    """This is a decorator which can be used to mark functions
    as deprecated. It will result in a warning being emmitted
    when the function is used."""

    @functools.wraps(func)
    def new_func(*args, **kwargs):
        warnings.simplefilter('always', DeprecationWarning) #turn off filter
        warnings.warn("Call to deprecated function {}.".format(func.__name__), category=DeprecationWarning, stacklevel=2)
        warnings.simplefilter('default', DeprecationWarning) #reset filter
        return func(*args, **kwargs)

    return new_func


def sort_dict_by_value(dictionary : Dict, reverse=True):
    return OrderedDict(sorted(dictionary.items(), key=operator.itemgetter(1), reverse=reverse))


def is_list_of_list(list_of_items: List):
    return any(isinstance(el, list) for el in list_of_items)


def flatten(list_of_items):
    if is_list_of_list(list_of_items):
        return list(itertools.chain(*list_of_items))
    else:
        return list_of_items


def export_list_of_tuple_into_csv(output_path, list_tuple_values, encoding='utf-8', header=None):
    with open(output_path, mode='w', encoding=encoding) as outfile:
        csv_writer=csv.writer(outfile, delimiter=",",lineterminator='\n',quoting=csv.QUOTE_MINIMAL)
        if header:
            csv_writer.writerow(header)
        for tuple in list_tuple_values:
            csv_writer.writerow([tuple[0], tuple[1]])


def export_list_of_tuples_into_json(output_path, list_tuple_values, encoding='utf-8'):
    import json, io
    with io.open(output_path, mode='w', encoding=encoding) as outfile:
        json.dump(dict(list_tuple_values),outfile,ensure_ascii=False)


class NoDaemonProcess(multiprocessing.Process):
    # make 'daemon' attribute always return False
    def _get_daemon(self):
        return False
    def _set_daemon(self, value):
        pass
    daemon = property(_get_daemon, _set_daemon)

# We sub-class multiprocessing.pool.Pool instead of multiprocessing.Pool
# because the latter is only a wrapper function, not a proper class.
class MultiprocPool(multiprocessing.pool.Pool):
    Process = NoDaemonProcess