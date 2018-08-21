#!/usr/bin/env python
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
This is a paralleled and customisable implementation of the TextRank algorithm. 
The original TextRank applies to a single document. This implementation can be directly applied to a large
 corpus.

TextRank algorithm look into the structure of word co-occurrence networks,
where nodes are word types and edges are word cooccurrence.

Important words can be thought of as being endorsed by other words, and this leads to an interesting 
phenomenon. Words that are most important, viz. keywords, emerge as the most central words in the 
resulting network, with high degree and PageRank.

The final important step is post-filtering. Extracted phrases are disambiguated and normalized for 
morpho-syntactic variations and lexical synonymy (Csomai and Mihalcea 2007). Adjacent words are also 
sometimes collapsed into phrases, for a more readable output.


Mihalcea, R., & Tarau, P. (2004, July). TextRank: Bringing order into texts. Association for Computational Linguistics.

"""

import logging
import math
import string
import warnings
from itertools import repeat
from multiprocessing import Pool

import networkx as nx
import numpy as np

from jgtextrank.preprocessing.normaliser import normalize
from jgtextrank.preprocessing.segmentation import pos_tagging, word_2_tokenised_sentences
from jgtextrank.utility import avg_dicts, CorpusContent2RawSentences, get_top_n_from_dict, flatten, is_list_of_list, \
    sort_dict_by_value, isSubStringOf, export_list_of_tuple_into_csv, export_list_of_tuples_into_json

__author__ = 'Jie Gao <j.gao@sheffield.ac.uk>'

__all__ = ["Vertex", "preprocessing", "preprocessing_tokenised_context",
           "_pos_tagging_tokenised_corpus_context", "_syntactic_filter",
           "_syntactic_filter_context", "_is_multiple_context", "_get_cooccurs",
           "_get_cooccurs_from_single_context", "_build_vertices_representations", "_compute_vertex",
           "build_cooccurrence_graph", "_draw_edges", "_is_top_t_vertices_connection",
           "_reweight_filtered_terms", "_term_size_normalize", "_log_normalise", "_gaussian_normalise",
           "_keywords_extraction_from_preprocessed_context",
           "_collapse_adjacent_keywords", "_load_preprocessed_corpus_context",
           "keywords_extraction", "keywords_extraction_from_segmented_corpus",
           "keywords_extraction_from_tagged_corpus",
           "keywords_extraction_from_corpus_directory", "compute_TeRGraph", "compute_neighborhood_size"]

_logger = logging.getLogger("textrank")

# GLOBAL VARIABLES

# Define the maximum number of cpu cores to use
MAX_PROCESSES = 1

# convient functions
pos_tag = pos_tagging()

# default noun and adjective syntactic filter
noun_adjective_filter = lambda t: filter(lambda a: a[1] == 'NNS' or a[1] == 'NNP'
                                                   or a[1] == 'NN' or a[1] == 'JJ', t)


class Vertex(object):
    def __init__(self, word, word_type, tag=None):
        """
        Vertex (or nodes) are word type and edges are word collocations in the structure of word co-occurrence networks.

        the Vertex will represent and be updated for whole corpus

        :param word is the original surface form of the node
        :param word_type is the normalised form of the node
        :param tag can be any tag assigned to the node (typically PoS category)
        """
        self.word = word
        self.word_type = word_type
        self.tag = tag
        self.score = -1
        # add co-occurred vertex with word surface form
        # todo: may add auto-generated vertex ID to refere to a more rich representation of vertex
        self.co_occurs = []
        # trace back to the context list which current vertex pertains to
        self.context = []

    def __str__(self):
        # + "' tag: '" + self.tag + "'"
        return "vertex word: '" + self.word + "', word type (normed): '" \
               + self.word_type + "'" + ", co-occur vertices: " + str(self.co_occurs)

    def __repr__(self):
        return str(self)


def preprocessing(text, syntactic_categories={"NNS", "NNP", "NN", "JJ"},
                  stop_words=None, lemma=False):
    """
    pre-processing pipeline: sentence splitting -> tokenisation ->
    Part-of-Speech(PoS) tagging -> syntactic filtering (default with sentential context)

    Text segmentation: using NLTK's recommended English word tokenizer (currently an improved :class:`.TreebankWordTokenizer`
    along with :class:`.PunktSentenceTokenizer`

    PoS tagging: Use NLTK's currently recommended part of speech tagger ('taggers/averaged_perceptron_tagger/english.pickle')

    You can download both via

            >>> import nltk
            >>> nltk.download('punkt')
            >>> nltk.download('averaged_perceptron_tagger')

    :type text: string
    :param text: plain text
    :type syntactic_categories: set [of string], required
    :param syntactic_categories: Default with noun and adjective categories.
                    Syntactic categories (default as Part-Of-Speech(PoS) tags) is defined to
                    filter accepted graph vertices (default with word-based tokens as single syntactic unit).

                    Any word that is not matched with the predefined categories will be removed based on corresponding the PoS tag.

                    Best result is found with noun and adjective categories only in original paper.
    :type stop_words: set of [string {‘english’}], or None (default), Optional
    :param stop_words:  remove stopwords from PoS tagged context (token tuple list).
                The stop words are considered as noisy common/function words.
                By provide a list of stop words can improve vertices network connectivity
                and increase weights to more meaningful words.
    :rtype: generatorType (of tuple)
    :return: result: a tuple list of tokenised context(default in sentence level) text
                            and the corresponding PoS tagged context text filtered by syntactic filter
    """

    tokenised_sentences = word_2_tokenised_sentences(text)

    return preprocessing_tokenised_context(tokenised_sentences, syntactic_categories=syntactic_categories,
                                           stop_words=stop_words, lemma=lemma)


def preprocessing_tokenised_context(tokenised_context, syntactic_categories={"NNS", "NNP", "NN", "JJ"},
                                    stop_words=None, lemma=False):
    """
    pre-processing tokenised corpus context (recommend as sentences)

    pipeline:  Part-of-Speech tagging -> syntactic filtering (default with sentential context)

    :type tokenised_context: generator or iterable object
    :param tokenised_context: generator of tokenised context(default with sentences)
    :type syntactic_categories: set [of string], required
    :param syntactic_categories: Default with noun and adjective categories.
                    Syntactic categories (default as Part-Of-Speech(PoS) tags) are defined to
                    filter accepted graph vertices (default with word-based tokens as single syntactic unit).

                    Any word that is not matched with the predefined categories will be removed based on corresponding the PoS tag.

                    Best result is found with noun and adjective categories only in original paper.
    :type stop_words: set of [string {‘english’}], or None (default), Optional
    :param stop_words:  remove stopwords from PoS tagged context (token tuple list).
                The stop words are considered as noisy common/function words.
                By provide a list of stop words can improve vertices network connectivity
                and increase weights to more meaningful words.
    :rtype: generator[of tuple]
    :return: pre-processed raw text tokens splitted with context and filtered text tokens splitted with context
    """
    _check_required_values(False, syntactic_categories)

    pos_filter = lambda t: filter(lambda a: a[1] in syntactic_categories, t)
    stop_words_filter = None if stop_words is None else lambda t: filter(lambda a: a[0] not in stop_words, t)

    tagged_tokenised_context_collection = _pos_tagging_tokenised_corpus_context(tokenised_context, lemma=lemma)

    for tokenised_context, pos_tagged_context in tagged_tokenised_context_collection:
        yield tokenised_context, _syntactic_filter_context(pos_tagged_context, pos_filter=pos_filter,
                                                           stop_words_filter=stop_words_filter)


def _pos_tagging_tokenised_corpus_context(tokenised_corpus_context, lemma=False):
    """

    :type tokenised_corpus_context: generator or iterable object
    :param tokenised_corpus_context: generator of tokenised context(default as sentences)
    :rtype: generator [of tuple]
    :return: tuple<tokenised sentences, pos tagged tokenised sentences>
    """
    for tokenised_sentence in tokenised_corpus_context:
        sentence_token_list = list(tokenised_sentence)
        tagged_sentence_token_list = pos_tag(sentence_token_list)

        sentence_token_list, normed_tagged_sentence_token_list = _normalise_tagged_token_list(
            tagged_sentence_token_list, lemma=lemma)

        yield (sentence_token_list, normed_tagged_sentence_token_list)


def _normalise_tagged_token_list(tagged_token_list, lemma=False):
    """
    :type: list [of tuple [string, string]]
    :param tagged_token_list: tagged and tokenized text list
    :rtype: tuple
    :return: normalised token list and normalised tagged token list
    """

    normed_tagged_token_list = [(normalize(tagged_token[0], lemma=lemma, pos_tag=tagged_token[1]), tagged_token[1]) for
                                tagged_token in tagged_token_list]
    normed_token_list = [norm_token[0] for norm_token in normed_tagged_token_list]
    return normed_token_list, normed_tagged_token_list


def _syntactic_filter(pos_tagged_tokenised_sents, pos_filter=None, stop_words_filter=None):
    """
    all lexical units that pass the syntactic filter will be added to the graph

    Best result is observed for nouns and adjective only in the paper

    :type pos_tagged_tokenised_sents: list
    :param pos_tagged_tokenised_sents: example input, [[(token, pos_tag), ...], [(token, pos_tag), ...], ...]
    :param pos_filter: filter function to remove unwanted tokens from PoS tagged token tuple list based on corresponding PoS tag
    :param stop_words_filter: filter function to remove stopwords from PoS tagged token tuple list
    :rtype: generator [of list [of tuple]]
    :return: filtered context taggged with pos categories
    """
    _logger.info("syntactic filtering ...")
    # print("pos_tagged_tokenised_sents: ", list(pos_tagged_tokenised_sents))
    if not is_list_of_list(pos_tagged_tokenised_sents):
        raise Exception("Incorrected format. Expect: a list of textual context token list. "
                        "Example: [[(token, pos_tag), ...], [(token, pos_tag), ...], ...]")

    for pos_tagged_tokenised_sent in pos_tagged_tokenised_sents:
        # print("pos_tagged_tokenised_sent: ", pos_tagged_tokenised_sent)
        yield _syntactic_filter_context(pos_tagged_tokenised_sent, pos_filter, stop_words_filter)


def _syntactic_filter_context(pos_tagged_tokens, pos_filter=None, stop_words_filter=None,
                              punc_filter=lambda t: filter(lambda a: a[0] not in string.punctuation, t)):
    """
    syntactic filtering for single context(default as single PoS tagged tokenised sentence)

    all lexical units that pass the syntactic filter will be added to the graph

    Best result is observed for nouns and adjective only in the paper

    :type pos_tagged_tokens: tuple list
    :param pos_tagged_tokens: PoS tagged tokens, e.g., [(token, pos_tag), ...]
    :param pos_filter: PoS filter function to define acceptable syntactic context, default with noun and adjective words
    :param stop_words_filter: filter function to remove stopwords from PoS tagged token tuple list
    :param punc_filter: filter function to remove punctuation from the syntactic context in the case of mistagged PoS category
    :return: filtered context taggged with pos categories
    """

    if pos_filter is None:
        pos_filter = noun_adjective_filter

    filtered_context = pos_filter(pos_tagged_tokens)

    if stop_words_filter:
        pos_filtered_context = list(filtered_context)
        # print("pos_filtered_context: ", pos_filtered_context)
        filtered_context = stop_words_filter(pos_filtered_context)

    final_filtered_context = list(punc_filter(filtered_context))
    # print("final_filtered_context: ", final_filtered_context)
    return final_filtered_context


def _is_multiple_context(corpus_context):
    return is_list_of_list(corpus_context)


def _get_cooccurs(syntactic_unit, vertices_cooccur_context_corpus, all_filtered_context_tokens=None, window_size=2):
    """
    get word co-occurrence from filtered context

    :param syntactic_unit: word (i.e., vertex) surface form
    :param vertices_cooccur_context_corpus: a list of tokens representing every single context of corpus
                where word cooccur can be computed from
    :param window_size: default with 2 for forward context and backward context
    :rtype: set
    :return: all co-occurrences of the input
    """

    all_cooccurs = []

    if _is_multiple_context(vertices_cooccur_context_corpus):
        for filtered_corpus_context in vertices_cooccur_context_corpus:
            all_cooccurs.append(_get_cooccurs_from_single_context(syntactic_unit, filtered_corpus_context, window_size))
    else:
        all_cooccurs = _get_cooccurs_from_single_context(syntactic_unit, vertices_cooccur_context_corpus, window_size)

    flatten_all_cooccurs = flatten(all_cooccurs)

    # remove co-occur words that are not accepted by syntactic filter
    if all_filtered_context_tokens:
        cooccur_filter = lambda t: filter(lambda a: a in all_filtered_context_tokens, t)
        flatten_all_cooccurs = cooccur_filter(flatten_all_cooccurs)

    return set(flatten_all_cooccurs)


def _get_cooccurs_from_single_context(syntactic_unit, tokenised_context, window_size=2):
    """
    get co-occurred syntactic units within specific context window by the given unit

    This implementation is default with forward and backward context.

    :type syntactic_unit: string
    :param syntactic_unit: syntactic unit(e.g., token)
    :type tokenised_context: list [of string]
    :param tokenised_context: tokensed context with a list of tokenised syntactic units
    :type window_size: int
    :param window_size: context forward and backward window size that is used to compute co-occurrences
    :rtype: list [of string]
    :return: co-occurrences of given syntactic unit
    """
    if syntactic_unit not in tokenised_context:
        return []

    # current_index = tokenised_context.index(syntactic_unit)
    all_indices = [i for i, x in enumerate(tokenised_context) if x == syntactic_unit]

    candidate_cooccurs = []
    for current_index in all_indices:
        context_size = len(tokenised_context)

        forward_context = [(current_index - forward - 1) for forward in range(window_size) if
                           (current_index - forward - 1) >= 0 and (current_index - forward - 1) <= context_size - 1
                           and (current_index - forward - 1) < current_index]

        backward_context = [(current_index + backward + 1) for backward in range(window_size) if
                            (current_index + backward + 1) >= 0 and (current_index + backward + 1) <= context_size - 1
                            and (current_index + backward + 1) > current_index]

        # print(syntactic_unit, ", tokenised_context: ", tokenised_context)
        # print("[%s] backward context: %s" %(syntactic_unit, [tokenised_context[context_index] for context_index in backward_context]))

        cooccur_context = forward_context + backward_context

        candidate_cooccurs.append([tokenised_context[cooccur_word_index] for cooccur_word_index in cooccur_context])

    return flatten(candidate_cooccurs)


def _build_vertices_representations(all_tokenised_filtered_context, all_tokenised_context=None,
                                    conn_with_original_ctx=True, window_size=2):
    """
    build vertices representations for graph network

    :type all_tokenised_filtered_context: list [of string]
    :param all_tokenised_filtered_context: tokenised context text filtered by PoS based syntactic filter for co-occurrence
            Note: the input is expected to be pre-processed
    :type conn_with_original_ctx: bool
    :param conn_with_original_ctx: True if checking two vertices co-occurrence link from original context
                                else checking connections from filtered context
    :type window_size: int
    :param window_size: a window of N words -> TODO: forward cooccurence and backward coocurrence can be implemented
    :rtype: list [of Vertex]
    :return: list of Vertices
    """
    _logger.info("computing vertices representations...")
    vertices = list()

    # the filtered tokens are the basic unit of vertices
    all_filtered_context_tokens = set([context_token for context_token in flatten(all_tokenised_filtered_context)])

    if conn_with_original_ctx:
        vertices_coccur_context = all_tokenised_context
    else:
        vertices_coccur_context = all_tokenised_filtered_context

    # print("_build_vertices_representations >> all_tokenised_filtered_context: ", all_tokenised_filtered_context)

    global MAX_PROCESSES
    with Pool(processes=MAX_PROCESSES) as pool:
        vertices = pool.starmap(_compute_vertex, zip(all_filtered_context_tokens, repeat(vertices_coccur_context),
                                                     repeat(all_filtered_context_tokens), repeat(window_size)))

    _logger.info("total size of vertices: %s", len(vertices))
    # print("all vertices: ", vertices)
    return vertices


def _compute_vertex(syntactic_unit, vertices_cooccur_context_corpus, all_filtered_context_tokens=None, window_size=2):
    """
    :type syntactic_unit: String
    :param syntactic_unit: syntactic filtered (selected) token unit
    :param vertices_cooccur_context_corpus: a list of tokens representing every single context
                where word cooccur can be computed from
    :param syntactic_filtered_context: a list of tokens representing every single context of corpus
                where word cooccur can be computed from
    :rtype: Vertex
    :return: Vertex with co-occurrences
    """
    # Tips: search the following printout in either console or log is a simple way to check current progress
    _logger.info("compute vertex in a single thread ...")
    # vertex = Vertex(syntactic_unit[0], normalize(syntactic_unit[0]), syntactic_unit[1])
    # word and word_type difference are not really useful and implemented from the efficiency consideration
    # if lemmatization is choosed, the syntactic unit and context are expected to be pre-normalised and lemmatised
    vertex = Vertex(syntactic_unit, syntactic_unit)

    # print("filtered_context_list in _compute_vertex : ", filtered_context_list)
    cooccured_syntactic_units = _get_cooccurs(syntactic_unit, vertices_cooccur_context_corpus,
                                              all_filtered_context_tokens, window_size)
    # print(syntactic_unit, ", cooccured_syntactic_units: ", cooccured_syntactic_units)
    vertex.co_occurs = [syntactic_unit for syntactic_unit in cooccured_syntactic_units]

    return vertex


def build_cooccurrence_graph(preprocessed_context, directed=False, weighted=False,
                             conn_with_original_ctx=True, window=2):
    """
    build cooccurrence graph from filtered context
    and only consider single words as candidates for addition to the graph

    prepare vertex representation -> add vertex > add edges

    For directed or undirected, the conclusion of the paper is that "no 'direction' that can be established between
    co-occurring words."

    :type preprocessed_context: generator or list/iterable
    :param preprocessed_context: a tuple list of tokenised and PoS tagged text filtered by syntactic filter
    :type directed: bool
    :type weighted: bool. Not supported yet
    :param directed: default as False, best results observed with undirected graph;
            :TODO: for directed graph, not fully supported yet and need to define forward co-occurrence and backward co-occurrence
                For directed graph, a direction should be set following the natural flow of the text
    :type conn_with_original_ctx: bool
    :param conn_with_original_ctx: True if checking two vertices co-occurrence link from original context
                                else checking connections from filtered context
            More vertices connection can be built if 'conn_with_original_ctx' is set to False
    :type window_size: int
    :param window_size: a window of N words
    :rtype: tuple[of [nx.graph, list]]
    :return: (networkx) graph object readily to score along with all tokenised raw text splitted by context

    """
    all_tokenised_context = []
    all_tokenised_filtered_context = []
    for tokenised_context, context_syntactic_units in preprocessed_context:
        all_tokenised_context.append(tokenised_context)
        # PoS tags are removed
        all_tokenised_filtered_context.append(
            [context_syntactic_unit[0] for context_syntactic_unit in context_syntactic_units])

    vertices = _build_vertices_representations(all_tokenised_filtered_context,
                                               all_tokenised_context=all_tokenised_context,
                                               conn_with_original_ctx=conn_with_original_ctx, window_size=window)

    cooccurence_graph = nx.Graph()

    cooccurence_graph.add_nodes_from([vertex.word_type for vertex in vertices], weight=1.0)
    cooccurence_graph.add_weighted_edges_from(_draw_edges(vertices))

    if directed:
        return cooccurence_graph.to_directed(), all_tokenised_context
    else:
        return cooccurence_graph.to_undirected(), all_tokenised_context


def _draw_edges(vertices, weight=1.0):
    """
    draw edges to make connections between co-occurred word types (i.e., normalised word surface form)
    the co-occur edge weight is default to 1.0

    see also <link href="http://stackoverflow.com/questions/9136539/how-do-weighted-edges-affect-pagerank-in-networkx"/>

    :param vertices: vertices to be loaded
    :return: edges tuple list
    """
    # print([(vertex.word_type,vertex.co_occurs) for vertex in vertices])

    edges = []
    for vertex in vertices:
        for co_occur in vertex.co_occurs:
            edges.append((vertex.word_type, co_occur, weight))
    return edges


def _is_top_t_vertices_connection(collapsed_term, top_t_vertices):
    """

    :type collapsed_term: list [of list [of string]]
    :param collapsed_term: list of tokenised terms collapsed from original context that will form Single-word term or Multi-word Term
    :param top_t_vertices: top T weighted vertices
    :return: True if the input contains any of top T vertex
    """
    return any(top_t_vertex[0] in collapsed_term for top_t_vertex in top_t_vertices)


def _reweight_filtered_terms(collapsed_terms, top_t_vertices, all_vertices, weight_comb="norm_max", mu=5):
    """
    weight key terms with page rank weights of vertices

    get max value of syntactic units for multi-word terms and penalise repeated vertice with maximum value if any

    repeated vertex (i.e., syntactic unit or single word) will be normalised by occurrence

    :type collapsed_terms: list [of list [of string]]
    :param collapsed_terms: collection of tokenised Single-Word or Multi-Word candidate terms
    :type top_t_vertices: list [of tuple]
    :param top_t_vertices: weighted top T vertices
    :type all_vertices: list [of tuple]
    :param all_vertices: all the weighted top T vertices
    :type weight_comb: str
    :param weight_comb:  {'avg', 'norm_avg', 'log_norm_avg', 'gaussian_norm_avg', 'sum', 'norm_sum', 'log_norm_sum',
                'gaussian_norm_sum', 'max', 'norm_max', 'log_norm_max', 'gaussian_norm_max',
                'len_log_norm_max', 'len_log_norm_avg', 'len_log_norm_sum'}, default 'norm_max'
            The weight combination method for multi-word candidate terms weighing.

            - 'max' : maximum value of vertices weights
            - 'avg' : avarage vertices weight
            - 'sum' : sum of vertices weights
            - 'norm_max' : MWT unit size normalisation of 'max' weight
            - 'norm_avg' : MWT unit size normalisation of 'avg' weight
            - 'norm_sum' : MWT unit size normalisation of 'sum' weight
            - 'log_norm_max' : logarithm based normalisation of 'max' weight
            - 'log_norm_avg' : logarithm based normalisation of 'avg' weight
            - 'log_norm_sum' : logarithm based normalisation of 'sum' weight
            - 'gaussian_norm_max' : gaussian normalisation of 'max' weight
            - 'gaussian_norm_avg' : gaussian normalisation of 'avg' weight
            - 'gaussian_norm_sum' : gaussian normalisation of 'sum' weight
            - 'len_log_norm_max': log2(|a| + 0.1) * 'max' adapted from CValue (Frantzi, 2000) formulate
            - 'len_log_norm_avg': log2(|a| + 0.1) * 'avg' adapted from CValue (Frantzi, 2000) formulate
            - 'len_log_norm_sum': log2(|a| + 0.1) * 'sum' adapted from CValue (Frantzi, 2000) formulate

            NOTE: \*_norm_\*" penalises/smooth the longer term (than default 5 token size)
                to achieve a saturation level as term size grows
    :type mu: int, optional
    :param mu: mean value to set a center point (default to 5) in order to rank the candidates higher that are near the central point
            This param is only required for normalisation based MWT weighting method
    :rtype: dict [of term:weight]
    :return: dict with key as term string and value is the weight
    """
    _logger.info("MWTs weighing ...")
    weighted_terms = dict()
    log2a = 0
    sigma = 0
    if "norm" in weight_comb:
        sigma = _get_sigma_from_all_candidates(collapsed_terms)
        # print("sigma is ", sigma)

    for collapsed_term in collapsed_terms:
        if _is_top_t_vertices_connection(collapsed_term, top_t_vertices):
            # initialisation
            final_score = float(0)
            avg_score = 0
            sum_score = 0
            max_score = 0

            # compute term length (i.e.,number of words/tokens)
            # all_syntactic_units = collapsed_term.split(' ')
            all_syntactic_units = collapsed_term
            unit_size = len(all_syntactic_units)

            if "len_log" in weight_comb:
                # log(a + 0.1) to smooth unigrams
                log2a = math.log2(unit_size + 0.1)

            if "avg" in weight_comb:
                avg_score = _get_average_score(all_syntactic_units, all_vertices, unit_size)

            if "sum" in weight_comb:
                sum_score = _get_sum_score(all_syntactic_units, all_vertices)

            if "max" in weight_comb:
                max_score = _get_max_score(all_syntactic_units, all_vertices)

            if weight_comb == "avg":
                final_score = avg_score
            elif weight_comb == "norm_avg":
                final_score = _term_size_normalize(avg_score, unit_size)
            elif weight_comb == "log_norm_avg":
                final_score = _log_normalise(avg_score, mu, unit_size)
            elif weight_comb == "gaussian_norm_avg":
                final_score = _gaussian_normalise(avg_score, mu, sigma, unit_size)
            elif weight_comb == "len_log_norm_avg":
                final_score = log2a * avg_score
            elif weight_comb == "sum":
                final_score = sum_score
            elif weight_comb == "norm_sum":
                final_score = _term_size_normalize(sum_score, unit_size)
            elif weight_comb == "log_norm_sum":
                final_score = _log_normalise(sum_score, mu, unit_size)
            elif weight_comb == "gaussian_norm_sum":
                final_score = _gaussian_normalise(sum_score, mu, sigma, unit_size)
            elif weight_comb == "len_log_norm_sum":
                final_score = log2a * sum_score
            elif weight_comb == "max":
                final_score = max_score
            elif weight_comb == "norm_max":
                final_score = _term_size_normalize(max_score, unit_size)
            elif weight_comb == "log_norm_max":
                final_score = _log_normalise(max_score, mu, unit_size)
            elif weight_comb == "gaussian_norm_max":
                final_score = _gaussian_normalise(max_score, mu, sigma, unit_size)
            elif weight_comb == "len_log_norm_max":
                final_score = log2a * max_score
            else:
                raise ValueError("Unsupported weight combination option: '%s'", weight_comb)

            concatenated_collapsed_term = " ".join(collapsed_term)
            weighted_terms[concatenated_collapsed_term] = round(final_score, 5)

    _logger.info("done.")
    return weighted_terms


def _gaussian_normalise(base_score, mu, sigma, unit_size):
    norm_value = 1 - _probability_density(unit_size, mu, sigma)
    base_score = base_score * float(norm_value)
    return base_score


def _log_normalise(base_score, mu, unit_size):
    if unit_size > 1:
        # print("_log_normalise with mu=", mu, " , unit_size:", unit_size)
        base_score = base_score / math.log(unit_size, mu)
    return base_score


def _term_size_normalize(base_score, unit_size):
    return base_score / float(unit_size)


def _get_sigma_from_all_candidates(collapsed_terms):
    """
    compute standard deviation of term length in MWTs
    :param collapsed_terms: list, list of tokenised terms
    :rtype: ndarray
    :return: standard_deviation
    """
    all_terms_size = [len(collapsed_term) for collapsed_term in collapsed_terms]
    return np.std(all_terms_size)


def _probability_density(x_value, mu, sigma):
    """
     probability density of the normal distribution

     see also https://en.wikipedia.org/wiki/Normal_distribution
    :param x_value:
    :param mu:
    :param sigma:
    :return:
    """
    pd = (1 / (sigma * np.sqrt(2 * math.pi))) * math.exp(- math.pow((x_value - mu), 2) / (2 * math.pow(sigma, 2)))
    return pd


def _get_plus_score(all_syntactic_units, boosted_term_size_range, boosted_word_length_range, combined_weight,
                    unit_size):
    """
    Experimental weighting method to provide extra small fraction weight to the final score

    More weight can be given to longer term

    :type all_syntactic_units: list (of str)
    :param all_syntactic_units: all the tokens of a candidate term(SWT or MWT)
    :type boosted_term_size_range: (int, int) | None
    :param boosted_term_size_range: range of token size of a candidate term that will be boosted with a small weight fraction
    :type boosted_word_length_range: (int, int) | None
    :param boosted_word_length_range: range of word length (number of character) that will be boosted with a small weight fraction
    :type combined_weight: float
    :param combined_weight: combined the weight (i.e., 'avg' or 'max') of current candidate term
            This weight is important and used as base value for final boosted weight
    :type unit_size: int
    :param unit_size: token size of current candidate term
    :return: a small weight fraction that can be added to the final weight
    """
    all_syntactic_units_lengths = [len(term_unit) for term_unit in all_syntactic_units]
    min_word_length = min(all_syntactic_units_lengths)
    max_word_length = max(all_syntactic_units_lengths)
    avg_word_length = sum(all_syntactic_units_lengths) / unit_size
    plus_weight = combined_weight
    if boosted_word_length_range is not None and boosted_term_size_range is not None \
            and unit_size in boosted_term_size_range and min_word_length in boosted_word_length_range \
            and max_word_length in boosted_word_length_range:
        # add a small fraction to the final weight when all the syntactic unit length in in a normal range
        plus_weight = combined_weight * math.log(avg_word_length, 2)
    elif boosted_word_length_range is None and boosted_term_size_range is not None and unit_size in boosted_term_size_range:
        plus_weight = combined_weight * math.log(avg_word_length, 2)
    elif boosted_word_length_range is not None and boosted_term_size_range is None and \
            min_word_length in boosted_word_length_range and max_word_length in boosted_word_length_range:
        plus_weight = combined_weight * math.log(avg_word_length, 2)

    return plus_weight


def _get_max_score(all_syntactic_units, all_vertices):
    """
    get max term unit score (normalised by term unit frequency in MWTs)
    :param all_syntactic_units:
    :param all_vertices:
    :return:
    """
    # print("all_vertices: ", all_vertices)
    # print("collapsed_term: ", collapsed_term)
    # max_score = max([all_vertices[term_unit] / float(all_syntactic_units.count(term_unit)) for term_unit in collapsed_term.split(' ')])
    max_score = max([all_vertices[term_unit] / float(all_syntactic_units.count(term_unit)) for term_unit in all_syntactic_units])
    return max_score


def _get_average_score(all_syntactic_units, all_vertices, unit_size):
    """
    get average score from single candidate term

    :param all_syntactic_units: tokens of single candidate term
    :param all_vertices: all the vertices used for computing combined weight
    :param unit_size: size of multi-word candidate term
    :return:
    """
    avg_score = _get_sum_score(all_syntactic_units, all_vertices) / float(unit_size)
    return avg_score


def _get_sum_score(all_syntactic_units, all_vertices):
    return sum(
        [all_vertices[term_unit] / float(all_syntactic_units.count(term_unit)) for term_unit in all_syntactic_units])


def _weight_nodes_with_centrality_metrics(scoring_method, cooccurrence_graph):
    """
    Centrality measures (such as "current flow betweeness", "current flow closeness", "communicability_betweenness")
        does not support loosely connected graph and betweeness centrality measures cannot compute on single isolated nodes.

    This method is to adapt those centrality algorithms to run on loosely connected graph by iterating over connected subgraphs and ignore isolated nodes.
    The weighted nodes in every separated subgraph will be combined.

    :param scoring_method: centrality measurement method
    :param cooccurrence_graph:
    :return: dict, weighted nodes combined from subgraphs if graph is not (strongly) connected
    """
    if nx.is_connected(cooccurrence_graph):
        weighted_nodes = scoring_method(cooccurrence_graph)
    else:
        warnings.warn("Graph is not (strongly) connected. Nodes will be measured in connected subgraphs.")
        weighted_nodes = {}
        # iteratively score connected sub-graph
        for c in nx.connected_components(cooccurrence_graph):
            connected_graph = cooccurrence_graph.subgraph(c)
            try:
                weighted_nodes.update(scoring_method(connected_graph))
            except ZeroDivisionError:
                # ignore the error caused by isolated nodes
                continue

    # remove isolated nodes which weights will be inf value
    weighted_nodes = {k: v for k, v in weighted_nodes.items() if v != float("inf")}
    return weighted_nodes


def compute_TeRGraph(term_graph):
    """
    compute graph vertices with TeRGraph algorithms

    This algorithm is based on the assumption that term representativeness in a graph for a specific domain depends on
    the number of neighbors that it has, and the number of neighbors of its neighbors. A term with more neighbors is
    less representative of the specific domain.

    Original paper requires a connected graph and this method will set isolated nodes to 0 (by default).

    Lossio-Ventura, J. A., Jonquet, C., Roche, M., & Teisseire, M. (2014, September).
        Yet another ranking function for automatic multiword term extraction.
        In International Conference on Natural Language Processing (pp. 52-64). Springer, Cham.

    :param term_graph: NetworkX graph
    :return: dict, all nodes weighted with TeRGraph metric
    """
    all_nodes = term_graph.nodes()
    node_weights = dict()

    _logger.debug("total nodes: %s", len(all_nodes))

    total_isolated_nodes = 0
    for node in all_nodes:
        list_of_neighbors = list(term_graph.neighbors(node))
        # num_of_neighbors
        n_a = len(list_of_neighbors)
        if n_a == 0:
            total_isolated_nodes += 1

        # print("---> list_of_neighbors: ", list_of_neighbors)
        n_t_i = 0 if n_a == 0 else sum(
            [len(list(term_graph.neighbors(neighbor_node))) for neighbor_node in list_of_neighbors])
        # print("total number of neighbors of neighbors: ", n_t_i)
        # 0.5 is the smooth value to avoid zero division when it happens with isolated nodes
        node_weight = 0 if n_a == 0 else math.log2(1.5 + 1 / (n_a + n_t_i))
        node_weights[node] = node_weight

    _logger.debug("total isolated nodes: %s", total_isolated_nodes)
    _logger.debug("total weighted nodes: %s", len(node_weights))
    return node_weights


def compute_neighborhood_size(term_cooccur_graph):
    """
    Number of immediate neighbors to a node

    a version of node degree that disregards self-loops (e.g., "again, again, again")

    :param term_graph: NetworkX graph
    :return: dict, all nodes weighted with neighborhood size
    """
    node_weights = dict()
    num_of_selfloops = nx.number_of_selfloops(term_cooccur_graph)
    if num_of_selfloops > 0:
        _logger.warning("remove %s selfloops.", num_of_selfloops)
        term_cooccur_graph.remove_edges_from(nx.selfloop_edges(term_cooccur_graph))

    all_nodes = term_cooccur_graph.nodes()
    for node in all_nodes:
        neighbors = nx.all_neighbors(term_cooccur_graph, node)
        node_weights[node] = len(list(neighbors))

    return node_weights


def _keywords_extraction_from_preprocessed_context(preprocessed_corpus_context,
                                                   top_p=0.3, top_t=None, window=2, directed=False,
                                                   weighted=False, conn_with_original_ctx=True,
                                                   max_iter=100, tol=1.0e-6, solver="pagerank",
                                                   weight_comb="norm_max", mu=5):
    """
    :type preprocessed_corpus_context: generator or list/iterable
    :param preprocessed_corpus_context: a tuple list of tokenised context text
                            and the corresponding PoS tagged context text filtered by syntactic filter
    :type top_p: float, optional
    :param top_p: the top Percentage of vertices are retained for post-processing, default as 1/3 of all vertices
    :type top_t: int, optional
    :param top_t: the top T vertices in the ranking are retained for post-processing
    :type window: int, optional
    :param window: co-occurrence window size (default with forward and backward context)
    :type directed: bool, optional
    :param directed: directed or undirected graph (a preserved parameters)
    :type weighted: bool, optional
    :param weighted: default as unweighted graph, Custom weighted graph is not supported yet, Default as False
            Best result is found with unweighted graph in the original paper
    :type conn_with_original_ctx: bool, optional
    :param conn_with_original_ctx: True if checking two vertices co-occurrence link from original context
                                else checking connections from filtered context
            More vertices connection can be built if 'conn_with_original_ctx' is set to False
    :type max_iter: int, optional
    :param max_iter: number of maximum iteration of pagerank, katz_centrality
                Note: number of iteration and error tolerance can affect the performance and top N precision of the ranking
    :type tol: float, optional, default 1.0e-6
    :param tol: Error tolerance used to check convergence, the value varies for specific solver
    :type solver: string, optional
    :param solver: {'pagerank', 'pagerank_numpy', 'pagerank_scipy', 'betweenness_centrality', 'degree_centrality',
                    'hits', 'closeness_centrality', 'edge_betweenness_centrality', 'eigenvector_centrality',
                    'katz_centrality', 'communicability_betweenness', 'current_flow_closeness', 'current_flow_betweenness',
                    'edge_current_flow_betweenness', 'load_centrality', 'clustering_coefficient',
                    'TeRGraph','coreness', 'neighborhood_size'}, default 'pagerank'
        PageRank Algorithms supported in networkx to use in the vertices ranking.

        - 'betweenness_centrality' computes the shortest-path betweenness centrality of a node
        - 'degree_centrality' computes the degree centrality for nodes.
        - 'hits' computes HITS algorithm for a node. The Avg(Authority, Hub) is computed
        - 'closeness_centrality' computes closeness centrality for nodes.
        - 'edge_betweenness_centrality' computes betweenness centrality for edges.
                                Maximum edge betweenness value in all the possible edge pairs is adopted for each vertex
        - 'eigenvector_centrality' computes the eigenvector centrality for the cooocurrence graph.
        - 'katz_centrality' computes the Katz centrality for the nodes based on the centrality of its neighbors.
        - 'communicability_betweenness' computes subgraph communicability for all pairs of nodes
        - 'current_flow_closeness' computes current-flow closeness centrality for nodes.
        - 'current_flow_betweenness' computes current-flow betweenness centrality for nodes.
        - 'edge_current_flow_betweenness' computes current-flow betweenness centrality for edges.
        - 'load_centrality' computes edge load. This is a experimental algorithm in nextworkx
                                    that counts the number of shortest paths which cross each edge.
        - 'clustering_coefficient' computes the clustering coefficient for nodes. Only undirected graph is supported.
        - 'TeRGraph' (Lossio-Ventura, 2014) computes the TeRGraph weights for nodes.
                        The solver requires a connected graph and isolated nodes will be set to 0.
        - 'coreness' (Batagelj & Zaversnik, 2003) measures how "deep" a node(word/phrase) is in the co-occurrence network.
                This indicates how strongly the node is connected to the network. The "deeper" a word, the more it is important.
                The metric is not suitable for ranking terms directly, but it is proved as useful feature for keywords extraction.
                Note: self-loops edges (e.g.,"again, again and again") will be removed.
        - 'neighborhood_size' computes the number of immediate neighbors to a node.
                    This is a version of node degree that disregards self-loops

        Note: Centrality measures (such as "current flow betweeness", "current flow closeness", "communicability_betweenness")
            does not support loosely connected graph and betweeness centrality measures cannot compute on single isolated nodes.
             It is recommended to re-consider the graph construction method or increase context window size to
             ensure a (strongly) connected graph.

    :type weight_comb: str, optional
    :param weight_comb: weight combination method for multi-word candidate terms.
                Options: avg, norm_avg, log_norm_avg, gaussian_norm_avg, sum, norm_sum, log_norm_sum,
                gaussian_norm_sum, max, norm_max, log_norm_max, gaussian_norm_max

            '\*_norm_\*' penalises longer term (than default 5 token size)
    :type mu: int, optional
    :param mu: mean value to set a center point (default to 5) in order to rank the candidates higher that are near the central point
            This param is only required and effective for normalisation based MWT weighting method
    :rtype: tuple (dict,list)
    :return: weighted terms, top vertices
    """
    cooccurrence_graph, original_tokenised_context = build_cooccurrence_graph(preprocessed_corpus_context,
                                                                              directed=directed,
                                                                              weighted=weighted,
                                                                              conn_with_original_ctx=conn_with_original_ctx,
                                                                              window=window)

    if solver == "pagerank":
        weighted_nodes = nx.pagerank(cooccurrence_graph, weight='weight', max_iter=max_iter, tol=tol)
    elif solver == "pagerank_numpy":
        weighted_nodes = nx.pagerank_numpy(cooccurrence_graph)
    elif solver == "pagerank_scipy":
        weighted_nodes = nx.pagerank_scipy(cooccurrence_graph, max_iter=max_iter, tol=tol)
    elif solver == "betweenness_centrality":
        weighted_nodes = nx.betweenness_centrality(cooccurrence_graph)
    elif solver == "edge_betweenness_centrality":
        weighted_nodes = nx.edge_betweenness_centrality(cooccurrence_graph)
        weighted_nodes = _flatten_nodes_betweenness_weights(weighted_nodes)
    elif solver == "degree_centrality":
        weighted_nodes = nx.degree_centrality(cooccurrence_graph)
    elif solver == "closeness_centrality":
        weighted_nodes = nx.closeness_centrality(cooccurrence_graph)
    elif solver == "hits":
        # importance of a vertex as a hub and as an authority
        # Avg(Authority, Hub) is computed
        hits_values = nx.hits(cooccurrence_graph, max_iter=max_iter, tol=tol)
        hub_nodes = hits_values[0]
        authorities_nodes = hits_values[1]
        weighted_nodes = avg_dicts(hub_nodes, authorities_nodes)
    elif solver == "eigenvector_centrality":
        weighted_nodes = nx.eigenvector_centrality(cooccurrence_graph, max_iter=max_iter, tol=tol)
    elif solver == "katz_centrality":
        #  max_iter=max_iter, tol=tol
        weighted_nodes = nx.katz_centrality_numpy(cooccurrence_graph)
    elif solver == "communicability_betweenness":
        weighted_nodes = _weight_nodes_with_centrality_metrics(nx.communicability_betweenness_centrality,
                                                               cooccurrence_graph)
    elif solver == "current_flow_closeness":
        weighted_nodes = _weight_nodes_with_centrality_metrics(nx.current_flow_closeness_centrality, cooccurrence_graph)
    elif solver == "current_flow_betweenness":
        weighted_nodes = _weight_nodes_with_centrality_metrics(nx.current_flow_betweenness_centrality,
                                                               cooccurrence_graph)
    elif solver == "edge_current_flow_betweenness":
        weighted_nodes = _weight_nodes_with_centrality_metrics(nx.edge_current_flow_betweenness_centrality,
                                                               cooccurrence_graph)
        weighted_nodes = _flatten_nodes_betweenness_weights(weighted_nodes)
    elif solver == "load_centrality":
        weighted_nodes = nx.load_centrality(cooccurrence_graph)
    elif solver == "clustering_coefficient":
        weighted_nodes = nx.clustering(cooccurrence_graph)
    elif solver == "TeRGraph":
        weighted_nodes = compute_TeRGraph(cooccurrence_graph)
    elif solver == "coreness":
        #remove self-loops
        cooccurrence_graph.remove_edges_from(nx.selfloop_edges(cooccurrence_graph))
        weighted_nodes = nx.core_number(cooccurrence_graph)
    elif solver == "neighborhood_size":
        weighted_nodes = compute_neighborhood_size(cooccurrence_graph)
    else:
        ValueError("The node weighting solver supports only pagerank, "
                   "pagerank_numpy, pagerank_scipy, betweenness_centrality, "
                   "edge_betweenness_centrality, degree_centrality, closeness_centrality, hits, "
                   "eigenvector_centrality, katz_centrality, communicability_betweenness, "
                   "current_flow_closeness, current_flow_betweenness, edge_current_flow_betweenness, "
                   "load_centrality,clustering_coefficient,TeRGraph,coreness,neighborhood_size got '%s'"
                   % solver)

    if top_t is None:
        top_t = round(len(weighted_nodes) * top_p)

    # top T vertices in the ranking are retained for post-processing
    top_t_vertices = get_top_n_from_dict(weighted_nodes, top_t)

    _logger.debug("top T(t=%s) vertices: %s ...", top_t, top_t_vertices[:10])
    # post-processing
    # collapse sequence of adjacent keywords into a multi-word keyword
    collapsed_terms = _collapse_adjacent_keywords(weighted_nodes, flatten(original_tokenised_context))

    weighted_terms = _reweight_filtered_terms(collapsed_terms, top_t_vertices, weighted_nodes, weight_comb=weight_comb,
                                              mu=mu)

    return weighted_terms, top_t_vertices


def _flatten_nodes_betweenness_weights(weighted_nodes):
    """
    Betweenness metrics weights the nodes betweenness.

    Example result is :
        [(('systems', 'linear'), 0.30303030303030337), (('systems', 'systems'), 0.257575757575758), ...]

    The method is to flatten edge betweenness and maximum edge betweenness value in all the possible edge pairs is adopted for each vertex

    :param weighted_nodes:
    :return:
    """
    max_edge_dict = {}
    for edge_tuple, value in weighted_nodes.items():
        edge_node_1 = edge_tuple[0]
        edge_node_2 = edge_tuple[1]
        if edge_node_1 not in max_edge_dict or \
                (edge_node_1 in max_edge_dict and max_edge_dict[edge_node_1] < value):
            max_edge_dict[edge_node_1] = value

        if edge_node_2 not in max_edge_dict or \
                (edge_node_2 in max_edge_dict and max_edge_dict[edge_node_2] < value):
            max_edge_dict[edge_node_2] = value
    weighted_nodes = max_edge_dict
    return weighted_nodes


def _collapse_adjacent_keywords(weighted_keywords, original_tokenised_text):
    """
    :type weighted_keywords: list [of tuple]
    :param weighted_keywords: keywords (key head words), weight pair
    :type original_tokenised_text: list [of string]
    :param original_tokenised_text: tokenised original raw text
    :rtype: list [of list [of string]]
    :return: tokenised keywords from context that will form single-word term and multi-word term
    """
    _logger.info("collapse adjacent keywords ...")

    keywords_tmp = [word for word, weight in weighted_keywords.items()]

    # normalised_tokenised_context = [token for token in original_tokenised_text]

    keyword_tag = 'k'
    mark_keyword = lambda token, keyword_dict: keyword_tag if token in keywords_tmp else ''
    marked_text_tokens = [(token, mark_keyword(token, keywords_tmp)) for token in original_tokenised_text]

    # print("keywords marked text", marked_text_tokens)

    _key_terms = list()
    _current_term_units=[]
    # use space to construct multi-word term later
    for marked_token in marked_text_tokens:
        if marked_token[1] == 'k':
            _current_term_units.append(marked_token[0])
        else:
            if _current_term_units:
                _key_terms.append(_current_term_units)
            # reset for next term candidate
            _current_term_units=[]

    _logger.info("done.")
    return _key_terms


def _check_required_values(weighted, syntactic_categories):
    if weighted:
        # TODO: to support weighted graph in the future
        _logger.warning("Weighted graph is not supported yet.")

    if syntactic_categories is None:
        raise ValueError("`syntactic_categories` cannot be None!")


def keywords_extraction(text, window=2, top_p=1, top_t=None, directed=False, weighted=False,
                        conn_with_original_ctx=True, syntactic_categories={"NNS", "NNP", "NN", "JJ"},
                        stop_words=None, lemma=False,
                        solver="pagerank", max_iter=100, tol=1.0e-6,
                        weight_comb="norm_max", mu=5,
                        workers=1):
    """
    TextRank keywords extraction for unstructured text

    :type text: string, required
    :param text: textual data for keywords extraction
    :type window: int, required
    :param window: co-occurrence window size (default with forward and backward context). Recommend: 2-10
    :type top_t: float or None, optional
    :param top_t: the top T vertices in the ranking are retained for post-processing
                Top T is computed from Top p if value is none
    :type top_p: float or None, optional
    :param top_p: the top Percentage(P) of vertices are retained for post-processing.
                Top 1/3 of all vertices is recommended in original paper.
    :type directed: bool, required
    :param directed: directed or undirected graph (a preserved parameters)
    :type weighted: bool, optional
    :param weighted: weighted or unweighted, Custom weighted graph is not supported yet, Default as False
                    Best result is found with unweighted graph in the original paper

    :type conn_with_original_ctx: bool, optional
    :param conn_with_original_ctx: whether build vertices connections from original context or filtered context,
                    True if checking two vertices co-occurrence link from original context,
                    else checking connections from filtered context by syntactic rule

                     More vertices connections can be built if 'conn_with_original_ctx' is set to False
    :type syntactic_categories: set [of string], required
    :param syntactic_categories: Default with noun and adjective categories.
                    Syntactic categories (default as Part-Of-Speech(PoS) tags) is defined to
                    filter accepted graph vertices (default with word-based tokens as single syntactic unit).

                    Any word that is not matched with the predefined categories will be removed based on corresponding the PoS tag.

                    Best result is found with noun and adjective categories only in original paper.
    :type stop_words: set of [string {‘english’}], or None (default), Optional
    :param stop_words:  remove stopwords from PoS tagged context (token tuple list).
                The stop words are considered as noisy common/function words.
                By provide a list of stop words can improve vertices network connectivity
                and increase weights to more meaningful words.
    :type lemma: bool
    :param lemma: if lemmatize text
    :type solver: string, optional
    :param solver: {'pagerank', 'pagerank_numpy', 'pagerank_scipy', 'betweenness_centrality', 'degree_centrality',
                    'hits', 'closeness_centrality', 'edge_betweenness_centrality', 'eigenvector_centrality',
                    'katz_centrality', 'communicability_betweenness', 'current_flow_closeness', 'current_flow_betweenness',
                    'edge_current_flow_betweenness', 'load_centrality', 'clustering_coefficient', 'TeRGraph',
                    'coreness', 'neighborhood_size'}, default 'pagerank'
        PageRank Algorithms supported in networkx to use in the vertices ranking.

        - 'pagerank' networkx pagerank implementation
        - 'pagerank_numpy' numpy pagerank implementation
        - 'pagerank_scipy' scipy pagerank implementation
        - 'betweenness_centrality' computes the shortest-path betweenness centrality of a node
        - 'degree_centrality' computes the degree centrality for nodes.
        - 'hits' computes HITS algorithm for a node. The avg. of Authority value and Hub value is computed
        - 'closeness_centrality' computes closeness centrality for nodes.
        - 'edge_betweenness_centrality' computes betweenness centrality for edges.
                                Maximum edge betweenness value in all the possible edge pairs is adopted for each vertex
        - 'eigenvector_centrality' computes the eigenvector centrality for the cooocurrence graph.
        - 'katz_centrality' computes the Katz centrality for the nodes based on the centrality of its neighbors.
        - 'communicability_betweenness' computes subgraph communicability for all pairs of nodes
        - 'current_flow_closeness' computes current-flow closeness centrality for nodes.
        - 'current_flow_betweenness' computes current-flow betweenness centrality for nodes.
        - 'edge_current_flow_betweenness' computes current-flow betweenness centrality for edges.
        - 'load_centrality' computes edge load. This is a experimental algorithm in nextworkx
                                    that counts the number of shortest paths which cross each edge.
        - 'clustering_coefficient' computes the clustering coefficient for nodes. Only undirected graph is supported.
        - 'TeRGraph': computes the TeRGraph (Lossio-Ventura, 2014) weights for nodes.
                        The solver requires a connected graph and isolated nodes will be set to 0.
        - 'coreness' (Batagelj & Zaversnik, 2003) measures how "deep" a node(word/phrase) is in the co-occurrence network.
                This indicates how strongly the node is connected to the network. The "deeper" a word, the more it is important.
                The metric is not suitable for ranking terms directly, but it is proved as useful feature for keywords extraction
        - 'neighborhood_size' computes the number of immediate neighbors to a node.
                    This is a version of node degree that disregards self-loops

        Note: Centrality measures (such as "current flow betweeness", "current flow closeness", "communicability_betweenness")
            does not support loosely connected graph and betweeness centrality measures cannot compute on single isolated nodes.
             It is recommended to re-consider the graph construction method or increase context window size to
             ensure a (strongly) connected graph.
    :type max_iter: int, optional
    :param max_iter: number of maximum iteration of pagerank, katz_centrality
    :type tol: float, optional, default 1.0e-6
    :param tol: Error tolerance used to check convergence, the value varies for specific solver
    :type weight_comb: str
    :param weight_comb:  {'avg', 'norm_avg', 'log_norm_avg', 'gaussian_norm_avg', 'sum', 'norm_sum', 'log_norm_sum',
                'gaussian_norm_sum', 'max', 'norm_max', 'log_norm_max', 'gaussian_norm_max',
                'len_log_norm_max', 'len_log_norm_avg', 'len_log_norm_sum'}, default 'norm_max'
            The weight combination method for multi-word candidate terms weighing.

            - 'max' : maximum value of vertices weights
            - 'avg' : avarage vertices weight
            - 'sum' : sum of vertices weights
            - 'norm_max' : MWT unit size normalisation of 'max' weight
            - 'norm_avg' : MWT unit size normalisation of 'avg' weight
            - 'norm_sum' : MWT unit size normalisation of 'sum' weight
            - 'log_norm_max' : logarithm based normalisation of 'max' weight
            - 'log_norm_avg' : logarithm based normalisation of 'avg' weight
            - 'log_norm_sum' : logarithm based normalisation of 'sum' weight
            - 'gaussian_norm_max' : gaussian normalisation of 'max' weight
            - 'gaussian_norm_avg' : gaussian normalisation of 'avg' weight
            - 'gaussian_norm_sum' : gaussian normalisation of 'sum' weight
            - 'len_log_norm_max': log2(|a| + 0.1) * 'max' adapted from CValue (Frantzi, 2000) formulate
            - 'len_log_norm_avg': log2(|a| + 0.1) * 'avg' adapted from CValue (Frantzi, 2000) formulate
            - 'len_log_norm_sum': log2(|a| + 0.1) * 'sum' adapted from CValue (Frantzi, 2000) formulate

            NOTE: \*_norm_\*" penalises/smooth the longer term (than default 5 token size)
                to achieve a saturation level as term size grows
    :type mu: int, optional
    :param mu: mean value to set a center point (default to 5) in order to rank the MWT candidates higher that are near the central point
            This param is only required and effective for normalisation based MWT weighting methods
    :type workers: int, optional
    :param workers: number of workers (CPU cores)

    :rtype: tuple [list[tuple[string,float]], dict[string:float]]
    :return: keywords: sorted keywords with weights along with Top T weighted vertices
    :raise: ValueError
    """
    _check_solver_option(solver)

    if solver in ['communicability_betweenness', 'current_flow_closeness', 'current_flow_betweenness',
                  'edge_current_flow_betweenness'] \
            and window < 5:
        _logger.warning("'%s' requires highly connected graph. "
                        "Please consider to increase context window size. "
                        "Isolated nodes would be removed by most of centrality "
                        "and betweeness metrics otherwise. ", solver)

    _check_required_values(weighted, syntactic_categories)
    _check_weight_comb_option(weight_comb)

    global MAX_PROCESSES
    MAX_PROCESSES = workers

    _logger.info("pre-processing text with syntactic_categories [%s] and stop words [%s] ... ",
                 syntactic_categories, stop_words)

    preprocessed_corpus_context = preprocessing(text, syntactic_categories=syntactic_categories,
                                                stop_words=stop_words, lemma=lemma)

    weighted_keywords, top_t_vertices = _keywords_extraction_from_preprocessed_context(preprocessed_corpus_context,
                                                                                       window=window,
                                                                                       top_p=top_p, top_t=top_t,
                                                                                       directed=directed,
                                                                                       weighted=weighted,
                                                                                       conn_with_original_ctx=conn_with_original_ctx,
                                                                                       solver=solver,max_iter=max_iter, tol=tol,
                                                                                       weight_comb=weight_comb, mu=mu)

    return list(sort_dict_by_value(weighted_keywords).items()), top_t_vertices


def _check_solver_option(solver):
    if solver not in ['pagerank', 'pagerank_numpy', 'pagerank_scipy',
                      'betweenness_centrality', 'degree_centrality', 'hits', 'closeness_centrality',
                      'edge_betweenness_centrality', 'eigenvector_centrality', 'katz_centrality',
                      'communicability_betweenness', 'current_flow_closeness', 'current_flow_betweenness',
                      'edge_current_flow_betweenness', 'load_centrality', 'clustering_coefficient',
                      'TeRGraph', 'coreness', 'neighborhood_size']:
        raise ValueError("PageRank solver supports only 'pagerank', "
                         "'pagerank_numpy', 'pagerank_scipy', 'betweenness_centrality', "
                         "'edge_betweenness_centrality', 'degree_centrality', "
                         "'closeness_centrality', 'hits', 'eigenvector_centrality', 'katz_centrality', "
                         "'communicability_betweenness','current_flow_closeness',"
                         " 'current_flow_betweenness', 'edge_current_flow_betweenness', 'load_centrality', "
                         "'clustering_coefficient','TeRGraph','coreness','neighborhood_size' got '%s'"
                         % solver)
    if solver == "pagerank_numpy" or solver== "katz_centrality":
        import pkg_resources
        pkg_resources.require("numpy")

    if solver == "pagerank_scipy":
        import pkg_resources
        pkg_resources.require("scipy")


def keywords_extraction_from_segmented_corpus(segmented_corpus_context, solver="pagerank",
                                              max_iter=100, tol=1.0e-6,
                                              window=2, top_p=0.3, top_t=None,
                                              directed=False, weighted=False,
                                              conn_with_original_ctx=True,
                                              syntactic_categories={"NNS", "NNP", "NN", "JJ"},
                                              stop_words=None, lemma=False,
                                              weight_comb="norm_max", mu=5,
                                              export=False, export_format="csv", export_path="", encoding="utf-8",
                                              workers=1):
    """
    TextRank keywords extraction for a list of context of tokenised textual corpus.
    This method allows any pre-defined keyword co-occurrence context criteria (e.g., sentence, or paragraph,
    or section, or a user-defined segment) and any pre-defined word segmentation

    :type tokenised_corpus_context: list|generator, required
    :param tokenised_corpus_context: pre-tokenised corpus formatted in pre-defined context list.
            Tokenised sentence list is the recommended(and default) context corpus in TextRank.
            You can also choose your own pre-defined co-occurrence context (e.g., paragraph, entire document, a user-defined segment).

           :Example: input:

            >>> context_1 = ["The", "quick", "brown", "fox", "jumped", "over", "the", "lazy", "dog", ".", "hey","diddle", "diddle", ",", "the", "cat", "and", "the", "fiddle","."]

            >>> context_2 = ["The", "cow", "jumped", "over", "the", "moon",".", "The", "little", "dog", "laughted", "to", "see","such", "fun", "."]

            >>> segmented_corpus_context = [context_1, context_2]
    :type solver: string, optional
    :param solver: {'pagerank', 'pagerank_numpy', 'pagerank_scipy', 'betweenness_centrality', 'degree_centrality',
                    'hits', 'closeness_centrality', 'edge_betweenness_centrality', 'eigenvector_centrality',
                    'katz_centrality', 'communicability_betweenness', 'current_flow_closeness', 'current_flow_betweenness',
                    'edge_current_flow_betweenness', 'load_centrality', 'clustering_coefficient', 'TeRGraph',
                    'coreness'}, default 'pagerank'
        PageRank Algorithms supported in networkx to use in the vertices ranking.

        - 'betweenness_centrality' computes the shortest-path betweenness centrality of a node
        - 'degree_centrality' computes the degree centrality for nodes.
        - 'hits' computes HITS algorithm for a node. The avg. of Authority value and Hub value is computed
        - 'closeness_centrality' computes closeness centrality for nodes.
        - 'edge_betweenness_centrality' computes betweenness centrality for edges.
                                Maximum edge betweenness value in all the possible edge pairs is adopted for each vertex
        - 'eigenvector_centrality' computes the eigenvector centrality for the cooocurrence graph.
        - 'katz_centrality' computes the Katz centrality for the nodes based on the centrality of its neighbors.
        - 'communicability_betweenness' computes subgraph communicability for all pairs of nodes
        - 'current_flow_closeness' computes current-flow closeness centrality for nodes.
        - 'current_flow_betweenness' computes current-flow betweenness centrality for nodes.
        - 'edge_current_flow_betweenness' computes current-flow betweenness centrality for edges.
        - 'load_centrality' computes edge load. This is a experimental algorithm in nextworkx
                                    that counts the number of shortest paths which cross each edge.
        - 'clustering_coefficient' computes the clustering coefficient for nodes. Only undirected graph is supported.
        - 'TeRGraph': computes the TeRGraph (Lossio-Ventura, 2014) weights for nodes.
                        The solver requires a connected graph and isolated nodes will be set to 0.
        - 'coreness' (Batagelj & Zaversnik, 2003) measures how "deep" a node(word/phrase) is in the co-occurrence network.
                This indicates how strongly the node is connected to the network. The "deeper" a word, the more it is important.
                The metric is not suitable for ranking terms directly, but it is proved as useful feature for keywords extraction
        - 'neighborhood_size' computes the number of immediate neighbors to a node.
                    This is a version of node degree that disregards self-loops

        Note: Centrality measures (such as "current flow betweeness", "current flow closeness", "communicability_betweenness")
            does not support loosely connected graph and betweeness centrality measures cannot compute on single isolated nodes.
             It is recommended to re-consider the graph construction method or increase context window size to
             ensure a (strongly) connected graph.
    :type max_iter: int, optional
    :param max_iter: number of maximum iteration of pagerank, katz_centrality
    :type tol: float, optional, default 1.0e-6
    :param tol: Error tolerance used to check convergence, the value varies for specific solver
    :type window: int, required
    :param window: co-occurrence window size (default with forward and backward context). Default value: 2
    :type top_p: float, optional
    :param top_p: the top Percentage of vertices are retained for post-processing, Default as 1/3 of all vertices
    :type top_t: int|None(default), optional
    :param top_t: the top T vertices in the ranking are retained for post-processing
    :type directed: bool, required
    :param directed: directed or undirected graph, best result is found with undirected graph in the original paper. Default as False
    :type weighted: bool, required
    :param weighted: weighted or unweighted, Custom weighted graph is not supported yet, Default as False
                    Best result is found with unweighted graph in the original paper
            When this is set to True, graph construction component will try to construct a fully-connected graph
            by connecting isolated nodes (due to small context window) with low weight (default to 0.001)
            Please check if the ranking algorithm supports weighted graph
            Note: custom weights is not supported yet.

    :type conn_with_original_ctx: bool, optional
    :param conn_with_original_ctx: True if checking two vertices co-occurrence link from original context
                                else checking connections from filtered context
            More vertices connection can be built if 'conn_with_original_ctx' is set to False
    :type syntactic_categories: set[string], required
    :param syntactic_categories: Syntactic categories (default as Part-Of-Speech(PoS) tags) is defined to
                        filter accepted graph vertices (essentially word-based tokens).
                        Default with noun and adjective categories.

                        Any word that is not matched with the predefined categories will be removed
                        based on corresponding the PoS tag.

                        Best result is found with noun and adjective categories only in original paper.
    :type stop_words: set[string {‘english’}] | None (default), Optional
    :param stop_words:  remove stopwords from PoS tagged context (token tuple list)
                The stop words are considered as noisy common/function words.
                By provide a list of stop words can improve vertices network connectivity
                and increase weights to more meaningful words.
    :type lemma: bool
    :param lemma: if lemmatize text
    :type weight_comb: str
    :param weight_comb:  {'avg', 'norm_avg', 'log_norm_avg', 'gaussian_norm_avg', 'sum', 'norm_sum', 'log_norm_sum',
                'gaussian_norm_sum', 'max', 'norm_max', 'log_norm_max', 'gaussian_norm_max',
                'len_log_norm_max', 'len_log_norm_avg', 'len_log_norm_sum'}, default 'norm_max'
            The weight combination method for multi-word candidate terms weighing.

            - 'max' : maximum value of vertices weights
            - 'avg' : avarage vertices weight
            - 'sum' : sum of vertices weights
            - 'norm_max' : MWT unit size normalisation of 'max' weight
            - 'norm_avg' : MWT unit size normalisation of 'avg' weight
            - 'norm_sum' : MWT unit size normalisation of 'sum' weight
            - 'log_norm_max' : logarithm based normalisation of 'max' weight
            - 'log_norm_avg' : logarithm based normalisation of 'avg' weight
            - 'log_norm_sum' : logarithm based normalisation of 'sum' weight
            - 'gaussian_norm_max' : gaussian normalisation of 'max' weight
            - 'gaussian_norm_avg' : gaussian normalisation of 'avg' weight
            - 'gaussian_norm_sum' : gaussian normalisation of 'sum' weight
            - 'len_log_norm_max': log2(|a| + 0.1) * 'max' adapted from CValue (Frantzi, 2000) formulate
            - 'len_log_norm_avg': log2(|a| + 0.1) * 'avg' adapted from CValue (Frantzi, 2000) formulate
            - 'len_log_norm_sum': log2(|a| + 0.1) * 'sum' adapted from CValue (Frantzi, 2000) formulate

            NOTE: \*_norm_\*" penalises/smooth the longer term (than default 5 token size)
                to achieve a saturation level as term size grows
    :type mu: int, optional
    :param mu: mean value to set a center point (default to 5) in order to rank the candidates higher that are near the central point
            This param is only required and effective for normalisation based MWT weighting method
    :type export: bool
    :param export: True if export result else False
    :type export_format: string
    :param export_format: export file format. Support options: "csv"|"json". Default with "csv"
    :type export_path: string
    :param export_path: file path where the result will be exported to
    :type encoding: string, required
    :param encoding: encoding of the text, default as 'utf-8',
    :type workers: int
    :param workers: available CPU cores, default to use all the available CPU cores
    :rtype: tuple [list[tuple[string,float]], dict[string:float]]
    :return: keywords: sorted keywords with weights along with Top T weighted vertices
    """
    global MAX_PROCESSES
    MAX_PROCESSES = workers

    _check_required_values(weighted, syntactic_categories)
    _check_export_option(export, export_format)
    _check_weight_comb_option(weight_comb)

    pre_processed_tokenised_context = preprocessing_tokenised_context(segmented_corpus_context,
                                                                      syntactic_categories=syntactic_categories,
                                                                      stop_words=stop_words, lemma=lemma)

    weighted_keywords, top_t_vertices = _keywords_extraction_from_preprocessed_context(pre_processed_tokenised_context,
                                                                                       solver=solver,
                                                                                       max_iter=max_iter,
                                                                                       tol=tol, top_p=top_p, top_t=top_t,
                                                                                       directed=directed, window=window,
                                                                                       weighted=weighted,
                                                                                       conn_with_original_ctx=conn_with_original_ctx,
                                                                                       weight_comb=weight_comb, mu=mu)

    sorted_weighted_keywords = list(sort_dict_by_value(weighted_keywords).items())

    _export_result(sorted_weighted_keywords, export=export, export_format=export_format,
                   export_path=export_path, encoding=encoding)

    return sorted_weighted_keywords, top_t_vertices


def _export_result(weighted_term_results, export=False, export_format="csv", export_path="", encoding="utf-8"):
    if export:
        _logger.info("exporting sorted keywords into [%s]", export_path)
        if export_format.lower() == "csv":
            export_list_of_tuple_into_csv(export_path, weighted_term_results, header=["term", "weight"])
        else:
            export_list_of_tuples_into_json(export_path, weighted_term_results, encoding=encoding)

        _logger.info("complete result export.")


def _load_preprocessed_corpus_context(tagged_corpus_context, pos_filter=None, stop_words_filter=None, lemma=False):
    """
    load preprocessed corpus context(a tuple list of tokenised context and PoS tagged context) from tagged corpus context

    :type tagged_corpus_context: list[tuple[string,string]]|generator, required
    :param tagged_corpus_context: PoS tagged tokenised corpus textual context
    :rtype: generator[tuple[list[string],list[tuple[string,string]]]]
    :return: preprocessed corpus context (tokenised_context, syntactic filtered context)
    """
    for tagged_context in tagged_corpus_context:
        normed_token_list, normed_tagged_context = _normalise_tagged_token_list(tagged_context, lemma=lemma)
        yield normed_token_list, _syntactic_filter_context(normed_tagged_context, pos_filter, stop_words_filter)
        # yield [tagged_token[0] for tagged_token in tagged_context], \
        #      _syntactic_filter_context(tagged_context, pos_filter, stop_words_filter)


def keywords_extraction_from_tagged_corpus(tagged_corpus_context,
                                           solver="pagerank", max_iter=100,tol=1.0e-6,
                                           window=2, top_p=0.3, top_t=None, directed=False, weighted=False,
                                           conn_with_original_ctx=True,
                                           syntactic_categories={"NNS", "NNP", "NN", "JJ"},
                                           stop_words=None, lemma=False,weight_comb="norm_max", mu=5,
                                           export=False, export_format="csv", export_path="", encoding="utf-8",
                                           workers=1):
    """
    TextRank keywords extraction for pos tagged corpus context list

    This method allows to use external Part-of-Speech tagging, and any pre-defined keyword co-occurrence context criteria (e.g., sentence, or paragraph,
    or section, or a user-defined segment) and any pre-defined word segmentation

    :type tagged_corpus_context: list[list[tuple[string, string]]] or generator
    :param tagged_corpus_context: pre-tagged corpus in the form of tuple
    :type solver: string, optional
    :param solver: {'pagerank', 'pagerank_numpy', 'pagerank_scipy', 'betweenness_centrality', 'degree_centrality',
                    'hits', 'closeness_centrality', 'edge_betweenness_centrality', 'eigenvector_centrality',
                    'katz_centrality', 'communicability_betweenness', 'current_flow_closeness', 'current_flow_betweenness',
                    'edge_current_flow_betweenness', 'load_centrality', 'clustering_coefficient', 'TeRGraph',
                    'coreness'}, default 'pagerank'
        PageRank Algorithms supported in networkx to use in the vertices ranking.

        - 'betweenness_centrality' computes the shortest-path betweenness centrality of a node
        - 'degree_centrality' computes the degree centrality for nodes.
        - 'hits' computes HITS algorithm for a node. The avg. of Authority value and Hub value is computed
        - 'closeness_centrality' computes closeness centrality for nodes.
        - 'edge_betweenness_centrality' computes betweenness centrality for edges.
                                Maximum edge betweenness value in all the possible edge pairs is adopted for each vertex
        - 'eigenvector_centrality' computes the eigenvector centrality for the cooocurrence graph.
        - 'katz_centrality' computes the Katz centrality for the nodes based on the centrality of its neighbors.
        - 'communicability_betweenness' computes subgraph communicability for all pairs of nodes
        - 'current_flow_closeness' computes current-flow closeness centrality for nodes.
        - 'current_flow_betweenness' computes current-flow betweenness centrality for nodes.
        - 'edge_current_flow_betweenness' computes current-flow betweenness centrality for edges.
        - 'load_centrality' computes edge load. This is a experimental algorithm in nextworkx
                                    that counts the number of shortest paths which cross each edge.
        - 'clustering_coefficient' computes the clustering coefficient for nodes. Only undirected graph is supported.
        - 'TeRGraph': computes the TeRGraph (Lossio-Ventura, 2014) weights for nodes.
                        The solver requires a connected graph and isolated nodes will be set to 0.
        - 'coreness' (Batagelj & Zaversnik, 2003) measures how "deep" a node(word/phrase) is in the co-occurrence network.
                This indicates how strongly the node is connected to the network. The "deeper" a word, the more it is important.
                The metric is not suitable for ranking terms directly, but it is proved as useful feature for keywords extraction
        - 'neighborhood_size' computes the number of immediate neighbors to a node.
                    This is a version of node degree that disregards self-loops

        Note: Centrality measures (such as "current flow betweeness", "current flow closeness", "communicability_betweenness")
            does not support loosely connected graph and betweeness centrality measures cannot compute on single isolated nodes.
             It is recommended to re-consider the graph construction method or increase context window size to
             ensure a (strongly) connected graph.
    :type max_iter: int, optional
    :param max_iter: number of maximum iteration of pagerank, katz_centrality
    :type tol: float, optional, default 1e4
    :param tol: Error tolerance used to check convergence, the value varies for specific solver
    :type window: int, required
    :param window: co-occurrence window size (default with forward and backward context). Default value: 2
    :type top_p: float, optional
    :param top_p: the top Percentage of vertices are retained for post-processing, Default as 1/3 of all vertices
    :type top_t: int|None(default), optional
    :param top_t: the top T vertices in the ranking are retained for post-processing
    :type directed: bool, required
    :param directed: directed or undirected graph, best result is found with undirected graph in the original paper. Default as False
    :type weighted: bool, required
    :param weighted: weighted or unweighted, weighted graph is not supported yet, Default as False
                    Best result is found with unweighted graph in the original paper
    :type conn_with_original_ctx: bool, optional
    :param conn_with_original_ctx: True if checking two vertices connections from original context
                                else checking connections from filtered context
            More vertices connection can be built if 'conn_with_original_ctx' is set to False
    :type syntactic_categories: set[string], required
    :param syntactic_categories: Syntactic categories (default as Part-Of-Speech(PoS) tags) is defined to
                        filter accepted graph vertices (essentially word-based tokens).
                        Default with noun and adjective categories.

                        Any word that is not matched with the predefined categories will be removed
                        based on corresponding the PoS tag.

                        Best result is found with noun and adjective categories only in original paper.
    :type stop_words: set[string {‘english’}] | None (default), Optional
    :param stop_words:  remove stopwords from PoS tagged context (token tuple list)
                The stop words are considered as noisy common/function words.
                By provide a list of stop words can improve vertices network connectivity
                and increase weights to more meaningful words.
    :type lemma: bool
    :param lemma: if lemmatize text
    :type weight_comb: str
    :param weight_comb:  {'avg', 'norm_avg', 'log_norm_avg', 'gaussian_norm_avg', 'sum', 'norm_sum', 'log_norm_sum',
                'gaussian_norm_sum', 'max', 'norm_max', 'log_norm_max', 'gaussian_norm_max',
                'len_log_norm_max', 'len_log_norm_avg', 'len_log_norm_sum'}, default 'norm_max'
            The weight combination method for multi-word candidate terms weighing.

            - 'max' : maximum value of vertices weights
            - 'avg' : avarage vertices weight
            - 'sum' : sum of vertices weights
            - 'norm_max' : MWT unit size normalisation of 'max' weight
            - 'norm_avg' : MWT unit size normalisation of 'avg' weight
            - 'norm_sum' : MWT unit size normalisation of 'sum' weight
            - 'log_norm_max' : logarithm based normalisation of 'max' weight
            - 'log_norm_avg' : logarithm based normalisation of 'avg' weight
            - 'log_norm_sum' : logarithm based normalisation of 'sum' weight
            - 'gaussian_norm_max' : gaussian normalisation of 'max' weight
            - 'gaussian_norm_avg' : gaussian normalisation of 'avg' weight
            - 'gaussian_norm_sum' : gaussian normalisation of 'sum' weight
            - 'len_log_norm_max': log2(|a| + 0.1) * 'max' adapted from CValue (Frantzi, 2000) formulate
            - 'len_log_norm_avg': log2(|a| + 0.1) * 'avg' adapted from CValue (Frantzi, 2000) formulate
            - 'len_log_norm_sum': log2(|a| + 0.1) * 'sum' adapted from CValue (Frantzi, 2000) formulate

            NOTE: \*_norm_\*" penalises/smooth the longer term (than default 5 token size)
                to achieve a saturation level as term size grows
    :type mu: int, optional
    :param mu: mean value to set a center point (default to 5) in order to rank the candidates higher that are near the central point
            This param is only required and effective for normalisation based MWT weighting method
    :type export: bool
    :param export: True if export result else False
    :type export_format: string
    :param export_format: {'csv', 'json'}, default 'csv'
                    export file format
    :type export_path: string
    :param export_path: file path where the result will be exported to
    :type encoding: string, required
    :param encoding: encoding of export file, default as 'utf-8',
    :type workers: int
    :param workers: available CPU cores, default to use all the available CPU cores
    :rtype: tuple [list[tuple[string,float]], dict[string:float]]
    :return: keywords: sorted keywords with weights along with Top T weighted vertices
    """
    global MAX_PROCESSES
    MAX_PROCESSES = workers

    _check_required_values(weighted, syntactic_categories)
    _check_weight_comb_option(weight_comb)

    # tokenised_context, context_syntactic_units in preprocessed_context
    preprocessed_corpus_context = _load_preprocessed_corpus_context(tagged_corpus_context,
                                                                    pos_filter=lambda t: filter(
                                                                        lambda a: a[1] in syntactic_categories, t),
                                                                    stop_words_filter=None if stop_words is None else
                                                                    lambda t: filter(lambda a: a[0] not in stop_words,
                                                                                     t),
                                                                    lemma=lemma)

    weighted_keywords, top_t_vertices = _keywords_extraction_from_preprocessed_context(preprocessed_corpus_context,
                                                                                       window=window, top_p=top_p, top_t=top_t,
                                                                                       directed=directed,weighted=weighted,
                                                                                       conn_with_original_ctx=conn_with_original_ctx,
                                                                                       solver=solver,
                                                                                       max_iter=max_iter, tol=tol,
                                                                                       weight_comb=weight_comb, mu=mu)

    sorted_weighted_keywords = list(sort_dict_by_value(weighted_keywords).items())

    _export_result(sorted_weighted_keywords, export=export, export_format=export_format,
                   export_path=export_path, encoding=encoding)

    return sorted_weighted_keywords, top_t_vertices


def _check_export_option(export, export_format):
    if not isinstance(export, bool):
        raise ValueError("'export' must be a boolean value.")

    if export:
        if export_format is None:
            raise ValueError("'export' should not be None.")

        if export_format.lower() not in ["csv", "json"]:
            raise ValueError("Only 'csv' or 'json' is allowed as export format.")


def _check_weight_comb_option(weight_comb):
    if not isinstance(weight_comb, str):
        raise ValueError("'weight_comb' must be a str value.")

    if weight_comb is None:
        raise ValueError("'weight_comb' should not be None.")

    if weight_comb not in ("avg", "norm_avg", "log_norm_avg", "gaussian_norm_avg", "sum", "norm_sum",
                           "log_norm_sum", "gaussian_norm_sum", "max", "norm_max", "log_norm_max",
                           "gaussian_norm_max",'len_log_norm_max', 'len_log_norm_avg', 'len_log_norm_sum'):
        raise ValueError("Unspported weight_comb '%s'! "
                         "Options are 'avg', 'norm_avg', 'log_norm_avg', 'gaussian_norm_avg', 'sum', "
                         "'norm_sum', 'log_norm_sum', 'gaussian_norm_sum', 'max', 'norm_max',"
                         " 'log_norm_max', 'gaussian_norm_max', "
                         "'len_log_norm_max', 'len_log_norm_avg', 'len_log_norm_sum'. " % weight_comb)


def keywords_extraction_from_corpus_directory(corpus_dir, encoding="utf-8", solver="pagerank",
                                              max_iter=100, tol=1e-4, window=2,
                                              top_p=0.3, top_t=None, directed=False, weighted=False,
                                              syntactic_categories={"NNS", "NNP", "NN", "JJ"},
                                              stop_words=None, lemma=False, weight_comb="norm_max", mu=5,
                                              export=False, export_format="csv", export_path="", workers=1):
    """

    :type corpus_dir: string
    :param corpus_dir: corpus directory where text files are located and will be read and processed
    :type encoding: string, required
    :param encoding: encoding of the text, default as 'utf-8',
    :type solver: string, optional
    :param solver: {'pagerank', 'pagerank_numpy', 'pagerank_scipy', 'betweenness_centrality', 'degree_centrality',
                    'hits', 'closeness_centrality', 'edge_betweenness_centrality', 'eigenvector_centrality',
                    'katz_centrality', 'communicability_betweenness', 'current_flow_closeness', 'current_flow_betweenness',
                    'edge_current_flow_betweenness', 'load_centrality', 'clustering_coefficient', 'TeRGraph',
                    'coreness'}, default 'pagerank'
        PageRank Algorithms supported in networkx to use in the vertices ranking.

        - 'betweenness_centrality' computes the shortest-path betweenness centrality of a node
        - 'degree_centrality' computes the degree centrality for nodes.
        - 'hits' computes HITS algorithm for a node. The avg. of Authority value and Hub value is computed
        - 'closeness_centrality' computes closeness centrality for nodes.
        - 'edge_betweenness_centrality' computes betweenness centrality for edges.
                                Maximum edge betweenness value in all the possible edge pairs is adopted for each vertex
        - 'eigenvector_centrality' computes the eigenvector centrality for the cooocurrence graph.
        - 'katz_centrality' computes the Katz centrality for the nodes based on the centrality of its neighbors.
        - 'communicability_betweenness' computes subgraph communicability for all pairs of nodes
        - 'current_flow_closeness' computes current-flow closeness centrality for nodes.
        - 'current_flow_betweenness' computes current-flow betweenness centrality for nodes.
        - 'edge_current_flow_betweenness' computes current-flow betweenness centrality for edges.
        - 'load_centrality' computes edge load. This is a experimental algorithm in nextworkx
                                    that counts the number of shortest paths which cross each edge.
        - 'clustering_coefficient' computes the clustering coefficient for nodes. Only undirected graph is supported.
        - 'TeRGraph': computes the TeRGraph (Lossio-Ventura, 2014) weights for nodes.
                        The solver requires a connected graph and isolated nodes will be set to 0.
        - 'coreness' (Batagelj & Zaversnik, 2003) measures how "deep" a node(word/phrase) is in the co-occurrence network.
                This indicates how strongly the node is connected to the network. The "deeper" a word, the more it is important.
                The metric is not suitable for ranking terms directly, but it is proved as useful feature for keywords extraction
        - 'neighborhood_size' computes the number of immediate neighbors to a node.
                    This is a version of node degree that disregards self-loops

        Note: Centrality measures (such as "current flow betweeness", "current flow closeness", "communicability_betweenness")
            does not support loosely connected graph and betweeness centrality measures cannot compute on single isolated nodes.
             It is recommended to re-consider the graph construction method or increase context window size to
             ensure a (strongly) connected graph.
    :type max_iter: int, optional
    :param max_iter: number of maximum iteration of pagerank, katz_centrality
    :type tol: float, optional, default 1e-4
    :param tol: Error tolerance used to check convergence, the value varies for specific solver
    :type window: int, required
    :param window: co-occurrence window size (default with forward and backward context). Default value: 2
    :type top_p: float, required
    :param top_p: the top Percentage of vertices are retained for post-processing, Default as 1/3 of all vertices
    :type top_t: int|None(default), optional
    :param top_t: the top T vertices in the ranking are retained for post-processing
                   if None is provided, top T will be computed from top P. Otherwise, top T will be used to filter vertices

    :type directed: bool, required
    :param directed: directed or undirected graph, best result is found with undirected graph in the original paper. Default as False
    :type weighted: bool, required
    :param weighted: weighted or unweighted, weighted graph is not supported yet, Default as False
                    Best result is found with unweighted graph in the original paper
    :type syntactic_categories: set[string], required
    :param syntactic_categories: Syntactic categories (default as Part-Of-Speech(PoS) tags) is defined to
                        filter accepted graph vertices (essentially word-based tokens).
                        Default with noun and adjective categories.

                        Any word that is not matched with the predefined categories will be removed
                        based on corresponding the PoS tag.

                        Best result is found with noun and adjective categories only in original paper.
    :type stop_words: set[string {‘english’}] | None (default), Optional
    :param stop_words:  remove stopwords from PoS tagged context (token tuple list)
                The stop words are considered as noisy common/function words.
                By provide a list of stop words can improve vertices network connectivity
                and increase weights to more meaningful words.
    :type lemma: bool
    :param lemma: if lemmatize text
    :type weight_comb: str
    :param weight_comb:  {'avg', 'norm_avg', 'log_norm_avg', 'gaussian_norm_avg', 'sum', 'norm_sum', 'log_norm_sum',
                'gaussian_norm_sum', 'max', 'norm_max', 'log_norm_max', 'gaussian_norm_max',
                'len_log_norm_max', 'len_log_norm_avg', 'len_log_norm_sum'}, default 'norm_max'
            The weight combination method for multi-word candidate terms weighing.

            - 'max' : maximum value of vertices weights
            - 'avg' : avarage vertices weight
            - 'sum' : sum of vertices weights
            - 'norm_max' : MWT unit size normalisation of 'max' weight
            - 'norm_avg' : MWT unit size normalisation of 'avg' weight
            - 'norm_sum' : MWT unit size normalisation of 'sum' weight
            - 'log_norm_max' : logarithm based normalisation of 'max' weight
            - 'log_norm_avg' : logarithm based normalisation of 'avg' weight
            - 'log_norm_sum' : logarithm based normalisation of 'sum' weight
            - 'gaussian_norm_max' : gaussian normalisation of 'max' weight
            - 'gaussian_norm_avg' : gaussian normalisation of 'avg' weight
            - 'gaussian_norm_sum' : gaussian normalisation of 'sum' weight
            - 'len_log_norm_max': log2(|a| + 0.1) * 'max' adapted from CValue (Frantzi, 2000) formulate
            - 'len_log_norm_avg': log2(|a| + 0.1) * 'avg' adapted from CValue (Frantzi, 2000) formulate
            - 'len_log_norm_sum': log2(|a| + 0.1) * 'sum' adapted from CValue (Frantzi, 2000) formulate

            NOTE: \*_norm_\*" penalises/smooth the longer term (than default 5 token size)
                to achieve a saturation level as term size grows
    :type mu: int, optional
    :param mu: mean value to set a center point (default to 5) in order to rank the candidates higher that are near the central point
            This param is only required and effective for normalisation based MWT weighting method
    :type export: bool
    :param export: True if export result else False
    :type export_format: string
    :param export_format: export file format.Support options: "csv"|"json". Default with "csv"
    :type export_path: string
    :param export_path: file path where the result will be exported to
    :type workers: int
    :param workers: available CPU cores that can be used to parallelize co-occurrence computation
    :rtype: tuple [list[tuple[string,float]], dict[string:float]]
    :return: keywords: sorted keywords with weights along with Top T weighted vertices
    """
    _check_export_option(export, export_format)
    _check_weight_comb_option(weight_comb)

    tokenised_sentences = CorpusContent2RawSentences(corpus_dir, encoding=encoding)

    return keywords_extraction_from_segmented_corpus(tokenised_sentences, solver=solver,
                                                     max_iter=max_iter, tol=tol,
                                                     window=window, top_p=top_p, top_t=top_t,
                                                     directed=directed, weighted=weighted,
                                                     syntactic_categories=syntactic_categories, stop_words=stop_words,
                                                     lemma=lemma, weight_comb=weight_comb, mu=mu,
                                                     export=export, export_format=export_format,
                                                     export_path=export_path, encoding=encoding,
                                                     workers=workers)
