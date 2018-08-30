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

import logging
import math
import numpy as np

from jgtextrank.utility import MultiprocPool

_logger = logging.getLogger("jgtextrank.metrics")

__author__ = 'Jie Gao <j.gao@sheffield.ac.uk>'

__all__ = ["_get_max_score", "_get_average_score", "_get_sum_score", "_term_size_normalize",
           "_log_normalise", "_probability_density", "_gaussian_normalise", "_get_plus_score",
           "TermGraphValue", "GCValue"]

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


def _term_size_normalize(base_score, unit_size):
    return base_score / float(unit_size)


def _log_normalise(base_score, mu, unit_size):
    if unit_size > 1:
        # print("_log_normalise with mu=", mu, " , unit_size:", unit_size)
        base_score = base_score / math.log(unit_size, mu)
    return base_score


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


def _gaussian_normalise(base_score, mu, sigma, unit_size):
    """
    gaussian normalisation of 'base' weight
    :param base_score: float, base weight of candidate terms
    :param mu: int, mean value to set a center point (default to 5) in order to rank the candidates higher that are near the central point
            This param is only required for normalisation based MWT weighting method
    :param sigma: float64, standard deviation of term length in MWTs
    :param unit_size: int, size of MWTs
    :return:float
    """
    norm_value = 1 - _probability_density(unit_size, mu, sigma)
    return base_score * float(norm_value)


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


class TermGraphValue(object):
    """
    Metrics to weigh Multi-Word Terms(MWTs)
    """
    def __init__(self, weight_comb="norm_max", mu=5, parallel_workers=1):
        self._logger = logging.getLogger("jgtextrank.metrics")
        self._logger.info(self.__class__.__name__)
        self.parallel_workers = parallel_workers
        self.weight_comb = weight_comb
        self.mu = mu

    @staticmethod
    def g_value(collapsed_term, all_vertices, weight_comb="norm_sum", mu=5, **kwargs):
        final_score = float(0)
        log2a = 0
        avg_score = 0
        sum_score = 0
        max_score = 0
        sigma = 0
        if "sigma" in kwargs:
            sigma = kwargs["sigma"]

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

        return round(final_score, 5)

    def _is_top_t_vertices_connection(self, collapsed_term, top_t_vertices):
        """

        :type collapsed_term: list [of list [of string]]
        :param collapsed_term: list of tokenised terms collapsed from original context that will form Single-word term or Multi-word Term
        :param top_t_vertices: top T weighted vertices
        :return: True if the input contains any of top T vertex
        """
        return any(top_t_vertex[0] in collapsed_term for top_t_vertex in top_t_vertices)

    def _concatenate_terms(self, weighted_candidates):
        return dict((" ".join(tokenised_term), score) for tokenised_term, score in weighted_candidates)

    def _get_sigma_from_all_candidates(self, collapsed_terms):
        """
        compute standard deviation of term length in MWTs
        :param collapsed_terms: list, list of tokenised terms
        :rtype: ndarray
        :return: standard_deviation
        """
        all_terms_size = [len(collapsed_term) for collapsed_term in collapsed_terms]
        return np.std(all_terms_size)

    def weighing(self, all_candidates, all_vertices, top_t_vertices):
        if all_candidates is None or len(all_candidates) == 0:
            self._logger.info("No candidate found. Skip weighing.")
            return {}

        self._logger.info(" Total [%s] candidates to weigh...", len(all_candidates))

        sigma = 0
        if "norm" in self.weight_comb:
            sigma = self._get_sigma_from_all_candidates(all_candidates)

        with MultiprocPool(processes=int(self.parallel_workers)) as pool:
            optional_params = dict()
            optional_params["weight_comb"] = self.weight_comb
            optional_params["mu"] = self.mu
            if sigma != 0:
                optional_params["sigma"] = sigma

            weighted_all_candidates = pool.starmap(TermGraphValue.calculate,
                                                   [(candidate,all_candidates, all_vertices, optional_params) for candidate
                                                    in all_candidates if self._is_top_t_vertices_connection(candidate, top_t_vertices)])

        return self._concatenate_terms(weighted_all_candidates)


    @staticmethod
    def calculate(candidate_term, all_candidates, all_vertices, optional_params=None):
        if optional_params is None:
            optional_params = dict()

        weight_comb="norm_max"
        if "weight_comb" in optional_params:
            weight_comb = optional_params["weight_comb"]

        mu = 5
        if "mu" in optional_params:
            mu = optional_params["mu"]

        sigma = 0
        if "sigma" in optional_params:
            sigma = optional_params["sigma"]

        final_score = TermGraphValue.g_value(candidate_term, all_vertices,
                                             weight_comb, mu, sigma=sigma)

        return (candidate_term, final_score)


class GCValue(TermGraphValue):
    """
    Experimental metrics to weight MWTs
    """
    def __init__(self, weight_comb="len_log_norm_avg", mu=5, parallel_workers=1):
        super().__init__(weight_comb, mu, parallel_workers)

    @staticmethod
    def _get_longer_terms(term, all_candidates):
        """
        the number of candidate terms that contain current term

        Simply term normalisation is applied. Could be extended with "solr_term_normaliser"
        params:
            term, current term tokens
            all candidates: all candidates
        return longer term list
        """
        try:
            return [longer_term for longer_term in all_candidates
                    if term != longer_term and set(term).issubset(set(longer_term))]
        except AttributeError:
            import traceback
            _logger.error(traceback.format_exc())
            _logger.error("AttributeError when processing candidate term [%s]", term)
        return []

    def weighing(self, all_candidates, all_vertices, top_t_vertices):
        if all_candidates is None or len(all_candidates) == 0:
            self._logger.info("No candidate found. Skip weighing.")
            return {}

        self._logger.info(" Total [%s] candidates to weigh...", len(all_candidates))
        with MultiprocPool(processes=int(self.parallel_workers)) as pool:
            weighted_all_candidates = pool.starmap(GCValue.calculate,
                                                   [(candidate,all_candidates, all_vertices) for candidate
                                                    in all_candidates if self._is_top_t_vertices_connection(candidate, top_t_vertices)])

        self._logger.info(" all candidates gc-value computation is completed.")
        return super()._concatenate_terms(weighted_all_candidates)

    @staticmethod
    def _sum_ga_candidates(candidate_list, all_vertices):
        return sum([TermGraphValue.g_value(candidate, all_vertices, weight_comb="len_log_norm_avg") for candidate in candidate_list])

    @staticmethod
    def calculate(candidate_term, all_candidates, all_vertices, optional_params=None):
        if optional_params is None:
            optional_params = dict()

        longer_terms = GCValue._get_longer_terms(candidate_term, all_candidates)
        a = len(candidate_term)
        # log(a + 0.1) for unigrams smoothing
        log2a = math.log(a + 0.1, 2)
        g_a = TermGraphValue.g_value(candidate_term, all_vertices, weight_comb="len_log_norm_avg")

        if longer_terms:
            p_ta = len(longer_terms)
            sum_gb = GCValue._sum_ga_candidates(longer_terms, all_vertices)
            term_gcvalue = log2a * (g_a - (1 / p_ta) * sum_gb)
        else:
            term_gcvalue = log2a * g_a

        return (candidate_term, round(term_gcvalue, 5))