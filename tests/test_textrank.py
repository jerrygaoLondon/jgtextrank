import unittest
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'jgtextrank'))
sys.path.append(os.path.join(os.path.dirname(__file__)))

import types
import warnings

from collections import Counter
import networkx as nx
import matplotlib as mpl

if os.environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')

import matplotlib.pyplot as plt

from jgtextrank.utility import sort_dict_by_value, flatten
from jgtextrank.core import preprocessing, preprocessing_tokenised_context, _syntactic_filter, \
    _get_cooccurs_from_single_context, _get_cooccurs, build_cooccurrence_graph, \
    _build_vertices_representations, keywords_extraction, _is_top_t_vertices_connection, _collapse_adjacent_keywords


def ignore_warnings(test_func):
    def do_test(self, *args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ResourceWarning)
            test_func(self, *args, **kwargs)
    return do_test


class TestTextRank(unittest.TestCase):

    def test_syntactic_filtering(self):
        tagged_abstract_context_list = [[('Compatibility', 'NN'), ('of', 'IN'), ('systems', 'NNS'), ('of', 'IN'),
                                  ('linear', 'JJ'), ('constraints', 'NNS'), ('over', 'IN'), ('the', 'DT'),
                                  ('set', 'NN'), ('of', 'IN'), ('natural', 'JJ'), ('numbers', 'NNS'),
                                  ('.', '.')], [('Criteria', 'NNP'), ('of', 'IN'), ('compatibility', 'NN'),
                                  ('of', 'IN'), ('a', 'DT'), ('system', 'NN'), ('of', 'IN'), ('linear', 'JJ'),
                                  ('Diophantine', 'NNP'), ('equations', 'NNS'), (',', ','), ('strict', 'JJ'),
                                  ('inequations', 'NNS'), (',', ','), ('and', 'CC'), ('nonstrict', 'JJ'),
                                  ('inequations', 'NNS'), ('are', 'VBP'), ('considered', 'VBN'), ('.', '.')]]

        filtered_context_syntactic_units = _syntactic_filter(tagged_abstract_context_list)
        assert isinstance(filtered_context_syntactic_units, types.GeneratorType)

        all_filtered_context = []

        for context_syntactic_units in filtered_context_syntactic_units:
            assert isinstance(context_syntactic_units, list)
            all_filtered_context.append(context_syntactic_units)

        flattened_all_filtered_context = flatten(all_filtered_context)

        assert len(flattened_all_filtered_context) == 17
        assert ('of', 'IN') not in flattened_all_filtered_context
        assert ('.', '.') not in flattened_all_filtered_context
        assert ('a', 'DT') not in flattened_all_filtered_context
        assert ('and', 'CC') not in flattened_all_filtered_context
        assert ('Compatibility', 'NN') in flattened_all_filtered_context
        assert ('linear', 'JJ') in flattened_all_filtered_context
        assert ('considered', 'VBN') not in flattened_all_filtered_context

        tagged_abstract_context_list2 = [[('Compatibility', 'NN'), ('of', 'IN'), ('systems', 'NNS'), ('of', 'IN'),
                                         ('linear', 'JJ'), ('constraints', 'NNS'), ('over', 'IN'), ('the', 'DT'),
                                         ('set', 'NN'), ('of', 'IN'), ('natural', 'JJ'), ('numbers', 'NNS'),
                                         ('.', '.')], [('Criteria', 'NNP'), ('of', 'IN'), ('compatibility', 'NN'),
                                                       ('of', 'IN'), ('a', 'DT'), ('system', 'NN'), ('of', 'IN'), ('[', 'NN'), ('linear', 'JJ'),
                                                       ('Diophantine', 'NNP'), ('equations', 'NNS'), (']', 'NN'), (',', ','), ('strict', 'JJ'),
                                                       ('inequations', 'NNS'), (',', ','), ('and', 'CC'), ('nonstrict', 'JJ'),
                                                       ('inequations', 'NNS'), ('are', 'VBP'), ('considered', 'VBN'), ('.', '.')]]
        filtered_context_syntactic_units = _syntactic_filter(tagged_abstract_context_list2)
        assert isinstance(filtered_context_syntactic_units, types.GeneratorType)
        all_filtered_context = []

        for context_syntactic_units in filtered_context_syntactic_units:
            assert isinstance(context_syntactic_units, list)
            all_filtered_context.append(context_syntactic_units)

        flattened_all_filtered_context = flatten(all_filtered_context)
        assert len(flattened_all_filtered_context) == 17, "punctuations should be removed from filtered context"
        assert ('[', 'NN') not in flattened_all_filtered_context
        assert (']', 'NN') not in flattened_all_filtered_context

    def test_pre_processing(self):
        example_abstract = "Compatibility of systems of linear constraints over the set of natural numbers. " \
                           "Criteria of compatibility of a system of linear Diophantine equations, strict inequations, " \
                           "and nonstrict inequations are considered. Upper bounds for components of a minimal set of " \
                           "solutions and algorithms of construction of minimal generating sets of solutions for all " \
                           "types of systems are given. These criteria and the corresponding algorithms for " \
                           "constructing a minimal supporting set of solutions can be used in solving all the " \
                           "considered types systems and systems of mixed types."
        # original_tokenised_sentences, syntactic_units
        syntactic_filtered_context = preprocessing(example_abstract)
        assert isinstance(syntactic_filtered_context, types.GeneratorType)

        all_tokenised_context = []
        all_filtered_context = []
        for tokenised_context, context_syntactic_units in syntactic_filtered_context:
            assert isinstance(tokenised_context, list)
            assert isinstance(context_syntactic_units, list)
            assert len(tokenised_context) > 0
            assert len(context_syntactic_units) > 0
            assert isinstance(context_syntactic_units[0], tuple)
            all_tokenised_context.append(tokenised_context)
            all_filtered_context.append(context_syntactic_units)

        assert len(all_tokenised_context) == 4, "Context size should be 4. The default context is sentence level."
        assert len(all_filtered_context) == 4, "PoS filtered context should be 4. " \
                                               "The default context is sentence level."

        flatten_all_tokenised_context = flatten(all_tokenised_context)
        assert len(flatten_all_tokenised_context) == 91, "total tokens are 91"
        flatten_all_filtered_context = flatten(all_filtered_context)
        assert len(flatten_all_filtered_context) == 41, "total size of filtered context tokens are 41"

        check_filtered_context = [True if filtered_token[1] == 'NN' or filtered_token[1] == 'NNS'
                                            or filtered_token[1] == 'JJ' or filtered_token[1] == 'NNP'
                                    else False  for filtered_token in flatten_all_filtered_context]
        assert len(set(check_filtered_context)) == 1, "the default 'noun_adjective_filter' should be applied."

        assert "." not in flatten_all_filtered_context
        assert ('solutions', 'NNS') in flatten_all_filtered_context
        assert ('minimal', 'JJ') in flatten_all_filtered_context
        assert ('equations', 'NNS') in flatten_all_filtered_context

    def test_get_cooccurs_from_single_context(self):
        filtered_context = ['Compatibility', 'systems', 'linear', 'constraints', 'set', 'natural', 'numbers']
        syntactic_unit_1 = 'systems'

        cooccur_context_1_1 = _get_cooccurs_from_single_context(syntactic_unit_1,filtered_context)
        assert len(cooccur_context_1_1) == 3, "the number of co-occur words of 'systems' in windows=2 context should be 3"
        assert 'Compatibility' in cooccur_context_1_1, "Left side context window contains 'Compatibility'"
        assert 'linear' in cooccur_context_1_1
        assert 'constraints' in cooccur_context_1_1

        cooccur_context_1_2 = _get_cooccurs_from_single_context(syntactic_unit_1,filtered_context, window_size=1)
        assert len(cooccur_context_1_2) == 2, "the number of co-occur words of 'systems' in windows=1 context should be 2"
        assert 'Compatibility' in cooccur_context_1_2, "Left side context window contains 'Compatibility'"
        assert 'linear' in cooccur_context_1_2

        syntactic_unit_2 = 'Compatibility'
        cooccur_context_2_1 = _get_cooccurs_from_single_context(syntactic_unit_2, filtered_context, window_size=2)
        assert len(cooccur_context_2_1) == 2, "the number of co-occur words of 'Compatibility' in windows=2 context should be 2"
        assert 'systems' in cooccur_context_2_1
        assert 'linear' in cooccur_context_2_1

        syntactic_unit_3 = 'constraints'
        cooccur_context_3_1 = _get_cooccurs_from_single_context(syntactic_unit_3, filtered_context)
        assert len(cooccur_context_3_1) == 4
        assert 'linear' in cooccur_context_3_1
        assert 'systems' in cooccur_context_3_1
        assert 'set' in cooccur_context_3_1
        assert 'natural' in cooccur_context_3_1

        cooccur_context_3_2 = _get_cooccurs_from_single_context(syntactic_unit_3, filtered_context, window_size=3)
        assert len(cooccur_context_3_2) == 6
        assert 'Compatibility' in cooccur_context_3_2
        assert 'systems' in cooccur_context_3_2
        assert 'linear' in cooccur_context_3_2
        assert 'set' in cooccur_context_3_2
        assert 'natural' in cooccur_context_3_2
        assert 'numbers' in cooccur_context_3_2

        cooccur_context_3_3 = _get_cooccurs_from_single_context(syntactic_unit_3, filtered_context, window_size=4)
        assert len(cooccur_context_3_3) == 6
        assert 'Compatibility' in cooccur_context_3_3
        assert 'systems' in cooccur_context_3_3
        assert 'linear' in cooccur_context_3_3
        assert 'set' in cooccur_context_3_3
        assert 'natural' in cooccur_context_3_3
        assert 'numbers' in cooccur_context_3_3

        syntactic_unit_4 = 'numbers'
        cooccur_context_4_1 = _get_cooccurs_from_single_context(syntactic_unit_4,filtered_context)
        assert len(cooccur_context_4_1) == 2
        assert 'set' in cooccur_context_4_1
        assert 'natural' in cooccur_context_4_1

    def test_get_cooccurs(self):
        filtered_context_corpus = [['Compatibility', 'systems', 'linear', 'constraints', 'set', 'natural', 'numbers'],
                            ['criteria', 'corresponding', 'algorithms', 'minimal', 'supporting', 'set',
                             'solutions', 'solving', 'types', 'systems', 'systems', 'mixed', 'types']]

        syntactic_unit_1 = 'systems'
        all_cooccur_context_1_1 = _get_cooccurs(syntactic_unit_1, filtered_context_corpus)
        assert len(all_cooccur_context_1_1) == 7
        assert 'Compatibility' in all_cooccur_context_1_1
        assert 'linear' in all_cooccur_context_1_1
        assert 'constraints' in all_cooccur_context_1_1
        assert 'solving' in all_cooccur_context_1_1
        assert 'types' in all_cooccur_context_1_1
        assert 'mixed' in all_cooccur_context_1_1
        assert 'systems' in all_cooccur_context_1_1

        syntactic_unit_2 = 'numbers'
        all_cooccur_context_2_1 = _get_cooccurs(syntactic_unit_2, filtered_context_corpus)
        assert len(all_cooccur_context_2_1) == 2
        assert 'set' in all_cooccur_context_2_1
        assert 'natural' in all_cooccur_context_2_1

        syntactic_unit_3 = 'set'
        all_cooccur_context_3_1 = _get_cooccurs(syntactic_unit_3, filtered_context_corpus, window_size=1)
        assert len(all_cooccur_context_3_1) == 4
        assert 'constraints' in all_cooccur_context_3_1
        assert 'natural' in all_cooccur_context_3_1
        assert 'supporting' in all_cooccur_context_3_1
        assert 'solutions' in all_cooccur_context_3_1

        all_cooccur_context_3_2 = _get_cooccurs(syntactic_unit_3, filtered_context_corpus, window_size=2)
        assert len(all_cooccur_context_3_2) == 8
        assert 'linear' in all_cooccur_context_3_2
        assert 'constraints' in all_cooccur_context_3_2
        assert 'natural' in all_cooccur_context_3_2
        assert 'numbers' in all_cooccur_context_3_2
        assert 'minimal' in all_cooccur_context_3_2
        assert 'supporting' in all_cooccur_context_3_2
        assert 'solutions' in all_cooccur_context_3_2
        assert 'solving' in all_cooccur_context_3_2

        syntactic_unit_4 = 'criteria'
        all_cooccur_context_4_1 = _get_cooccurs(syntactic_unit_4, filtered_context_corpus)
        assert len(all_cooccur_context_4_1) == 2
        assert 'corresponding' in all_cooccur_context_4_1
        assert 'algorithms' in all_cooccur_context_4_1

    def test_get_cooccurs_with_raw_context(self):

        all_tokenised_context=[['Upper', 'bounds', 'for', 'components', 'of', 'a', 'minimal', 'set', 'of',
                               'solutions', 'and', 'algorithms', 'of', 'construction', 'of', 'minimal',
                               'generating', 'sets', 'of', 'solutions', 'for', 'all', 'types', 'of', 'systems',
                               'are', 'given', '.']]
        filtered_context_corpus = ['Upper', 'bounds', 'components', 'minimal', 'solutions', 'algorithms',
                                    'construction', 'minimal', 'generating', 'sets', 'solutions', 'types',
                                    'systems']

        syntactic_unit_1 = 'components'
        all_cooccur_context_1_1 = _get_cooccurs(syntactic_unit_1, all_tokenised_context,
                                                all_filtered_context_tokens=filtered_context_corpus)
        #print("'", syntactic_unit_1, "' cooccurs: ", all_cooccur_context_1_1)
        assert len(all_cooccur_context_1_1) == 1
        assert 'bounds' in all_cooccur_context_1_1

        #example with two occurrences in one context
        syntactic_unit_2 = 'solutions'
        all_cooccur_context_2_1 = _get_cooccurs(syntactic_unit_2, all_tokenised_context,
                                                all_filtered_context_tokens=filtered_context_corpus)
        #print("'", syntactic_unit_2, "' cooccurs: ", all_cooccur_context_2_1)
        assert len(all_cooccur_context_2_1) == 2, "'solutions' has two occcurrences in current context. " \
                                                  "It should have two co-occurred words in two places."
        assert 'algorithms' in all_cooccur_context_2_1
        assert 'sets' in all_cooccur_context_2_1

    def test_build_vertices_representations(self):
        #original_tokenised_text = ['Here', 'are', 'details', 'from', 'the', '13th', 'Rail', 'Steel',
        #                           'Campaign','.', 'I', 'have', 'checked', 'the', 'Hydrogen', 'values',
        #                           'reported', 'to', 'you', 'by', 'our', 'IBM', 'mainframe', 'messages', '.']
        filtered_context = ['details', 'rail', 'steel', 'campaign', 'hydrogen',
                            'values', 'ibm', 'mainframe']
        #cooccurrence window size
        window_size = 2
        vertices = _build_vertices_representations(filtered_context, conn_with_original_ctx=False, window_size=window_size)

        assert 8 == len(vertices)

        for i in range(0, len(vertices)):
            vertex = vertices[i]
            if 'rail' == vertex.word_type:
                rail_vertex = vertex

            if 'ibm' == vertex.word_type:
                ibm_vertex = vertex

            if 'mainframe' == vertex.word_type:
                mainframe_vertex = vertex

            if 'hydrogen' == vertex.word_type:
                hydrogen_vertex = vertex

        assert len(rail_vertex.co_occurs) == 3
        assert 'details' in rail_vertex.co_occurs
        assert 'steel' in rail_vertex.co_occurs
        assert 'campaign' in rail_vertex.co_occurs

        assert len(ibm_vertex.co_occurs) == 3
        assert 'mainframe' in ibm_vertex.co_occurs
        assert 'values' in ibm_vertex.co_occurs
        assert 'hydrogen' in ibm_vertex.co_occurs

        assert len(mainframe_vertex.co_occurs) == 2
        assert 'values' in mainframe_vertex.co_occurs
        assert 'ibm' in mainframe_vertex.co_occurs

        assert len(hydrogen_vertex.co_occurs) == 4
        assert 'steel' in hydrogen_vertex.co_occurs
        assert 'ibm' in hydrogen_vertex.co_occurs
        assert 'values' in hydrogen_vertex.co_occurs
        assert 'ibm' in hydrogen_vertex.co_occurs

    def test_build_cooccurrence_graph(self):
        # example abstract taken from [Mihalcea04]
        example_abstract = "Compatibility of systems of linear constraints over the set of natural numbers. " \
                           "Criteria of compatibility of a system of linear Diophantine equations, strict inequations, " \
                           "and nonstrict inequations are considered. Upper bounds for components of a minimal set of " \
                           "solutions and algorithms of construction of minimal generating sets of solutions for all " \
                           "types of systems are given. These criteria and the corresponding algorithms for " \
                           "constructing a minimal supporting set of solutions can be used in solving all the " \
                           "considered types systems and systems of mixed types."

        syntactic_filtered_context = preprocessing(example_abstract)

        cooccurrence_graph, original_tokenised_context = build_cooccurrence_graph(syntactic_filtered_context, conn_with_original_ctx=False)

        #print("len(cooccurrence_graph.nodes()): ", len(cooccurrence_graph.nodes()))
        assert 25 == len(cooccurrence_graph.nodes())

        pr = nx.pagerank(cooccurrence_graph, tol=0.0001)

        #import matplotlib.pyplot as plt
        #nx.draw_networkx(cooccurrence_graph, pos=None, arrows=True, with_labels=True)
        #plt.show()

        pr_counter = Counter(pr)
        top_t_vertices = pr_counter.most_common(10)
        print("top t vertices: ", top_t_vertices)
        assert 'set' == top_t_vertices[0][0]
        assert 'minimal' == top_t_vertices[1][0]
        assert 'solutions' == top_t_vertices[2][0]
        assert 'linear' == top_t_vertices[3][0]
        assert 'systems' == top_t_vertices[4][0]
        assert 'algorithms' == top_t_vertices[5][0]
        assert 'inequations' == top_t_vertices[6][0]
        assert 'strict' == top_t_vertices[7][0]
        assert 'types' == top_t_vertices[8][0]
        assert 'equations' == top_t_vertices[9][0]

    def test_syntactic_filtering_with_custom_filter(self):
        tagged_abstract_tokens = [[('Compatibility', 'NN'), ('of', 'IN'), ('systems', 'NNS'), ('of', 'IN'),
                                  ('linear', 'JJ'), ('constraints', 'NNS'), ('over', 'IN'), ('the', 'DT'),
                                  ('set', 'NN'), ('of', 'IN'), ('natural', 'JJ'), ('numbers', 'NNS'),
                                  ('.', '.'), ('Criteria', 'NNP'), ('of', 'IN'), ('compatibility', 'NN'),
                                  ('of', 'IN'), ('a', 'DT'), ('system', 'NN'), ('of', 'IN'), ('linear', 'JJ'),
                                  ('Diophantine', 'NNP'), ('equations', 'NNS'), (',', ','), ('strict', 'JJ'),
                                  ('inequations', 'NNS'), (',', ','), ('and', 'CC'), ('nonstrict', 'JJ'),
                                  ('inequations', 'NNS'), ('are', 'VBP'), ('considered', 'VBN'), ('.', '.')]]

        custom_filter = lambda t : filter(lambda a: a[1] == 'NNS' or a[1] == 'NNP' or a[1] == 'NN'
                                                    or a[1] == 'JJ' or a[1] == 'VBN', t)

        syntactic_units = _syntactic_filter(tagged_abstract_tokens, pos_filter=custom_filter)
        syntactic_units = list(syntactic_units)
        assert len(syntactic_units) == 1
        print("syntactic_units filtered  with custom filter from pre-tagged text:")
        print(syntactic_units)
        print("len(syntactic_units): ", len(syntactic_units))
        assert len(syntactic_units[0]) == 18, "filtered context token size should be 18."
        assert ('of', 'IN') not in syntactic_units[0]
        assert ('.', '.') not in syntactic_units[0]
        assert ('a', 'DT') not in syntactic_units[0]
        assert ('the', 'DT') not in syntactic_units[0]
        assert ('and', 'CC') not in syntactic_units[0]
        assert ('Compatibility', 'NN') in syntactic_units[0]
        assert ('linear', 'JJ') in syntactic_units[0]
        assert ('considered', 'VBN') in syntactic_units[0]

    def test_term_betweeness_ranking_via_cooccur_graph(self):
        example_abstract = "Compatibility of systems of linear constraints over the set of natural numbers. " \
                           "Criteria of compatibility of a system of linear Diophantine equations, strict inequations, " \
                           "and nonstrict inequations are considered. Upper bounds for components of a minimal set of " \
                           "solutions and algorithms of construction of minimal generating sets of solutions for all " \
                           "types of systems are given. These criteria and the corresponding algorithms for " \
                           "constructing a minimal supporting set of solutions can be used in solving all the " \
                           "considered types systems and systems of mixed types."
        preprocessed_corpus_context = preprocessing(example_abstract)
        cooccurrence_graph, original_tokenised_context = build_cooccurrence_graph(preprocessed_corpus_context)
        betweenness = nx.betweenness_centrality(cooccurrence_graph)

        #nx.draw_networkx(cooccurrence_graph, pos=None, arrows=True, with_labels=True)
        #plt.show()

        btweeness_ranked_terms = sort_dict_by_value(betweenness)
        print("ranked terms via betweenness: ", btweeness_ranked_terms)
        btweeness_ranked_terms = list(btweeness_ranked_terms)

        assert "linear" == btweeness_ranked_terms[0]
        assert "systems" == btweeness_ranked_terms[1]
        assert "equations" == btweeness_ranked_terms[2]
        assert "strict" == btweeness_ranked_terms[3]
        assert "set" == btweeness_ranked_terms[4]
        #assert "inequations" == btweeness_ranked_terms[5]
        #assert "compatibility" == btweeness_ranked_terms[6]

    def test_is_top_t_vertices_connection(self):
        top_t_vertices = [('numbers', 1.46), ('inequations', 1.45), ('linear', 1.29),
                          ('diophantine', 1.28), ('upper', 0.99), ('bounds', 0.99), ('strict', 0.77)]
        term_candidate_1 = "linear constrains"
        result_term_candidate_1 = _is_top_t_vertices_connection(term_candidate_1, top_t_vertices)
        assert result_term_candidate_1 is True, "'"+result_term_candidate_1+"' is a top T vertex connection"

        term_candidate_2 = "linear diophantine equations"
        result_term_candidate_2 = _is_top_t_vertices_connection(term_candidate_2, top_t_vertices)
        assert result_term_candidate_2 is True, "'"+result_term_candidate_2+"' is a top T vertex connection"

        term_candidate_3 = "natural numbers"
        result_term_candidate_3 = _is_top_t_vertices_connection(term_candidate_3, top_t_vertices)
        assert result_term_candidate_3 is True, "'"+result_term_candidate_3+"' is a top T vertex connection"

        term_candidate_4 = "nonstrict inequations"
        result_term_candidate_4 = _is_top_t_vertices_connection(term_candidate_4, top_t_vertices)
        assert result_term_candidate_4 is True, "'"+term_candidate_4+"' is a top T vertex connection"

        term_candidate_5 = "strict inequations"
        result_term_candidate_5 = _is_top_t_vertices_connection(term_candidate_5, top_t_vertices)
        assert result_term_candidate_5 is True, "'"+term_candidate_5+"' is a top T vertex connection"

        term_candidate_6 = "upper bounds"
        result_term_candidate_6 = _is_top_t_vertices_connection(term_candidate_6, top_t_vertices)
        assert result_term_candidate_6 is True, "'"+term_candidate_6+"' is a top T vertex connection"

        term_candidate_7 = "minimal generating sets"
        result_term_candidate_7 = _is_top_t_vertices_connection(term_candidate_7, top_t_vertices)
        assert result_term_candidate_7 is False, "'"+term_candidate_7+"' is NOT a top T vertex connection"

        term_candidate_8 = "solutions"
        result_term_candidate_8 = _is_top_t_vertices_connection(term_candidate_8, top_t_vertices)
        assert result_term_candidate_8 is False, "'"+term_candidate_8+"' is NOT a top T vertex connection"

        term_candidate_9 = "types systems"
        result_term_candidate_9 = _is_top_t_vertices_connection(term_candidate_9, top_t_vertices)
        assert result_term_candidate_9 is False, "'"+term_candidate_9+"' is NOT a top T vertex connection"

        term_candidate_10 = "algorithms"
        result_term_candidate_10 = _is_top_t_vertices_connection(term_candidate_10, top_t_vertices)
        assert result_term_candidate_10 is False, "'"+term_candidate_10+"' is NOT a top T vertex connection"

    def test_collapse_adjacent_keywords(self):
        weighted_keywords = {'sets': 0.03472, 'supporting': 0.03448, 'compatibility': 0.04089,
                             'components': 0.00643, 'minimal': 0.06524, 'algorithms': 0.05472, 'inequations': 0.04641,
                             'corresponding': 0.02194, 'numbers': 0.02379, 'systems': 0.083597, 'constraints': 0.02148,
                             'linear': 0.08849, 'natural': 0.040847, 'diophantine': 0.0370565, 'mixed': 0.03591,
                             'equations': 0.054968, 'strict': 0.041742, 'set': 0.066734, 'construction': 0.03580,
                             'system': 0.02148, 'types': 0.03591, 'criteria': 0.02381, 'upper': 0.00643,
                             'nonstrict': 0.026167, 'solutions': 0.050879}
        original_tokenised_text= ['compatibility', 'of', 'systems', 'of', 'linear', 'constraints', 'over',
                                  'the', 'set', 'of', 'natural', 'numbers', '.', 'criteria', 'of', 'compatibility',
                                  'of', 'a', 'system', 'of', 'linear', 'diophantine', 'equations', ',',
                                  'strict', 'inequations', ',', 'and', 'nonstrict', 'inequations', 'are',
                                  'considered', '.', 'upper', 'bounds', 'for', 'components', 'of', 'a',
                                  'minimal', 'set', 'of', 'solutions', 'and', 'algorithms', 'of',
                                  'construction', 'of', 'minimal', 'generating', 'sets', 'of', 'solutions',
                                  'for', 'all', 'types', 'of', 'systems', 'are', 'given', '.', 'these',
                                  'criteria', 'and', 'the', 'corresponding', 'algorithms', 'for', 'constructing',
                                  'a', 'minimal', 'supporting', 'set', 'of', 'solutions', 'can', 'be', 'used',
                                  'in', 'solving', 'all', 'the', 'considered', 'types', 'systems', 'and',
                                  'systems', 'of', 'mixed', 'types', '.']
        key_terms = _collapse_adjacent_keywords(weighted_keywords, original_tokenised_text)
        print("key terms collapsed from context: ", key_terms)
        assert len(key_terms) == 29
        assert key_terms[0][0] == 'compatibility'
        assert key_terms[1][0] == 'systems'
        assert key_terms[2][0] == 'linear'
        assert key_terms[2][1] == 'constraints'
        assert key_terms[3][0] == 'set'
        assert key_terms[4][0] == 'natural'
        assert key_terms[4][1] == 'numbers'
        assert key_terms[5][0] == 'criteria'

        S0021999113005652_weighted_keywords = {'degradation': 0.03048, 'future': 0.004573, 'result': 0.004573,
                                               'exchange': 0.03367, 'progress': 0.004573, 'important': 0.03048,
                                               'modelling': 0.030487, 'extensive': 0.03048, 'reynolds': 0.02551,
                                               'figure': 0.004573170731707318, 'datum': 0.004573, 'impact': 0.03048,
                                               'study': 0.00457, 'function': 0.004573, 'environmental': 0.0304878,
                                               'effect': 0.030487, 'air': 0.03070, 'flow': 0.016393,
                                               'schmidt': 0.02551, 'fig': 0.030487, 'turbulent': 0.004573,
                                               'rate': 0.024854, 'chemical': 0.03582, 'number': 0.036786,
                                               'interface': 0.0045731, 'reaction': 0.047672, 'depict': 0.0304878,
                                               'practical': 0.03048, 'interesting': 0.004573,
                                               'investigation': 0.0304878, 'concentration': 0.0304878,
                                               'worth': 0.0045731, 'increase': 0.04951, 'bulk': 0.00457,
                                               'water': 0.055614, 'efficiency': 0.015095, 'equilibrium': 0.030487,
                                               'product': 0.030487, 'aquarium': 0.0248545,
                                               'by(24)cb⁎ =∫01〈cb⁎〉(z⁎)dz⁎': 0.030487, 'acidification': 0.016393,
                                               'gas': 0.018886, 'information': 0.03048}
        S0021999113005652_tokenised_text = ['it', 'be', 'interesting', 'to', 'quantify', 'the', 'effect',
                                            'of', 'the', 'schmidt', 'number', 'and', 'the', 'chemical',
                                            'reaction', 'rate', 'on', 'the', 'bulk', '-', 'mean', 'concentration',
                                            'of', 'b', 'in', 'water', '.', 'the', 'datum', 'could', 'present',
                                            'important', 'information', 'on', 'evaluate', 'the', 'environmental',
                                            'impact', 'of', 'the', 'degradation', 'product', 'of', 'b', ',',
                                            'as', 'well', 'as', 'acidification', 'of', 'water', 'by', 'the',
                                            'chemical', 'reaction', '.', 'here', ',', 'the', 'bulk', '-',
                                            'mean', 'concentration', 'of', 'b', 'be', 'define',
                                            'by(24)cb⁎ =∫01〈cb⁎〉(z⁎)dz⁎', 'fig', '.', '15', 'depict', 'the',
                                            'effect', 'of', 'the', 'schmidt', 'and', 'the', 'chemical',
                                            'reaction', 'rate', 'on', 'the', 'bulk', '-', 'mean',
                                            'concentration', 'cb⁎ .', 'it', 'be', 'worth', 'to', 'mention',
                                            'here', 'that', 'the', 'bulk', '-', 'mean', 'concentration', 'of',
                                            'b', 'reach', 'approximately', '0.6', 'as', 'the', 'chemical',
                                            'reaction', 'rate', 'and', 'the', 'schmidt', 'number', 'increase',
                                            'to', 'infinite', ',', 'and', 'the', 'concentration', 'be',
                                            'small', 'than', 'the', 'equilibrium', 'concentration', 'of', 'a',
                                            'at', 'the', 'interface', '.', 'this', 'figure', 'indicate',
                                            'that', 'progress', 'of', 'the', 'chemical', 'reaction', 'be',
                                            'somewhat', 'interfere', 'by', 'turbulent', 'mix', 'in', 'water',
                                            ',', 'and', 'the', 'efficiency', 'of', 'the', 'chemical',
                                            'reaction', 'be', 'up', 'to', 'approximately', '60', '%', '.',
                                            'the', 'efficiency', 'of', 'the', 'chemical', 'reaction', 'in',
                                            'water', 'will', 'be', 'a', 'function', 'of', 'the', 'reynolds',
                                            'number', 'of', 'the', 'water', 'flow', ',', 'and', 'the',
                                            'efficiency', 'could', 'increase', 'as', 'the', 'reynolds',
                                            'number', 'increase', '.', 'we', 'need', 'an', 'extensive',
                                            'investigation', 'on', 'the', 'efficiency', 'of', 'the', 'aquarium',
                                            'chemical', 'reaction', 'in', 'the', 'near', 'future', 'to', 'extend',
                                            'the', 'result', 'of', 'this', 'study', 'further', 'to', 'establish',
                                            'practical', 'modelling', 'for', 'the', 'gas', 'exchange',
                                            'between', 'air', 'and', 'water', '.']
        S0021999113005652_key_terms = _collapse_adjacent_keywords(S0021999113005652_weighted_keywords, S0021999113005652_tokenised_text)
        print("S0021999113005652_key_terms: ", S0021999113005652_key_terms)
        assert len(S0021999113005652_key_terms) == 57
        assert S0021999113005652_key_terms[0][0] == "interesting"
        assert S0021999113005652_key_terms[1][0] == "effect"
        assert S0021999113005652_key_terms[2][0] == "schmidt"
        assert S0021999113005652_key_terms[2][1] == "number"
        assert S0021999113005652_key_terms[3][0] == "chemical"
        assert S0021999113005652_key_terms[3][1] == "reaction"
        assert S0021999113005652_key_terms[3][2] == "rate"
        assert S0021999113005652_key_terms[4][0] == "bulk"
        assert S0021999113005652_key_terms[5][0] == "concentration"
        assert S0021999113005652_key_terms[6][0] == "water"
        assert S0021999113005652_key_terms[7][0] == "datum"
        assert S0021999113005652_key_terms[8][0] == "important"
        assert S0021999113005652_key_terms[8][1] == "information"
        assert S0021999113005652_key_terms[9][0] == "environmental"
        assert S0021999113005652_key_terms[9][1] == "impact"
        assert S0021999113005652_key_terms[16][0] == "by(24)cb⁎ =∫01〈cb⁎〉(z⁎)dz⁎"
        assert S0021999113005652_key_terms[16][1] == "fig"

    @ignore_warnings
    def test_keywords_extraction(self):
        example_abstract = "Compatibility of systems of linear constraints over the set of natural numbers. " \
                           "Criteria of compatibility of a system of linear Diophantine equations, strict inequations, " \
                           "and nonstrict inequations are considered. Upper bounds for components of a minimal set of " \
                           "solutions and algorithms of construction of minimal generating sets of solutions for all " \
                           "types of systems are given. These criteria and the corresponding algorithms for " \
                           "constructing a minimal supporting set of solutions can be used in solving all the " \
                           "considered types systems and systems of mixed types."

        results, top_vertices = keywords_extraction(example_abstract, top_p = 0.3, weight_comb="sum")
        print("extracted keywords:"+ str(results))
        print("top_vertices: ", top_vertices)

        assert 13 == len(results)

        term_list = [term[0] for term in results]
        assert "linear diophantine equations" == term_list[0]
        assert "minimal supporting set" == term_list[1]
        assert "minimal set" == term_list[2]
        assert "types systems" == term_list[3]
        assert "linear constraints" == term_list[4]
        assert "strict inequations" == term_list[5]
        assert "systems" == term_list[6]
        assert "corresponding algorithms" == term_list[7]
        assert "nonstrict inequations" == term_list[8]
        assert "set" in term_list
        assert "minimal" in term_list
        assert "algorithms" in term_list
        assert "solutions" in term_list
        assert "natural numbers" not in term_list

        assert 'linear' == top_vertices[0][0]
        assert 'systems' == top_vertices[1][0]
        assert 'set' == top_vertices[2][0]
        assert 'minimal' == top_vertices[3][0]
        assert 'equations' == top_vertices[4][0]
        assert 'algorithms' == top_vertices[5][0]
        assert 'solutions' == top_vertices[6][0]
        assert 'inequations' == top_vertices[7][0]

        print("after enabling lemmatization....")
        results, top_vertices = keywords_extraction(example_abstract, top_p = 0.3, lemma=True, weight_comb="sum")

        assert 12 == len(results)
        print("extracted keywords after lemmatization: ", results)
        print("top_vertices after lemmatization: ", top_vertices)

        term_list = [term[0] for term in results]
        assert "minimal supporting set" == term_list[0]
        assert "linear diophantine equation" == term_list[1]
        assert "minimal set" == term_list[2]
        assert "type system" == term_list[3]
        assert "linear constraint" == term_list[4]
        assert "strict inequations" == term_list[5]
        assert "system" == term_list[6]
        assert "corresponding algorithm" == term_list[7]
        assert "nonstrict inequations" == term_list[8]

        assert 'system' == top_vertices[0][0]
        assert 'set' == top_vertices[1][0]
        assert 'linear' == top_vertices[2][0]
        assert 'algorithm' == top_vertices[3][0]
        assert 'equation' == top_vertices[4][0]
        assert 'minimal' == top_vertices[5][0]
        assert 'inequations' == top_vertices[6][0]

    def test_keywords_extraction2(self):
        """
        test keywords extraction with example nodes (with custom syntactic filters  and step list) in the paper
        """
        example_abstract = "Compatibility of systems of linear constraints over the set of natural numbers. " \
                           "Criteria of compatibility of a system of linear Diophantine equations, strict inequations, " \
                           "and nonstrict inequations are considered. Upper bounds for components of a minimal set of " \
                           "solutions and algorithms of construction of minimal generating sets of solutions for all " \
                           "types of systems are given. These criteria and the corresponding algorithms for " \
                           "constructing a minimal supporting set of solutions can be used in solving all the " \
                           "considered types systems and systems of mixed types."

        custom_categories = {'NNS', 'NNP', 'NN', 'JJ', 'VBZ'}
        # manually filter few nodes not appearing in the given example of original paper
        stop_words={'set', 'mixed', 'corresponding', 'supporting'}
        ranked_terms, top_vertices = keywords_extraction(example_abstract, top_p = 1, top_t=None, directed=False,
                                           syntactic_categories=custom_categories, stop_words=stop_words, weight_comb="sum")

        print("ranked terms with custom filters 1: ", ranked_terms)
        print("top_vertices with custom filters 1: ", top_vertices)

        top_vertices_names = [top_vertex[0] for top_vertex in top_vertices]
        assert 'supporting' not in top_vertices_names
        assert 'corresponding' not in top_vertices_names
        assert 'mixed' not in top_vertices_names
        assert 'set' not in top_vertices_names

        assert 'linear diophantine equations' == ranked_terms[0][0]
        assert 'linear constraints' == ranked_terms[1][0]
        assert 'types systems' == ranked_terms[2][0]
        assert 'upper bounds' == ranked_terms[3][0]
        assert 'strict inequations' == ranked_terms[4][0]
        assert 'natural numbers' == ranked_terms[5][0]
        assert 'systems' == ranked_terms[6][0]
        assert 'nonstrict inequations' == ranked_terms[7][0]
        assert 'compatibility' == ranked_terms[8][0]
        assert 'construction' == ranked_terms[9][0] or 'minimal' == ranked_terms[9][0] \
               or 'algorithms' ==  ranked_terms[9][0] or 'solutions'  == ranked_terms[9][0] \
               or 'sets' == ranked_terms[9][0]
        # >>> [('linear diophantine equations', 0.19805), ('linear constraints', 0.12147),
        #       ('types systems', 0.10493), ('upper bounds', 0.10114), ('strict inequations', 0.09432),
        #       ('natural numbers', 0.09091), ('systems', 0.08092), ('nonstrict inequations', 0.07741),
        #       ('compatibility', 0.04666), ('algorithms', 0.04545), ('minimal', 0.04545),
        #       ('construction', 0.04545), ('sets', 0.04545), ('solutions', 0.04545),
        #       ('components', 0.03522), ('criteria', 0.02665), ('types', 0.02401), ('system', 0.02348)]

        stop_words={'set', 'mixed', 'corresponding', 'supporting', "minimal"}
        ranked_terms, top_vertices = keywords_extraction(example_abstract, top_p = 1, top_t=None, directed=False,
                                                         syntactic_categories=custom_categories, stop_words=stop_words)
        print("ranked terms with custom filters 2: ", ranked_terms)
        print("top_vertices with custom filters 2: ", top_vertices)

        top_vertices_names = [top_vertex[0] for top_vertex in top_vertices]
        assert 'minimal' not in top_vertices_names
        assert 'supporting' not in top_vertices_names
        assert 'corresponding' not in top_vertices_names
        assert 'mixed' not in top_vertices_names
        assert 'set' not in top_vertices_names
        # [('linear diophantine equations', 0.20748), ('linear constraints', 0.12726), ('types systems', 0.10992),
        # ('upper bounds', 0.10596), ('strict inequations', 0.09881), ('natural numbers', 0.09524),
        # ('systems', 0.08477), ('nonstrict inequations', 0.0811), ('solutions', 0.06182), ('algorithms', 0.06182),
        # ('compatibility', 0.04889), ('components', 0.0369), ('sets', 0.03342), ('construction', 0.03342),
        # ('criteria', 0.02792), ('types', 0.02516), ('system', 0.02459)]

    def test_keywords_extraction3(self):
        """
        test with different pagerank algorithms
        """
        example_abstract = "Compatibility of systems of linear constraints over the set of natural numbers. " \
                           "Criteria of compatibility of a system of linear Diophantine equations, strict inequations, " \
                           "and nonstrict inequations are considered. Upper bounds for components of a minimal set of " \
                           "solutions and algorithms of construction of minimal generating sets of solutions for all " \
                           "types of systems are given. These criteria and the corresponding algorithms for " \
                           "constructing a minimal supporting set of solutions can be used in solving all the " \
                           "considered types systems and systems of mixed types."

        results, top_vertices = keywords_extraction(example_abstract, top_p = 0.3, solver="pagerank_numpy", weight_comb="sum")
        print("ranked terms computed with 'pagerank_numpy': ", results)
        print("top_vertices computed with 'pagerank_numpy': ", top_vertices)

        assert len(results) == 13
        results, top_vertices = keywords_extraction(example_abstract, top_p = 0.3, solver="pagerank_scipy", weight_comb="sum")
        print("ranked terms computed with 'pagerank_scipy': ", results)
        print("top_vertices computed with 'pagerank_scipy': ", top_vertices)

        assert len(results) == 13

        results, top_vertices = keywords_extraction(example_abstract, top_p = 0.3, solver="betweenness_centrality", weight_comb="sum")
        print("ranked terms computed with 'betweenness_centrality': ", results)
        print("top_vertices computed with 'betweenness_centrality': ", top_vertices)
        assert len(results) == 11

        results, top_vertices = keywords_extraction(example_abstract, top_p = 0.3, solver="degree_centrality", weight_comb="sum")
        print("ranked terms computed with 'degree_centrality': ", results)
        print("top_vertices computed with 'degree_centrality': ", top_vertices)
        assert top_vertices[0][0] == 'systems'
        assert top_vertices[1][0] == 'linear'
        assert top_vertices[2][0] == 'minimal' or top_vertices[2][0] == 'set'
        # top 30% results is not stable for degree_centrality
        # assert len(results) == 11 or len(results) == 12

        results, top_vertices = keywords_extraction(example_abstract, top_p = 0.3, solver="hits", weight_comb="sum")
        print("ranked terms computed with 'hits': ", results)
        print("top_vertices computed with 'hits': ", top_vertices)
        assert top_vertices[0][0] == 'systems'
        assert top_vertices[1][0] == 'linear'
        assert top_vertices[2][0] == 'mixed' or top_vertices[2][0] == 'types'
        assert top_vertices[4][0] == 'equations'
        assert len(results) == 7 or len(results) == 8

        results, top_vertices = keywords_extraction(example_abstract, top_p = 0.3, solver="closeness_centrality", weight_comb="sum")
        print("ranked terms computed with 'closeness_centrality': ", results)
        print("top_vertices computed with 'closeness_centrality': ", top_vertices)
        assert len(results) == 10 or len(results) == 11

        results, top_vertices = keywords_extraction(example_abstract, top_p = 0.3, solver="edge_betweenness_centrality", weight_comb="sum")
        print("ranked terms computed with 'edge_betweenness_centrality': ", results)
        print("top_vertices computed with 'edge_betweenness_centrality': ", top_vertices)
        assert len(results) == 8 or len(results) == 10

        results, top_vertices = keywords_extraction(example_abstract, top_p = 0.3, solver="eigenvector_centrality", max_iter=1000, weight_comb="sum")
        print("ranked terms computed with 'eigenvector_centrality': ", results)
        print("top_vertices computed with 'eigenvector_centrality': ", top_vertices)
        assert len(results) == 7 or len(results) == 8

        results, top_vertices = keywords_extraction(example_abstract, top_p = 0.3, solver="katz_centrality", weight_comb="sum")
        print("ranked terms computed with 'katz_centrality': ", results)
        print("top_vertices computed with 'katz_centrality': ", top_vertices)
        assert len(results) == 11

        results, top_vertices = keywords_extraction(example_abstract, top_p = 0.3, solver="communicability_betweenness",
                            window=5, weighted=False, weight_comb="sum")
        print("ranked terms computed with 'communicability_betweenness': ", results)
        print("top_vertices computed with 'communicability_betweenness': ", top_vertices)
        print(len(results))
        assert results[0][0] == 'minimal supporting set'
        assert results[1][0] == 'minimal set'
        assert results[2][0] == 'linear diophantine equations'
        assert len(results) == 12

        results, top_vertices = keywords_extraction(example_abstract, top_p = 0.3, solver="current_flow_closeness",
                                                    weighted=False, weight_comb="sum")
        print("ranked terms computed with 'current_flow_closeness': ", results)
        print("top_vertices computed with 'current_flow_closeness': ", top_vertices)
        print(len(results))
        assert len(results) == 9
        assert results[0][0] == 'minimal supporting set'
        assert results[1][0] == 'minimal set'
        assert top_vertices[0][0] == 'set'
        assert top_vertices[1][0] == 'minimal'

        results, top_vertices = keywords_extraction(example_abstract, top_p = 0.3, solver="current_flow_betweenness",
                                                    weighted=False, weight_comb="sum")
        print("ranked terms computed with 'current_flow_betweenness': ", results)
        print("top_vertices computed with 'current_flow_betweenness': ", top_vertices)
        print(len(results))
        assert len(results) == 11
        assert top_vertices[0][0] == 'systems'
        assert top_vertices[1][0] == 'linear'
        assert top_vertices[2][0] == 'set'

        results, top_vertices = keywords_extraction(example_abstract, top_p = 0.3, solver="edge_current_flow_betweenness",
                                                    weighted=False, weight_comb="sum")
        print("ranked terms computed with 'edge_current_flow_betweenness': ", results)
        print("top_vertices computed with 'edge_current_flow_betweenness': ", top_vertices)
        print(len(results))
        assert len(results) == 10 or len(results) == 11
        assert top_vertices[0][0] == 'systems' or top_vertices[0][0] == 'linear'
        assert top_vertices[1][0] == 'linear' or top_vertices[1][0] == 'systems'
        assert top_vertices[2][0] == 'strict' or top_vertices[2][0] == 'equations'

        results, top_vertices = keywords_extraction(example_abstract, top_p = 0.3, solver="load_centrality",
                                                    weighted=False, weight_comb="sum")
        print("ranked terms computed with 'load_centrality': ", results)
        print("top_vertices computed with 'load_centrality': ", top_vertices)
        print(len(results))
        assert len(results) == 11
        assert results[0][0] == 'linear diophantine equations'
        assert results[1][0] == 'linear constraints'
        assert results[2][0] == 'systems' or results[2][0] == 'types systems'
        assert top_vertices[0][0] == 'linear'
        assert top_vertices[1][0] == 'systems'
        assert top_vertices[2][0] == 'equations'

        results, top_vertices = keywords_extraction(example_abstract, top_p = 0.3, solver="clustering_coefficient",
                                                    weighted=False, weight_comb="sum")
        print("ranked terms computed with 'clustering_coefficient': ", results)
        print("top_vertices computed with 'clustering_coefficient': ", top_vertices)
        assert results[0][0] == 'mixed types'
        assert results[1][0] == 'linear diophantine equations'
        assert results[2][0] == 'minimal supporting set'
        assert len(results) == 9

        results, top_vertices = keywords_extraction(example_abstract, top_p = 0.3, solver="TeRGraph",
                            weighted=False, weight_comb="sum")
        print("ranked terms computed with 'TeRGraph': ", results)
        print("top_vertices computed with 'TeRGraph': ", top_vertices)
        assert len(results) == 7
        assert results[0][0] == 'nonstrict inequations'
        assert results[1][0] == 'natural numbers'
        assert results[2][0] == 'corresponding algorithms'

        coreness_results, coreness_top_vertices = keywords_extraction(example_abstract, top_p = 1, solver="coreness",
                                                    weighted=False, weight_comb="sum")
        print("ranked terms computed with 'coreness': ", coreness_results)
        print("top_vertices computed with 'coreness': ", coreness_top_vertices)
        coreness_results_dict = {k:v for k, v in coreness_results}
        coreness_top_vertices_dict = {k:v for k, v in coreness_top_vertices}

        assert  len(coreness_results) == 23
        assert coreness_results_dict['minimal supporting set'] == 6
        assert coreness_results_dict['linear diophantine equations'] == 6
        assert coreness_results_dict['types systems'] == 4
        assert coreness_results_dict['minimal set'] == 4
        assert coreness_top_vertices_dict['minimal'] == 2
        assert coreness_top_vertices_dict['sets'] == 2
        assert coreness_top_vertices_dict['diophantine'] == 2
        assert coreness_top_vertices_dict['equations'] == 2
        assert coreness_top_vertices_dict['criteria'] == 1
        assert coreness_top_vertices_dict['upper'] == 0
        assert coreness_top_vertices_dict['components'] == 0

        mean_coreness_results, coreness_top_vertices = keywords_extraction(example_abstract, top_p = 1, solver="coreness",
                                                                      weighted=False, weight_comb="avg")
        print("ranked term phrases computed with Mean coreness: ", mean_coreness_results)
        mean_coreness_results_dict = {k:v for k, v in mean_coreness_results}
        assert mean_coreness_results_dict['types'] == 2
        assert mean_coreness_results_dict['minimal supporting set'] == 2
        assert mean_coreness_results_dict['components'] == 0
        assert mean_coreness_results_dict['linear diophantine equations'] == 2


        with self.assertRaises(ValueError) as context:
            keywords_extraction(example_abstract, top_p = 0.3, solver="my_pagerank")

            self.assertTrue("The node weighting solver supports only pagerank, "
                            "pagerank_numpy, pagerank_scipy, betweenness_centrality, "
                            "edge_betweenness_centrality, degree_centrality, closeness_centrality, hits, "
                            "eigenvector_centrality, katz_centrality, communicability_betweenness, "
                            "current_flow_closeness, current_flow_betweenness, edge_current_flow_betweenness, "
                            "load_centrality,clustering_coefficient,TeRGraph,coreness got 'my_pagerank'" in context.exception)

    def test_neighborhood_size(self):
        example_abstract = "Compatibility of systems of linear constraints over the set of natural numbers. " \
                           "Criteria of compatibility of a system of linear Diophantine equations, strict inequations, " \
                           "and nonstrict inequations are considered. Upper bounds for components of a minimal set of " \
                           "solutions and algorithms of construction of minimal generating sets of solutions for all " \
                           "types of systems are given. These criteria and the corresponding algorithms for " \
                           "constructing a minimal supporting set of solutions can be used in solving all the " \
                           "considered types systems and systems of mixed types."

        mean_neighbors_results, mean_neighbors_vertices = keywords_extraction(example_abstract, top_p = 1, solver="neighborhood_size",
                                                                              weighted=False, weight_comb="avg")
        print("ranked term phrases computed with Mean neighborhood size: ", mean_neighbors_results)
        mean_neighbors_results_dict = {k:v for k, v in mean_neighbors_results}
        mean_neighbors_vertices_dict = {k:v for k, v in mean_neighbors_vertices}
        print(len(mean_neighbors_results))
        assert len(mean_neighbors_results) == 23
        assert mean_neighbors_results_dict["set"] == 4.0
        assert mean_neighbors_results_dict["minimal"] == 4.0
        assert mean_neighbors_results_dict["minimal set"] == 4.0
        assert mean_neighbors_results_dict["linear constraints"] == 3.0
        assert mean_neighbors_results_dict["solutions"] == 3.0
        assert mean_neighbors_results_dict["nonstrict inequations"] == 1.5
        assert mean_neighbors_results_dict["linear diophantine equations"] == 3.33333
        print(mean_neighbors_vertices_dict)
        assert mean_neighbors_vertices_dict["linear"] == 5
        assert mean_neighbors_vertices_dict["set"] == 4
        assert mean_neighbors_vertices_dict["systems"] == 4
        assert mean_neighbors_vertices_dict["minimal"] == 4
        assert mean_neighbors_vertices_dict["algorithms"] == 3
        assert mean_neighbors_vertices_dict["compatibility"] == 2

    def test_keywords_extraction_with_mwt_scoring(self):
        example_abstract = "Compatibility of systems of linear constraints over the set of natural numbers. " \
                           "Criteria of compatibility of a system of linear Diophantine equations, strict inequations, " \
                           "and nonstrict inequations are considered. Upper bounds for components of a minimal set of " \
                           "solutions and algorithms of construction of minimal generating sets of solutions for all " \
                           "types of systems are given. These criteria and the corresponding algorithms for " \
                           "constructing a minimal supporting set of solutions can be used in solving all the " \
                           "considered types systems and systems of mixed types."

        results, top_vertices = keywords_extraction(example_abstract, top_p = 0.3, weight_comb="avg")
        print("extracted keywords with avg weighting:"+ str(results))

        term_list = [term[0] for term in results]
        assert "systems" == term_list[0]
        assert "set" == term_list[1]
        assert "minimal set" == term_list[2]
        assert "minimal" == term_list[3]
        assert "linear diophantine equations" == term_list[4]
        assert "types systems" == term_list[5]
        assert "minimal supporting set" == term_list[6]
        assert "linear constraints" == term_list[7]
        assert "algorithms" == term_list[8]

        results, top_vertices = keywords_extraction(example_abstract, top_p = 0.3, weight_comb="norm_avg")
        print("extracted keywords with norm_avg weighting:"+ str(results))

        term_list = [term[0] for term in results]
        assert "systems" == term_list[0]
        assert "set" == term_list[1]
        assert "minimal" == term_list[2]
        assert "algorithms" == term_list[3]
        assert "solutions" == term_list[4]
        assert "minimal set" == term_list[5]
        assert "types systems" == term_list[6]
        assert "linear constraints" == term_list[7]
        assert "strict inequations" == term_list[8]

        results, top_vertices = keywords_extraction(example_abstract, top_p = 0.3, weight_comb="log_norm_avg")
        print("extracted keywords with log_norm_avg weighting:"+ str(results))

        term_list = [term[0] for term in results]
        assert "minimal set" == term_list[0]
        assert "types systems" == term_list[1]
        assert "linear constraints" == term_list[2]
        assert "strict inequations" == term_list[3]
        assert "corresponding algorithms" == term_list[4]
        assert "linear diophantine equations" == term_list[5]
        assert "nonstrict inequations" == term_list[6]
        assert "systems" == term_list[7]
        assert "minimal supporting set" == term_list[8]

        results, top_vertices = keywords_extraction(example_abstract, top_p = 0.3, weight_comb="gaussian_norm_avg")
        print("extracted keywords with gaussian_norm_avg weighting:"+ str(results))

        term_list = [term[0] for term in results]
        assert "systems" == term_list[0]
        assert "set" == term_list[1]
        assert "minimal set" == term_list[2]
        assert "minimal" == term_list[3]
        assert "linear diophantine equations" == term_list[4]
        assert "types systems" == term_list[5]
        assert "minimal supporting set" == term_list[6]
        assert "linear constraints" == term_list[7]
        assert "algorithms" == term_list[8]

        results, top_vertices = keywords_extraction(example_abstract, top_p = 0.3, weight_comb="sum")
        print("extracted keywords with sum weighting:"+ str(results))

        term_list = [term[0] for term in results]
        assert "linear diophantine equations" == term_list[0]
        assert "minimal supporting set" == term_list[1]
        assert "minimal set" == term_list[2]
        assert "types systems" == term_list[3]
        assert "linear constraints" == term_list[4]
        assert "strict inequations" == term_list[5]
        assert "systems" == term_list[6]
        assert "corresponding algorithms" == term_list[7]
        assert "nonstrict inequations" == term_list[8]

        results, top_vertices = keywords_extraction(example_abstract, top_p = 0.3, weight_comb="norm_sum")
        print("extracted keywords with norm_sum weighting:"+ str(results))

        term_list = [term[0] for term in results]
        assert "systems" == term_list[0]
        assert "set" == term_list[1]
        assert "minimal set" == term_list[2]
        assert "minimal" == term_list[3]
        assert "linear diophantine equations" == term_list[4]
        assert "types systems" == term_list[5]
        assert "minimal supporting set" == term_list[6]
        assert "linear constraints" == term_list[7]
        assert "algorithms" == term_list[8]

        results, top_vertices = keywords_extraction(example_abstract, top_p = 0.3, weight_comb="log_norm_sum")
        print("extracted keywords with log_norm_sum weighting:"+ str(results))

        term_list = [term[0] for term in results]
        assert "minimal set" == term_list[0]
        assert "types systems" == term_list[1]
        assert "linear diophantine equations" == term_list[2]
        assert "linear constraints" == term_list[3]
        assert "minimal supporting set" == term_list[4]
        assert "strict inequations" == term_list[5]
        assert "corresponding algorithms" == term_list[6]
        assert "nonstrict inequations" == term_list[7]
        assert "systems" == term_list[8]

        results, top_vertices = keywords_extraction(example_abstract, top_p = 0.3, weight_comb="gaussian_norm_sum")
        print("extracted keywords with gaussian_norm_sum weighting:"+ str(results))

        term_list = [term[0] for term in results]
        assert "linear diophantine equations" == term_list[0]
        assert "minimal supporting set" == term_list[1]
        assert "minimal set" == term_list[2]
        assert "types systems" == term_list[3]
        assert "linear constraints" == term_list[4]
        assert "strict inequations" == term_list[5]
        assert "systems" == term_list[6]
        assert "corresponding algorithms" == term_list[7]
        assert "nonstrict inequations" == term_list[8]

        results, top_vertices = keywords_extraction(example_abstract, top_p = 0.3, weight_comb="max")
        print("extracted keywords with max weighting:"+ str(results))

        term_list = [term[0] for term in results]
        assert "linear constraints" == term_list[0] or "linear diophantine equations" == term_list[0]
        assert "linear diophantine equations" == term_list[1] or "linear constraints" == term_list[1]
        assert "systems" == term_list[2] or "types systems" == term_list[2]
        assert "systems" == term_list[3] or "types systems" == term_list[3]
        assert "set" == term_list[4] or "minimal set" == term_list[4] or "minimal supporting set" == term_list[4]
        assert "minimal set" == term_list[5] or "set" == term_list[5] or "minimal supporting set" == term_list[5]
        assert "minimal supporting set" == term_list[6] or "minimal set" == term_list[6] or "set" == term_list[6]
        assert "minimal" == term_list[7]
        assert "algorithms" == term_list[8] or "corresponding algorithms" == term_list[8]

        results, top_vertices = keywords_extraction(example_abstract, top_p = 0.3, weight_comb="norm_max")
        print("extracted keywords with norm_max weighting:"+ str(results))

        term_list = [term[0] for term in results]
        assert "systems" == term_list[0]
        assert "set" == term_list[1]
        assert "minimal" == term_list[2]
        assert "algorithms" == term_list[3]
        assert "solutions" == term_list[4]
        assert "linear constraints" == term_list[5]
        assert "types systems" == term_list[6]
        assert "minimal set" == term_list[7]
        assert "linear diophantine equations" == term_list[8]

        results, top_vertices = keywords_extraction(example_abstract, top_p = 0.3, weight_comb="log_norm_max")
        print("extracted keywords with log_norm_max weighting:"+ str(results))

        term_list = [term[0] for term in results]
        assert "linear constraints" == term_list[0]
        assert "types systems" == term_list[1]
        assert "minimal set" == term_list[2]
        assert "linear diophantine equations" == term_list[3]
        assert "corresponding algorithms" == term_list[4]
        assert "nonstrict inequations" == term_list[5] or "strict inequations" == term_list[5]
        assert "strict inequations" == term_list[6] or "nonstrict inequations" == term_list[6]
        assert "minimal supporting set" == term_list[7]
        assert "systems" == term_list[8]

        results, top_vertices = keywords_extraction(example_abstract, top_p = 0.3, weight_comb="gaussian_norm_max")
        print("extracted keywords with gaussian_norm_max weighting:"+ str(results))

        term_list = [term[0] for term in results]
        assert "linear constraints" == term_list[0]
        assert "linear diophantine equations" == term_list[1]
        assert "systems" == term_list[2] or "types systems" == term_list[2]
        assert "types systems" == term_list[3] or "systems" == term_list[3]
        assert "set" == term_list[4] or "minimal set" == term_list[4]
        assert "minimal set" == term_list[5] or "set" == term_list[5]
        assert "minimal supporting set" == term_list[6]
        assert "minimal" == term_list[7]
        assert "algorithms" == term_list[8] or "corresponding algorithms" == term_list[8]

        results, top_vertices = keywords_extraction(example_abstract, top_p = 0.3, weight_comb="len_log_norm_max")
        print("extracted keywords with len_log_norm_max weighting:"+ str(results))
        term_list = [term[0] for term in results]
        assert "linear diophantine equations" == term_list[0]
        assert "minimal supporting set" == term_list[1]
        assert "linear constraints" == term_list[2]
        assert "types systems" == term_list[3]

        results, top_vertices = keywords_extraction(example_abstract, top_p = 0.3, weight_comb="len_log_norm_avg")
        print("extracted keywords with len_log_norm_avg weighting:"+ str(results))
        term_list = [term[0] for term in results]
        assert "linear diophantine equations" == term_list[0]
        assert "minimal supporting set" == term_list[1]
        assert "minimal set" == term_list[2]
        assert "types systems" == term_list[3]

        results, top_vertices = keywords_extraction(example_abstract, top_p = 0.3, weight_comb="len_log_norm_sum")
        print("extracted keywords with len_log_norm_sum weighting:"+ str(results))
        term_list = [term[0] for term in results]
        assert "linear diophantine equations" == term_list[0]
        assert "minimal supporting set" == term_list[1]
        assert "minimal set" == term_list[2]
        assert "types systems" == term_list[3]
        assert "linear constraints" == term_list[4]

        with self.assertRaises(ValueError) as context:
            keywords_extraction(example_abstract, top_p = 0.3, weight_comb="my_norm")

            self.assertTrue("Unspported weight_comb 'my_norm'! "
                            "Options are 'avg', 'norm_avg', 'log_norm_avg', 'gaussian_norm_avg', 'sum', "
                            "'norm_sum', 'log_norm_sum', 'gaussian_norm_sum', 'max', 'norm_max',"
                            " 'log_norm_max', 'gaussian_norm_max', "
                            "'len_log_norm_max', 'len_log_norm_avg', 'len_log_norm_sum'. " in context.exception)

    def test_keywords_extraction_from_segmented_corpus(self):
        example_user_defined_context_corpus = [["Compatibility", "of", "systems", "of", "linear", "constraints",
                                             "over", "the", "set", "of", "natural", "numbers",".",
                                             "Criteria", "of", "compatibility", "of", "a", "system", "of",
                                             "linear", "Diophantine", "equations", ",", "strict", "inequations", ",",
                                             "and", "nonstrict", "inequations", "are", "considered", "."],
                                            ["Upper", "bounds", "for", "components", "of", "a", "minimal", "set",
                                             "of", "solutions", "and","algorithms","of", "construction", "of",
                                             "minimal", "generating", "sets", "of", "solutions", "for", "all",
                                             "types", "of", "systems", "are", "given", "."],
                                            ["These", "criteria", "and", "the", "corresponding", "algorithms",
                                             "for", "constructing", "a", "minimal", "supporting", "set", "of",
                                             "solutions", "can", "be", "used", "in", "solving", "all", "the",
                                             "considered", "types", "systems", "and", "systems", "of", "mixed",
                                             "types","."]]
        from jgtextrank.core import keywords_extraction_from_segmented_corpus
        results, top_vertices = keywords_extraction_from_segmented_corpus(example_user_defined_context_corpus, top_p=1, weight_comb="sum")
        print("extracted keywords with user defined corpus context:"+ str(results))
        print("top_vertices: ", top_vertices)
        assert 23 == len(results)
        term_list = [term[0] for term in results]

        assert "linear diophantine equations" in term_list
        assert "minimal supporting set" in term_list
        assert "minimal set" in term_list
        assert "types systems" in term_list
        assert "linear constraints" in term_list
        assert "strict inequations" in term_list
        assert "systems" in term_list
        assert "corresponding algorithms" in term_list
        assert "natural numbers" in term_list, "'natural numbers' is given more " \
                                               "weights than the weight with computed in default sentential context."
        assert "nonstrict inequations" in term_list
        assert "mixed types" in term_list
        assert "minimal" in term_list
        assert 'set' in term_list
        # [('linear diophantine equations', 0.17848), ('minimal supporting set', 0.16067),
        # ('minimal set', 0.12723), ('types systems', 0.1143), ('linear constraints', 0.10842),
        # ('strict inequations', 0.08805), ('systems', 0.07958), ('corresponding algorithms', 0.07575),
        # ('natural numbers', 0.07384), ('nonstrict inequations', 0.07262),
        # ('mixed types', 0.06943), ('minimal', 0.06362), ('set', 0.06361),
        # ('algorithms', 0.05406), ('solutions', 0.04964), ('criteria', 0.03779),
        # ('compatibility', 0.03606), ('construction', 0.0352), ('types', 0.03472),
        # ('sets', 0.03405), ('system', 0.02125), ('upper', 0.00644), ('components', 0.00644)]

    @ignore_warnings
    def test_keywords_extraction_from_tagged_corpus(self):
        from jgtextrank.core import keywords_extraction_from_tagged_corpus

        pos_tagged_corpus= [[('Compatibility', 'NN'), ('of', 'IN'), ('systems', 'NNS'), ('of', 'IN'),
                             ('linear', 'JJ'), ('constraints', 'NNS'), ('over', 'IN'), ('the', 'DT'),
                             ('set', 'NN'), ('of', 'IN'), ('natural', 'JJ'), ('numbers', 'NNS'), ('.', '.')],
                            [('Criteria', 'NNS'), ('of', 'IN'), ('compatibility', 'NN'), ('of', 'IN'),
                             ('a', 'DT'), ('system', 'NN'), ('of', 'IN'), ('linear', 'JJ'),
                             ('Diophantine', 'NNP'), ('equations', 'NNS'), (',', ','), ('strict', 'JJ'),
                             ('inequations', 'NNS'), (',', ','), ('and', 'CC'), ('nonstrict', 'JJ'),
                             ('inequations', 'NNS'), ('are', 'VBP'), ('considered', 'VBN'), ('.', '.')],
                            [('Upper', 'NNP'), ('bounds', 'VBZ'), ('for', 'IN'), ('components', 'NNS'),
                             ('of', 'IN'), ('a', 'DT'), ('minimal', 'JJ'), ('set', 'NN'), ('of', 'IN'),
                             ('solutions', 'NNS'), ('and', 'CC'), ('algorithms', 'NN'), ('of', 'IN'),
                             ('construction', 'NN'), ('of', 'IN'), ('minimal', 'JJ'), ('generating', 'VBG'),
                             ('sets', 'NNS'), ('of', 'IN'), ('solutions', 'NNS'), ('for', 'IN'), ('all', 'DT'),
                             ('types', 'NNS'), ('of', 'IN'), ('systems', 'NNS'), ('are', 'VBP'),
                             ('given', 'VBN'), ('.', '.')],
                            [('These', 'DT'), ('criteria', 'NNS'), ('and', 'CC'), ('the', 'DT'),
                             ('corresponding', 'JJ'), ('algorithms', 'NN'), ('for', 'IN'),
                             ('constructing', 'VBG'), ('a', 'DT'), ('minimal', 'JJ'), ('supporting', 'VBG'),
                             ('set', 'NN'), ('of', 'IN'), ('solutions', 'NNS'), ('can', 'MD'), ('be', 'VB'),
                             ('used', 'VBN'), ('in', 'IN'), ('solving', 'VBG'), ('all', 'PDT'), ('the', 'DT'),
                             ('considered', 'VBN'), ('types', 'NNS'), ('systems', 'NNS'), ('and', 'CC'),
                             ('systems', 'NNS'), ('of', 'IN'), ('mixed', 'JJ'), ('types', 'NNS'), ('.', '.')]]

        results, top_vertices = keywords_extraction_from_tagged_corpus(pos_tagged_corpus, top_p = 0.3, weight_comb="sum")
        print("extracted keywords from pre-tagged content:"+ str(results))
        print("top_vertices: ", top_vertices)

        print("len(results): ", len(results))
        assert 10 == len(results), "check possible changes/errors in solver and hyperparameter, e.g., num_iter, tol"

        term_list = [term[0] for term in results]
        assert "linear diophantine equations" in term_list
        assert "types systems" in term_list
        assert "linear constraints" in term_list
        assert "minimal set" in term_list
        assert "systems" in term_list
        assert "corresponding algorithms" in term_list
        assert "algorithms" in term_list
        assert "set" in term_list
        assert "solutions" in term_list
        assert "minimal" in term_list

        # after lemmatisation
        results, top_vertices = keywords_extraction_from_tagged_corpus(pos_tagged_corpus, top_p = 0.3, lemma=True)
        print("extracted keywords from pre-tagged content after lemmatisation: ", results)
        print("top_vertices after lemmatisation: ", top_vertices)

        assert len(results) == 11
        term_list = [term[0] for term in results]
        assert "linear diophantine equation" in term_list
        assert "type system" in term_list
        assert "minimal set" in term_list
        assert "linear constraint" in term_list
        assert "strict inequations" in term_list
        assert "corresponding algorithm" in term_list
        assert "system" in term_list
        assert "nonstrict inequations" in term_list
        assert "natural number" in term_list
        assert "algorithm" in term_list
        assert "set" in term_list

    def test_kea_with_text_formulate(self):
        """
        This is to test the content with formulate
            where simply splits the term units with space may have the conflicts with the original tokeniser
        :return:
        """
        from jgtextrank.core import _keywords_extraction_from_preprocessed_context

        S0021999113005652_textsnippet = [(['it', 'be', 'interesting', 'to', 'quantify', 'the', 'effect', 'of',
                                           'the', 'schmidt', 'number', 'and', 'the', 'chemical', 'reaction',
                                           'rate', 'on', 'the', 'bulk', '-', 'mean', 'concentration', 'of', 'b','in', 'water', '.'],
                                          [('interesting', 'JJ'), ('effect', 'NNS'), ('schmidt', 'NNP'), ('number', 'NN'),
                                           ('chemical', 'JJ'), ('reaction', 'NN'), ('rate', 'NN'), ('bulk', 'JJ'),
                                           ('concentration', 'NN'), ('water', 'NN')]),
                                         (['the', 'datum', 'could', 'present', 'important', 'information', 'on',
                                           'evaluate', 'the', 'environmental', 'impact', 'of', 'the', 'degradation',
                                           'product', 'of', 'b', ',', 'as', 'well', 'as', 'acidification', 'of',
                                           'water', 'by', 'the', 'chemical', 'reaction', '.'],
                                          [('datum', 'NNS'), ('important', 'JJ'), ('information', 'NN'),
                                           ('environmental', 'JJ'), ('impact', 'NNS'), ('degradation', 'NN'), ('product', 'NN'),
                                           ('acidification', 'NN'), ('water', 'NN'), ('chemical', 'JJ'), ('reaction', 'NN')]),
                                         (['here', ',', 'the', 'bulk', '-', 'mean', 'concentration', 'of', 'b',
                                           'be', 'define', 'by(24)cb⁎ =∫01〈cb⁎〉(z⁎)dz⁎', 'fig', '.'],
                                          [('bulk', 'JJ'), ('concentration', 'NN'), ('by(24)cb⁎ =∫01〈cb⁎〉(z⁎)dz⁎', 'NNP'), ('fig', 'NNP')]),
                                         (['15', 'depict', 'the', 'effect', 'of', 'the', 'schmidt', 'and', 'the',
                                           'chemical', 'reaction', 'rate', 'on', 'the', 'bulk', '-', 'mean', 'concentration', 'cb⁎ .'],
                                          [('depict', 'NNS'), ('effect', 'NN'), ('schmidt', 'NNP'), ('chemical', 'JJ'),
                                           ('reaction', 'NN'), ('rate', 'NN'), ('bulk', 'JJ'), ('concentration', 'NN')]),
                                         (['it', 'be', 'worth', 'to', 'mention', 'here', 'that', 'the', 'bulk', '-', 'mean',
                                           'concentration', 'of', 'b', 'reach', 'approximately', '0.6', 'as', 'the', 'chemical',
                                           'reaction', 'rate', 'and', 'the', 'schmidt', 'number', 'increase', 'to',
                                           'infinite', ',', 'and', 'the', 'concentration', 'be', 'small', 'than', 'the',
                                           'equilibrium', 'concentration', 'of', 'a', 'at', 'the', 'interface', '.'],
                                          [('worth', 'JJ'), ('bulk', 'JJ'), ('concentration', 'NN'), ('chemical', 'JJ'),
                                           ('reaction', 'NN'), ('rate', 'NN'), ('schmidt', 'NNP'), ('number', 'NN'),
                                           ('increase', 'NN'), ('concentration', 'NN'), ('equilibrium', 'NN'), ('concentration', 'NN'), ('interface', 'NN')]),
                                         (['this', 'figure', 'indicate', 'that', 'progress', 'of', 'the',
                                           'chemical', 'reaction', 'be', 'somewhat', 'interfere', 'by', 'turbulent',
                                           'mix', 'in', 'water', ',', 'and', 'the', 'efficiency', 'of', 'the',
                                           'chemical', 'reaction', 'be', 'up', 'to', 'approximately', '60', '%', '.'],
                                          [('figure', 'NN'), ('progress', 'NN'), ('chemical', 'JJ'), ('reaction', 'NN'),
                                           ('turbulent', 'JJ'), ('water', 'NN'), ('efficiency', 'NN'), ('chemical', 'JJ'), ('reaction', 'NN')]),
                                         (['the', 'efficiency', 'of', 'the', 'chemical', 'reaction', 'in', 'water',
                                           'will', 'be', 'a', 'function', 'of', 'the', 'reynolds', 'number', 'of',
                                           'the', 'water', 'flow', ',', 'and', 'the', 'efficiency', 'could', 'increase',
                                           'as', 'the', 'reynolds', 'number', 'increase', '.'],
                                          [('efficiency', 'NN'), ('chemical', 'JJ'), ('reaction', 'NN'), ('water', 'NN'),
                                           ('function', 'NN'), ('reynolds', 'NNP'), ('number', 'NN'), ('water', 'NN'),
                                           ('flow', 'NN'), ('efficiency', 'NN'), ('reynolds', 'NNP'), ('number', 'NN'), ('increase', 'NNS')]),
                                         (['we', 'need', 'an', 'extensive', 'investigation', 'on', 'the', 'efficiency',
                                           'of', 'the', 'aquarium', 'chemical', 'reaction', 'in', 'the', 'near',
                                           'future', 'to', 'extend', 'the', 'result', 'of', 'this', 'study',
                                           'further', 'to', 'establish', 'practical', 'modelling', 'for', 'the',
                                           'gas', 'exchange', 'between', 'air', 'and', 'water', '.'],
                                          [('extensive', 'JJ'), ('investigation', 'NN'), ('efficiency', 'NN'),
                                           ('aquarium', 'JJ'), ('chemical', 'NN'), ('reaction', 'NN'),
                                           ('future', 'NN'), ('result', 'NNS'), ('study', 'NN'), ('practical', 'JJ'),
                                           ('modelling', 'NN'), ('gas', 'NN'), ('exchange', 'NN'), ('air', 'NN'), ('water', 'NN')])]

        results, top_vertices = _keywords_extraction_from_preprocessed_context(S0021999113005652_textsnippet, top_p = 1, weight_comb="sum")

        print("extracted keywords from pre-tagged S0021999113005652 text snippet:"+ str(results))
        print("top_vertices: ", top_vertices)
        print("total key terms", len(results))
        assert len(results) == 37
        assert results["schmidt number"] == 0.06231
        assert results["chemical reaction rate"] == 0.10836
        assert results["water"] == 0.05561
        assert results["by(24)cb⁎ =∫01〈cb⁎〉(z⁎)dz⁎ fig"] == 0.06098
        assert results["water flow"] == 0.07201
        assert results["aquarium chemical reaction"] == 0.10836

    def test_visualise_cooccurrence_graph(self):
        """
        produce the co-occurrence graph close to the example picture in original paper

        :return: None
        """
        example_tokenised_corpus_context = [["Compatibility", "of", "systems", "of", "linear", "constraints",
                                             "over", "the", "set", "of", "natural", "numbers", "." ,
                                             "Criteria", "of", "compatibility", "of", "a", "system", "of",
                                             "linear", "Diophantine", "equations", "strict", "inequations", ",",
                                             "and", "nonstrict", "inequations", "are", "considered",".", "Upper",
                                             "bounds", "for", "components","of", "a", "minimal", "set", "of",
                                             "solutions", "and", "algorithms", "of", "construction", "of",
                                             "minimal", "generating", "sets", "of", "solutions", "for", "all",
                                             "types", "of", "systems", "are", "given", ".", "These", "criteria",
                                             "and", "the", "corresponding", "algorithms", "for",
                                             "constructing", "a", "minimal", "supporting", "set", "of",
                                             "solutions", "can", "be", "used", "in", "solving", "all", "the",
                                             "considered", "types", "systems", "and", "systems", "of", "mixed",
                                             "types", "."]]
        # try to include verbs into the graph
        custom_categories = {'NNS', 'NNP', 'NN', 'JJ', 'VBZ'}
        # manually filter few nodes not appearing in the given example of original paper
        stop_words={'set', 'mixed', 'corresponding', 'supporting'}

        preprocessed_context = preprocessing_tokenised_context(example_tokenised_corpus_context,
                                                               syntactic_categories=custom_categories,
                                                               stop_words=stop_words)
        cooccurrence_graph, original_tokenised_context = build_cooccurrence_graph(preprocessed_context)

        connected_components = list(nx.connected_components(cooccurrence_graph))
        print("visualising connected components:", connected_components)
        assert len(connected_components) == 3

        pos = nx.spring_layout(cooccurrence_graph,k=0.20,iterations=20)
        nx.draw_networkx(cooccurrence_graph, pos=pos, arrows=True, with_labels=True)
        plt.show()
        plt.savefig("test_sample_cooccurrence_graph.png") # save as png