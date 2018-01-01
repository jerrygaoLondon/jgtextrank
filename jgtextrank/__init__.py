"""
jgtextrank: Yet another Python implementation of TextRank
==================================

jgtextrank is a Python package for the creation, manipulation, and study of TextRank algorithm, a graph based keywords extraction and summarization approach


Website (including documentation)::

    https://github.com/jerrygaoLondon/jgtextrank

Source::

    https://github.com/jerrygaoLondon/jgtextrank

Bug reports::

    https://github.com/jerrygaoLondon/jgtextrank/issues

Simple example
--------------
Extract weighted keywords with an undirected graph::

    >>> from jgtextrank import keywords_extraction
    >>> example_abstract = "Compatibility of systems of linear constraints over the set of natural numbers. " \
                           "Criteria of compatibility of a system of linear Diophantine equations, strict inequations, " \
                           "and nonstrict inequations are considered. Upper bounds for components of a minimal set of " \
                           "solutions and algorithms of construction of minimal generating sets of solutions for all " \
                           "types of systems are given. These criteria and the corresponding algorithms for " \
                           "constructing a minimal supporting set of solutions can be used in solving all the " \
                           "considered types systems and systems of mixed types."
    >>> keywords_extraction(example_abstract, top_p = 1, directed=False, weight_comb="sum")[0][:15]
    [('linear diophantine equations', 0.18059), ('minimal supporting set', 0.16649), ('minimal set', 0.13201), ('types systems', 0.1194), ('linear constraints', 0.10997), ('strict inequations', 0.08832), ('systems', 0.08351), ('corresponding algorithms', 0.0767), ('nonstrict inequations', 0.07276), ('mixed types', 0.07178), ('set', 0.06674), ('minimal', 0.06527), ('natural numbers', 0.06466), ('algorithms', 0.05479), ('solutions', 0.05085)]


License
-------

Released under the MIT License::

Copyright (C) 2017, JIE GAO <j.gao@sheffield.ac.uk>

"""

from jgtextrank.core import preprocessing, preprocessing_tokenised_context, build_cooccurrence_graph, \
    keywords_extraction, keywords_extraction_from_segmented_corpus, \
    keywords_extraction_from_tagged_corpus, keywords_extraction_from_corpus_directory
__all__=["preprocessing", "preprocessing_tokenised_context", "build_cooccurrence_graph"
         "keywords_extraction", "keywords_extraction_from_segmented_corpus",
         "keywords_extraction_from_tagged_corpus",
         "keywords_extraction_from_corpus_directory"]

__version__ = '0.1.1'