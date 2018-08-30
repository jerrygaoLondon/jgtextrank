jgTextRank : Yet another Python implementation of TextRank
==========================================================
This is a parallelisable and highly customisable implementation of the TextRank algorithm [Mihalcea et al., 2004].
You can define your own co-occurrence context, syntactic categories(choose either "closed" filters or "open" filters),
stop words, feed your own pre-segmented/pre-tagged data, and many more. You can also
load co-occurrence graph directly from your text for visual analytics, debug and fine-tuning your custom settings.
This implementation can also be applied to large corpus for terminology extraction.
It can be applied to short text for supervised learning in order to provide more interesting features than conventional TF-IDF Vectorizer.

TextRank algorithm look into the structure of word co-occurrence networks,
where nodes are word types and edges are word cooccurrence.

Important words can be thought of as being endorsed by other words,
and this leads to an interesting phenomenon. Words that are most
important, viz. keywords, emerge as the most central words in the
resulting network, with high degree and PageRank. The final important
step is post-filtering. Extracted phrases are disambiguated and
normalized for morpho-syntactic variations and lexical synonymy
(Csomai and Mihalcea 2007). Adjacent words are also sometimes
collapsed into phrases, for a more readable output.

Mihalcea, R., & Tarau, P. (2004, July). TextRank: Bringing order into texts. Association for Computational Linguistics.

### Usage ###

#### Simple examples

Extract weighted keywords with an undirected graph:

    >>> from jgtextrank import keywords_extraction
    >>> example_abstract = "Compatibility of systems of linear constraints over the set of natural numbers. " \
                           "Criteria of compatibility of a system of linear Diophantine equations, strict inequations, " \
                           "and nonstrict inequations are considered. Upper bounds for components of a minimal set of " \
                           "solutions and algorithms of construction of minimal generating sets of solutions for all " \
                           "types of systems are given. These criteria and the corresponding algorithms for " \
                           "constructing a minimal supporting set of solutions can be used in solving all the " \
                           "considered types systems and systems of mixed types."
    >>> keywords_extraction(example_abstract, top_p = 1, directed=False)[0][:15]
    [('linear diophantine equations', 0.18059), ('minimal supporting set', 0.16649), ('minimal set', 0.13201), ('types systems', 0.1194), ('linear constraints', 0.10997), ('strict inequations', 0.08832), ('systems', 0.08351), ('corresponding algorithms', 0.0767), ('nonstrict inequations', 0.07276), ('mixed types', 0.07178), ('set', 0.06674), ('minimal', 0.06527), ('natural numbers', 0.06466), ('algorithms', 0.05479), ('solutions', 0.05085)]

Change syntactic filters to restrict vertices to only noun phrases for addition to the graph:

    >>> custom_categories = {'NNS', 'NNP', 'NN'}
    >>> keywords_extraction(example_abstract, top_p = 1, top_t=None,
                            directed=False, syntactic_categories=custom_categories)[0][:15]
    [('types systems', 0.17147), ('diophantine equations', 0.15503), ('supporting set', 0.14256), ('solutions', 0.13119), ('systems', 0.12452), ('algorithms', 0.09188), ('set', 0.09188), ('compatibility', 0.0892), ('construction', 0.05068), ('criteria', 0.04939), ('sets', 0.04878), ('types', 0.04696), ('system', 0.01163), ('constraints', 0.01163), ('components', 0.01163)]

You can provide an additional stop word list to filter unwanted candidate terms:

    >>> stop_list={'set', 'mixed', 'corresponding', 'supporting'}
    >>> keywords_extraction(example_abstract, top_p = 1, top_t=None,
                            directed=False,
                            syntactic_categories=custom_categories, stop_words=stop_list)[0][:15]
    [('types systems', 0.20312), ('diophantine equations', 0.18348), ('systems', 0.1476), ('algorithms', 0.11909), ('solutions', 0.11909), ('compatibility', 0.10522), ('sets', 0.06439), ('construction', 0.06439), ('criteria', 0.05863), ('types', 0.05552), ('system', 0.01377), ('constraints', 0.01377), ('components', 0.01377), ('numbers', 0.01377), ('upper', 0.01377)]

You can also use lemmatization (disabled by default) to increase the weight for terms appearing with various inflectional variations:

    >>> keywords_extraction(example_abstract, top_p = 1, top_t=None,
                            directed=False,
                            syntactic_categories=custom_categories,
                            stop_words=stop_list, lemma=True)[0][:15]
    [('type system', 0.2271), ('diophantine equation', 0.20513), ('system', 0.16497), ('algorithm', 0.14999), ('compatibility', 0.11774), ('construction', 0.07885), ('solution', 0.07885), ('criterion', 0.06542),('type', 0.06213), ('component', 0.01538), ('constraint', 0.01538), ('upper', 0.01538), ('inequations', 0.01538), ('number', 0.01538)]

The co-occurrence window size is 2 by default. You can try with a different number for your data:

    >>> keywords_extraction(example_abstract,  window=5,
                            top_p = 1, top_t=None, directed=False,
                            stop_words=stop_list, lemma=True)[0][:15]
    [('linear diophantine equation', 0.19172), ('linear constraint', 0.13484), ('type system', 0.1347), ('strict inequations', 0.12532), ('system', 0.10514), ('nonstrict inequations', 0.09483), ('solution', 0.06903), ('natural number', 0.06711), ('minimal', 0.06346), ('algorithm', 0.05762), ('compatibility', 0.05089), ('construction', 0.04541), ('component', 0.04418), ('criterion', 0.04086), ('type', 0.02956)]

Try with a centrality measures:

    >>> keywords_extraction(example_abstract, solver="current_flow_betweenness",
                            window=5, top_p = 1, top_t=None,
                            directed=False, stop_words=stop_list,
                            lemma=True)[0][:15]
    [('type system', 0.77869), ('system', 0.77869), ('solution', 0.32797), ('linear diophantine equation', 0.30657), ('linear constraint', 0.30657), ('minimal', 0.26052), ('algorithm', 0.21463), ('criterion', 0.19821), ('strict inequations', 0.19651), ('nonstrict inequations', 0.19651), ('compatibility', 0.1927), ('natural number', 0.11111), ('component', 0.11111), ('type', 0.10718), ('construction', 0.10039)]

Tuning your graph model as a black box can be problematic.
You can try to visualize your co-occurrence network with your sample dataset in order to manually validate your custom parameters:

    >>> from jgtextrank import preprocessing, build_cooccurrence_graph
    >>> import networkx as nx
    >>> import matplotlib.pyplot as plt
    >>> preprocessed_context = preprocessing(example_abstract, stop_words=stop_list, lemma=True)
    >>> cooccurrence_graph, context_tokens = build_cooccurrence_graph(preprocessed_context, window=2)
    >>> pos = nx.spring_layout(cooccurrence_graph,k=0.20,iterations=20)
    >>> nx.draw_networkx(cooccurrence_graph, pos=pos, arrows=True, with_requets labels=True)
    >>> plt.savefig("my_sample_cooccurrence_graph.png")
    >>> plt.show()


More examples (e.g., with custom co-occurrence context, how to extract from a corpus of text files,
feed your own pre-segmented/pre-tagged data), please see [jgTextRank wiki](https://github.com/jerrygaoLondon/jgtextrank/wiki)

### Documentation

For `jgtextrank` documentation, see:

* [textrank](http://htmlpreview.github.io/?https://github.com/jerrygaoLondon/jgtextrank/blob/master/docs/jgtextrank.html)

### Installation ###

To install from [PyPi](https://pypi.python.org/pypi/jgtextrank):

    pip install jgtextrank

To install from github

    pip install git+git://github.com/jerrygaoLondon/jgtextrank.git

or

    pip install git+https://github.com/jerrygaoLondon/jgtextrank.git

To install from source

    python setup.py install

### Dependencies

* [nltk](http://www.nltk.org/)

* [networkx](https://networkx.github.io/)

### Status

* Beta release (update)

    * Python implementation of TextRank algorithm for keywords extraction

    * Support directed/undirected and unweighted graph

    * >12 MWTs weighting methods

    * 3 pagerank implementations and >15 additional graph ranking algorithms

    * Parallelisation  of vertices co-occurrence computation (allow to set number of available worker instances)

    * Support various custom settings and parameters (e.g., use of lemmatization,
       co-occurrence window size, options for two co-occurrence context strategies,
       use of custom syntactic filters, use of custom stop words)

    * Keywords extraction from pre-processed (pre-segmented or pre-tagged) corpus/context

    * Keywords extraction from a given corpus directory of raw text files

    * Export ranked result into 'csv' or 'json' file

    * Support visual analytics of vertices network

### Contributions ###

This project welcomes contributions, feature requests and suggestions.
Please feel free to create issues or send me your
[pull requests](https://help.github.com/articles/creating-a-pull-request/).

**Important**: By submitting a patch, you agree to allow the project owners
to license your work under the MIT license.

### To Cite ###

Here's a Bibtex entry if you need to cite `jgTextRank` in your research paper:

    @Misc{jgTextRank,
    author =   {Gao, Jie},
    title =    {jgTextRank: Yet another Python implementation of TextRank},
    howpublished = {\url{https://github.com/jerrygaoLondon/jgtextrank/}},
    year = {2017}
    }

### Who do I talk to? ###

* Jie Gao <j.gao@sheffield.ac.uk>

### history ###
* 0.1.2 Beta version - Aug 2018
    * bug fixes
    * 15 additional graph ranking algorithms
* 0.1.1 Alpha version - 1st Jan 2018
