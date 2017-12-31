import os

from nltk.corpus.reader.wordlist import WordListCorpusReader
import string
lowercase_list = lambda t: [t_i.lower() for t_i in t]
punctuation_filter = lambda t : filter(lambda a: a not in string.punctuation, t)
flatten_list = lambda l: [item for sublist in l for item in sublist]

import spacy
spacy_nlp = spacy.load('en_core_web_sm')

from jgtextrank.preprocessing import segmentation


def pre_processing_corpus_with_spacy(corpus_dir, encoding="utf-8", lemma=True):
    for fname in os.listdir(corpus_dir):
        doc_content = ""
        for line in open(os.path.join(corpus_dir, fname), encoding=encoding):
            doc_content += line

        for sentence in segmentation.sent_tokenize(doc_content):
            yield pre_processing_text_with_spacy(sentence, lemma=lemma)


def pre_processing_text_with_spacy(text, lemma=True):
    """
    To install:

    1) pip install spacy
    2) python -m spacy download en

    see also https://spacy.io/usage/spacy-101#annotations-pos-deps
    see also https://spacy.io/usage/linguistic-features

    :param text:
    :return: tagged text
    """

    tagged_text = spacy_nlp(text)
    tagged_tokens = []
    for token in tagged_text:
        # print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_,token.shape_, token.is_alpha, token.is_stop)
        if lemma:
            tagged_tokens.append((token.text if token.pos_ == "PRON" else token.lemma_, token.tag_))
        else:
            tagged_tokens.append((token.text, token.tag_))
    return tagged_tokens


def load_genia_gs_terms(fname, is_norm=True):
    """
    load Gold standard terms from GENIA concept.txt

    1. remove concept predicates
    2. remove incorrect annotations
    3. synonym replacement
    :return: set
    """
    # remove concept predicates in genia 'concept' list
    # "Blood cell receptor" is not included in the content we use which is actually a "second" title of article "MEDLINE:94077003".
    print("load and normalise GENIA gs term list from [%s]" % fname)
    genia_gs_stop_words = ["*", "(OR", "(NOT", "(TO", "(THAN", "(VERSUS", "(AND", "(BUT", "(AS", "(AND/OR", "Blood cell receptor"]
    # replace GENIA GS concepts with surface forms in the raw corpus
    genia_annotation_mapping = {'responsivenessp': 'responsiveness',
                                'PU.1- /- mouse': 'PU.1-/- mice',
                                'activationp': 'activation',
                                'mammalian oncogenic virus':'mammalian oncogenic viruses',
                                'endrometrium':'endometrium',
                                'bonep': 'bone',
                                'yhiol':'Thiol',
                                'IFN-gamma leve':'IFN-gamma level',
                                'familiy':'family',
                                'frequencie':'frequencies'}

    gs_term_set = load_gs_terms(fname, stopwords=genia_gs_stop_words, normalised_gs_terms = genia_annotation_mapping, is_norm=is_norm)
    normalised_gs_term_list = [synonym_normalisation_4_genia(gs_term) for gs_term in gs_term_set]

    normalised_gs_term_list = list(filter(None,normalised_gs_term_list))

    return set(normalised_gs_term_list)


def load_gs_terms(fname, stopwords = [], normalised_gs_terms= dict(), is_norm=True):
    """
    load gs term set (with minimum pre-processing) from a "list" file

    :param fname: raw gs file
    :param stopwords: any gs term containing the stopwords will be filtered
    :param normalised_gs_terms: pre-configured normalisation dictionary; replace current form (key) with the different form (value)
    :return: set(), gs term set
    """
    reader = WordListCorpusReader('',fname)
    gs_term_list = reader.words()

    print("initially loaded gs terms size: ", len(gs_term_list))

    final_gs_terms = set()
    for gs_term in gs_term_list:
        # remove terms with annotated predicates
        # annotations= ["*", "(OR", "(NOT", "(TO", "(THAN", "(VERSUS", "(AND", "(BUT", "(AS", "(AND/OR"]
        if any(x in gs_term for x in stopwords):
            continue

        term = gs_term
        for key, value in normalised_gs_terms.items():
            term = term.replace(key, value)

        final_gs_terms.add(term)

    normalised_gs_terms_list = final_gs_terms
    #print("final_gs_terms: ", final_gs_terms)
    if is_norm:
        normalised_gs_terms_list = [normalise_term(gs_term) for gs_term in normalised_gs_terms_list]
        # filter duplicated normalised terms before the evaluation
        normalised_gs_terms_list = list(filter(None,normalised_gs_terms_list))
        print("normalised gs terms loaded.")

    #print("final loaded valid gs terms size: %s" % len(normalised_gs_terms_list))
    return set(normalised_gs_terms_list)


def normalise_term(term_str):
    return remove_punctuations(term_str).lower().strip()


def punctuation_filter_for_word_level(tokens=list()):
    filtered_tokens = punctuation_filter(tokens)

    punc_filtered_tokens = [list(remove_punctuations(current_token)) for current_token in filtered_tokens]

    return flatten_list(punc_filtered_tokens)


def remove_punctuations(raw_term):
    #print("remove_punctuations -> type(raw_term): ", type(raw_term))
    #print("remove_punctuations -> raw_term: ", raw_term)
    replace_all_punc_term_trans = raw_term.maketrans(string.punctuation, ' '*len(string.punctuation))
    space_replace_punc_all_term = raw_term.translate(replace_all_punc_term_trans)
    #remove multiple spaces and trailing spaces
    space_replace_punc_all_term = ' '.join(space_replace_punc_all_term.split())
    return space_replace_punc_all_term


def synonym_normalisation_4_genia(term_str):
    """
    synonym + inflectional(simply lemmatization replacement) normalisation for GENIA dataset evaluation

    this method applies to the surface form of both gs term ('concept.txt') and term candidate before further normalisation

    :param term_str:
    :return:
    """
    normed_term_str = term_str.replace('mouse', 'mice')
    normed_term_str = normed_term_str.replace('Mouse', 'Mice')
    normed_term_str = normed_term_str.replace('analyses', 'analysis')
    normed_term_str = normed_term_str.replace('Analyses', 'Analysis')
    normed_term_str = normed_term_str.replace('women', 'woman')
    normed_term_str = normed_term_str.replace('l cell resistance', 'lymphoid cell resistance')
    normed_term_str = normed_term_str.replace('DS lymphocyte', 'DS ones')
    normed_term_str = normed_term_str.replace('ds lymphocyte', 'ds ones')
    # to normalise gs terms to make it consistent with the pre-processing results from raw corpus, e.g., "Ph'", "5'-TGCGTCA-3'"
    normed_term_str = normed_term_str.rstrip("'")

    return normed_term_str


def term_precision(reference, test):
    """
    Given a set of reference values and a set of test values, return
    the fraction of test values that appear in the reference set.
    In particular, return card(``reference`` intersection ``test``)/card(``test``).
    If ``test`` is empty, then return None.

    :type reference: set
    :param reference: A set of reference values.
    :type test: set
    :param test: A set of values to compare against the reference set.
    :rtype: float or None
    """
    if (not hasattr(reference, 'intersection') or
            not hasattr(test, 'intersection')):
        raise TypeError('reference and test should be sets')

    if len(test) == 0:
        return None
    else:
        return len(reference.intersection(test)) / len(test)


def term_recall(reference, test):
    """
    Given a set of reference values and a set of test values, return
    the fraction of reference values that appear in the test set.
    In particular, return card(``reference`` intersection ``test``)/card(``reference``).
    If ``reference`` is empty, then return None.

    :type reference: set
    :param reference: A set of reference values.
    :type test: set
    :param test: A set of values to compare against the reference set.
    :rtype: float or None
    :param bool: bool value indicating whether to evaluation inclusivity
    :param inclusive
    """
    if (not hasattr(reference, 'intersection') or
            not hasattr(test, 'intersection')):
        raise TypeError('reference and test should be sets')

    if len(reference) == 0:
        return None
    else:
        return round(len(reference.intersection(test)) / len(reference), 3)

if __name__ == '__main__':
    #gs_terms = load_genia_gs_terms('genia_gs_terms.txt')
    #test_set =  ['natural killer cells activates endothelial cells', 'il 6 induced cells resemble plasma cells', 'human myeloid cell nuclear differentiation antigen gene promoter', 'lipopolysaccharide induced transcription factor regulating tumor necrosis factor alpha gene expression', 'responsive cells blocked il 1 induced gene transcription', 'cell receptor positive cells', 'multipotent eml cells harbor substantial nuclear hormone receptor coactivator activity', 'human gm csf receptor alpha promoter directs reporter gene activity', 'cells contained nuclear stat protein', 'human t cell leukemia line jurkat cells', 'peroxisome proliferator activated receptor activators target human endothelial cells', 'thp 1 cells c jun mrna expression increased', 'protein kinase c depleted endothelial cells', 'human t cell leukemia virus type 1 transformed mt 2 cells', 'cells protein kinase', 'human t cell leukemia virus type i infected cells', 'class ii transactivator independent endothelial cell mhc class ii gene activation induced', 'jurkat t cells induced dramatic cell aggregation', 'cells induces nuclear expression', 'human monocytic m csf receptor promoter directs reporter gene activity', 'human myeloid leukemia cells induced', 'pu 1 promoter directs cell type specific reporter gene expression', 'cells inhibits human immunodeficiency virus replication', 'human myeloid selective ccaat enhancer binding protein gene', 'human il 2 activated nk cells', 'cells expressing macrophage cell surface ags', 'normal human peripheral blood mononuclear cells', 'cell precursor acute lymphoblastic leukemia cells', 'endothelial cells increased expression', 'cell leukemia jurkat cells', 'cells expressing cell surface glycophorin', 'human immunodeficiency virus type 1 infected cells', 'cell prolymphocytic leukemia cells mediated', 'cells demonstrate accelerated cell cycle progression', 'il 1 activated human umbilical vein endothelial cells', 'normal bone marrow cells documents expression', 'cells enhances transcription factor', 'mature cells underwent premature cell death', 'primary human peripheral blood mononuclear cells', 'cells transcription factor', 'mature normal human myeloid cells', 'activated human nk cells', 'human histiocytic u937 cells mrna', 'normal immature human myeloid cells', 'cell transcription factor gata 3 stimulates hiv 1 expression', 'cell lineage cells arrested', 'cells expressing high v abl kinase activity', 'cells selectively enhances il 4 expression relative', 'normal human hematopoietic cells', 'transcription factor nf kappa b endothelial cell activation', 'duffy gene promoter abolishes erythroid gene expression', 'cell hybridoma hs 72 cells', 'transcription factor ccaat enhancer binding protein alpha', 'human peripheral blood nk cells', 'primary human blood mononuclear cells', 'cell surface protein expression', 'human peripheral blood mononuclear cells', 'human hl 60 myeloid leukemia cells differentiate', 'normal human cells', 'human endothelial cells demonstrated', 'stimulated human endothelial cells', 'human thp 1 monocytic leukemia cells cultured', 'transcription factor nf kappab regulates inducible oct 2 gene expression', 'human peripheral blood cells', 'human nk cells activate porcine ec', 'primary human erythroid cells', 'chinese hamster ovary cells expressing human recombinant alphaiibbeta3', 'human blood mononuclear cells cultured', 'cultured human blood mononuclear cells', 'activate human umbilical vein endothelial cells', 'cells decreased pkr expression', 'human cd34 hematopoietic progenitor cells isolated', 'human kg 1 myeloid leukemia cells', 'human monocytic cells results', 'human u 937 leukemia cells differentiate', 'human myeloid leukemia cells', 'treated cytokine stimulated human saphenous vein endothelial cells', 'irf family transcription factor gene expression', 'human cd3 cd16 natural killer cells express', 'human promyelocytic leukemia hl 60 cells', 'human monoblastic leukemia u937 cells', 'cells activates expression', 'human blood mononuclear cells', 'human leukemia u937 cells', 'activate human natural killer cells', 'purified human hematopoietic cells', 'human myeloblastic leukemia hl 60 cells', 'human leukemia hl60 cells respond', 'u 937 human promonocytic leukemia cells', 'human monocytic cells express interleukin 1beta', 'human nk cells provide', 'human u 937 leukemia cells', 'primary human cd34 hemopoietic progenitor cells', 'transfected human monocytic thp 1 cells', 'human red blood cells', 'human hl60 leukemia cells', 'transfected human cells suggests', 'hl60 human leukemia cells', 'human jurkat lymphoblastoid cells', 'cultured human umbilical vein endothelial cells', 'human mononuclear cells isolated', 'blast cells express lineage specific transcription factors', 'human promyelocytic leukemia cells', 'human breast cancer mcf 7 cells', 'cultured human dermal endothelial cells', 'human peripheral mononuclear cells', 'human cd34 erythroid progenitor cells', 'peripheral human mononuclear cells', 'lipopolysaccharide stimulated human monocytic cells treated', 'human leukemic cells studied', 'human myeloblastic leukemia ml 1 cells', 'ml 1 human myeloblastic leukemia cells', 'cultured human endothelial cells', 'resting human umbilical vein endothelial cells', 'human promyeloid leukemia cells', 'human myeloid leukemic cells', 'human leukemia cells', 'human primary haemopoietic cells', 'e1a expression marks cells', 'human purified cd34 cells', '1a9 m cells expressing human bcl2', 'human plasma cells', 'cytokine stimulated human umbilical vein endothelial cells', 'human umbilical arterial endothelial cells', 'tnf treated human umbilical vein endothelial cells', 'long term human lymphoid cells', 'human umbilical vein endothelial cells', 'lncap human prostate cancer cells', 'human myeloid u 937 cells', 'human dermal microvessel endothelial cells', 'human leukemic k562 cells', 'human prostatic cancer lncap cells', 'human cd34 hematopoietic progenitor cells', 'human cd34 hematopoietic stem cells', 'human aortic endothelial cells', 'bipotent human myeloid progenitor cells', 'mature human monocytic cells', 'human endothelial cells', 'human hematopoietic progenitor cells', 'human cancer lncap cells', 'transfected human erythroleukemia cells', 'human monocytic thp 1 cells', 'human thp 1 monocytic cells', 'cells infiltrating human genital herpes lesions', 'hematopoietic human erythroleukemia cells', 'human lymphoid cells', 'human mammary epithelial cells', 'human myeloid cell nuclear differentiation antigen promoter', 'human thp 1 macrophage cells', 'undifferentiated human monocytic cells', 'human hematopoietic cells', 'human natural killer cells', 'human myeloid cells', 'ifn gamma treated human cells infected', 'human intestinal epithelial cells', 'human lymphoma cells', 'human promonocytic u937 cells', 'human u937 promonocytic cells', 'human nk cells', 'human bronchial epithelial cells', 'u937 human monoblastic cells', 'porphyromonas gingivalis lipopolysaccharide stimulated human monocytic cells', 'human leukaemia cells carrying', 'human leukemic cells', 'lipopolysaccharide stimulated human monocytic cells', 'hiv infected human monocytic cells', 'k562 human erythroleukemia cells', 'thp 1 human monocytoid cells', 'transfected human colonic carcinoma cell line ht29 activates transcription', 'cells transcriptional activation', 'human prostatic epithelial cells', 'human monocytic cells', 'human thp 1 promonocytic cells', 'human allergen specific th2 cells', 'human tonsillar mononuclear cells', 'human k562 cells', 'human b lymphocyte precursor cells', 'u 937 human promonocytic cells', 'human promonocytic u 937 cells', 'human mononuclear cells', 'human epithelial cells', 'human th2 cells', 'human naive cells', 'human differentiated cells', 'murine baf3 cells involves activation', 'human lymphoblastoid cells', 'human dendritic cells', 'transcription factor ap 2 activates gene expression', 'human glomerular mesangial cells', 'human glial cells', 'human erythroleukemia cells', 'cotransfected human cells', 'human accessory cells', 'human jurkat t cells', 'ad transformed human cells', 'transcription factor activation protein', 'human nk t cells', 'human t lymphoblastoid cells', 'human monoblastic cells', 'human erythroleukemic cells']
    #test_set = ["human histiocytic U937 cells mRNA"]
    #test_set = [normalise_term(test_term) for test_term in test_set]
    #print(term_precision(gs_terms, set(test_set)))
    #print(normalise_term("human interleukin-2 receptor alpha gene ["))
    #print(gs_terms)
    #result = pre_processing_text_with_spacy("Filarial antigen induces increased expression of alternative activation genes in monocytes from patients with AFI.")
    #print(result)
    from jgtextrank import keywords_extraction_from_tagged_corpus
    pre_processed_corpus = pre_processing_corpus_with_spacy("GENIAcorpus302/text/files")
    spacy_pos_categories = {'NOUN', 'ADJ'}
    for sentence in pre_processed_corpus:
        print(sentence)
    #results, top_vertices = keywords_extraction_from_tagged_corpus(pre_processed_corpus, top_p = 0.3, syntactic_categories = spacy_pos_categories)
    #print("extracted keywords from pre-tagged content:"+ str(results))
    #print("top_vertices: ", top_vertices)

    #print(remove_punctuations("STAT and IFN regulatory factor (IRF) family transcription factor"))
    #print(normalise_term("STAT and IFN regulatory factor (IRF) family transcription factor"))