import copy
import pickle
import warnings
import math
from collections import Counter, OrderedDict, defaultdict
from functools import partial
import pathos
from operator import itemgetter
import numpy as np
import pandas as pd
import tqdm
from sklearn import preprocessing
import gensim
import typing
# import basic funcs
from .. import _BasicKits

"""
WARNINGS: THIS PACKAGE USE GENSIM MODEL!
"""
# --------------------------------------------------------------------------
# l0 level functions/classes
# --------------------------------------------------------------------------
"""dictionary/scorer making function"""


def expand_words_dimension_mean(
        word2vec_model,
        seed_words,
        n=50,
        restrict=None,
        min_similarity=0,
        filter_word_set=None,
):
    """For each dimensional mean vector, search for the closest n words


    Arguments:
        word2vec_model {gensim.models.word2vec} -- a gensim word2vec model
        seed_words {dict[str, list]} -- seed word dict of {dimension: [words]}

    Keyword Arguments:
        n {int} -- number of expanded words in each dimension (default: {50})
        restrict {float} -- whether to restrict the search to a fraction of most frequent words in vocab (default: {None})
        min_similarity {int} -- minimum cosine similarity to the seeds for a word to be included (default: {0})
        filter_word_set {set} -- do not include the words in this set to the expanded dictionary (default: {None})

    Returns:
        dict[str, set] -- expanded words, a dict of {dimension: set([words])}
    """
    vocab_number = len(word2vec_model.wv.vocab)
    expanded_words = {}
    all_seeds = set()
    for dim in seed_words.keys():
        all_seeds.update(seed_words[dim])
    if restrict != None:
        restrict = int(vocab_number * restrict)
    for dimension in seed_words:
        dimension_words = [
            word for word in seed_words[dimension] if word in word2vec_model.wv.vocab
        ]
        if len(dimension_words) > 0:
            similar_words = [
                pair[0]
                for pair in word2vec_model.wv.most_similar(
                    dimension_words, topn=n, restrict_vocab=restrict
                )
                if pair[1] >= min_similarity and pair[0] not in all_seeds
            ]
        else:
            similar_words = []
        if filter_word_set is not None:
            similar_words = [x for x in similar_words if x not in filter_word_set]
        similar_words = [
            x for x in similar_words if "[ner:" not in x
        ]  # filter out NERs
        expanded_words[dimension] = similar_words
    for dim in expanded_words.keys():
        expanded_words[dim] = expanded_words[dim] + seed_words[dim]
    for d, i in expanded_words.items():
        expanded_words[d] = set(i)
    return expanded_words


def rank_by_sim(expanded_words, seed_words, model) -> "dict[str: list]":
    """ Rank each dim in a dictionary based on similarity to the seend words mean
    Returns: expanded_words_sorted {dict[str:list]}
    """
    expanded_words_sorted = dict()
    for dimension in expanded_words.keys():
        dimension_seed_words = [
            word for word in seed_words[dimension] if word in model.wv.vocab
        ]
        similarity_dict = dict()
        for w in expanded_words[dimension]:
            if w in model.wv.vocab:
                similarity_dict[w] = model.wv.n_similarity(dimension_seed_words, [w])
            else:
                # print(w + "is not in w2v model")
                pass
        sorted_similarity_dict = sorted(
            similarity_dict.items(), key=itemgetter(1), reverse=True
        )
        sorted_similarity_list = [x[0] for x in sorted_similarity_dict]
        expanded_words_sorted[dimension] = sorted_similarity_list
    return expanded_words_sorted


def deduplicate_keywords(word2vec_model, expanded_words, seed_words):
    """
    If a word cross-loads, choose the most similar dimension. Return a deduplicated dict.
    """
    word_counter = Counter()

    for dimension in expanded_words:
        word_counter.update(list(expanded_words[dimension]))
    for dimension in seed_words:
        for w in seed_words[dimension]:
            if w not in word2vec_model.wv.vocab:
                seed_words[dimension].remove(w)

    word_counter = {k: v for k, v in word_counter.items() if v > 1}  # duplicated words
    dup_words = set(word_counter.keys())
    for dimension in expanded_words:
        expanded_words[dimension] = expanded_words[dimension].difference(dup_words)

    for word in list(dup_words):
        sim_w_dim = {}
        for dimension in expanded_words:
            dimension_seed_words = [
                word
                for word in seed_words[dimension]
                if word in word2vec_model.wv.vocab
            ]
            # sim_w_dim[dimension] = max([word2vec_model.wv.n_similarity([word], [x]) for x in seed_words[dimension]] )
            sim_w_dim[dimension] = word2vec_model.wv.n_similarity(
                dimension_seed_words, [word]
            )
        max_dim = max(sim_w_dim, key=sim_w_dim.get)
        expanded_words[max_dim].add(word)

    for dimension in expanded_words:
        expanded_words[dimension] = sorted(expanded_words[dimension])

    return expanded_words


def score_one_document_tf(document, expanded_words, list_of_list=False):
    """score a single document using term freq, the dimensions are sorted alphabetically

    Arguments:
        document {str} -- a document
        expanded_words {dict[str, set(str)]} -- an expanded dictionary

    Keyword Arguments:
        list_of_list {bool} -- whether the document is splitted (default: {False})

    Returns:
        [int] -- a list of : dim1, dim2, ... , document_length
    """
    if list_of_list is False:
        document = document.split()
    dimension_count = OrderedDict()
    for dimension in expanded_words:
        dimension_count[dimension] = 0
    c = Counter(document)
    for pair in c.items():
        for dimension, words in expanded_words.items():
            if pair[0] in words:
                dimension_count[dimension] += pair[1]
    # use ordereddict to maintain order of count for each dimension
    dimension_count = OrderedDict(sorted(dimension_count.items(), key=lambda t: t[0]))
    result = list(dimension_count.values())
    result.append(len(document))
    return result


def tf_scorer(documents, document_ids, expanded_words, n_core=1):
    """score using term freq for documents, the dimensions are sorted alphabetically

    Arguments:
        documents {[str]} -- list of documents
        document_ids {[str]} -- list of document IDs
        expanded_words {dict[str, set(str)]} -- dictionary for scoring

    Keyword Arguments:
        n_core {int} -- number of CPU cores (default: {1})

    Returns:
        pandas.DataFrame -- a dataframe with columns: Doc_ID, dim1, dim2, ..., document_length
    """
    if n_core > 1:
        count_one_document_partial = partial(
            score_one_document_tf, expanded_words=expanded_words, list_of_list=False
        )

        with pathos.multiprocessing.Pool(processes=n_core,
                                         initializer=_BasicKits._BasicFuncT.threads_interrupt_initiator
                                         ) as pool:

            results = list(pool.map(count_one_document_partial, documents))

    else:
        results = []
        for i, doc in enumerate(documents):
            results.append(
                score_one_document_tf(doc, expanded_words, list_of_list=False)
            )
    df = pd.DataFrame(
        results, columns=sorted(list(expanded_words.keys())) + ["document_length"]
    )
    df["Doc_ID"] = document_ids
    return df


def tf_idf_scorer(
        documents,
        document_ids,
        expanded_words,
        df_dict,
        N_doc,
        method="TFIDF",
        word_weights=None,
        normalize=False,
):
    """Calculate tf-idf score for documents

    Arguments:
        documents {[str]} -- list of documents (strings)
        document_ids {[str]} -- list of document ids
        expanded_words {{dim: set(str)}}} -- dictionary
        df_dict {{str: int}} -- a dict of {word:freq} that provides document frequencey of words
        N_doc {int} -- number of documents

    Keyword Arguments:
        method {str} --
            TFIDF: conventional tf-idf
            WFIDF: use wf-idf log(1+count) instead of tf in the numerator
            TFIDF/WFIDF+SIMWEIGHT: using additional word weights given by the word_weights dict
            (default: {TFIDF})
        normalize {bool} -- normalized the L2 norm to one for each document (default: {False})
        word_weights {{word:weight}} -- a dictionary of word weights (e.g. similarity weights) (default: None)

    Returns:
        [df] -- a dataframe with columns: Doc_ID, dim1, dim2, ..., document_length
        [contribution] -- a dict of total contribution (sum of scores in the corpus) for each word
    """
    print("Scoring using {}".format(method))
    contribution = defaultdict(int)
    results = []
    for i, doc in enumerate(tqdm.tqdm(documents)):
        document = doc.split()
        dimension_count = OrderedDict()
        for dimension in expanded_words:
            dimension_count[dimension] = 0
        c = Counter(document)
        for pair in c.items():
            for dimension, words in expanded_words.items():
                if pair[0] in words:
                    if method == "WFIDF":
                        w_ij = (1 + math.log(pair[1])) * math.log(
                            N_doc / df_dict[pair[0]]
                        )
                    elif method == "TFIDF":
                        w_ij = pair[1] * math.log(N_doc / df_dict[pair[0]])
                    elif method == "TFIDF+SIMWEIGHT":
                        w_ij = (
                                pair[1]
                                * word_weights[pair[0]]
                                * math.log(N_doc / df_dict[pair[0]])
                        )
                    elif method == "WFIDF+SIMWEIGHT":
                        w_ij = (
                                (1 + math.log(pair[1]))
                                * word_weights[pair[0]]
                                * math.log(N_doc / df_dict[pair[0]])
                        )
                    else:
                        raise Exception(
                            "The method can only be TFIDF, WFIDF, TFIDF+SIMWEIGHT, or WFIDF+SIMWEIGHT"
                        )
                    dimension_count[dimension] += w_ij
                    contribution[pair[0]] += w_ij / len(document)
        dimension_count = OrderedDict(
            sorted(dimension_count.items(), key=lambda t: t[0])
        )
        result = list(dimension_count.values())
        result.append(len(document))
        results.append(result)
    results = np.array(results)
    # normalize the length of tf-idf vector
    if normalize:
        results[:, : len(expanded_words.keys())] = preprocessing.normalize(
            results[:, : len(expanded_words.keys())]
        )
    df = pd.DataFrame(
        results, columns=sorted(list(expanded_words.keys())) + ["document_length"]
    )
    df["Doc_ID"] = document_ids
    return df, contribution


def compute_word_sim_weights(file_name):
    """Compute word weights in each dimension.
    Default weight is 1/ln(1+rank). For example, 1st word in each dim has weight 1.44,
    10th word has weight 0.41, 100th word has weigh 0.21.

    Arguments:
        file_name {str} -- expanded dictionary file

    Returns:
        sim_weights {{word:weight}} -- a dictionary of word weights
    """
    culture_dict_df = pd.read_csv(file_name, index_col=None)
    culture_dict = culture_dict_df.to_dict("list")
    sim_weights = {}
    for k in culture_dict.keys():
        culture_dict[k] = [x for x in culture_dict[k] if x == x]  # remove nan
    for key in culture_dict:
        for i, w in enumerate(culture_dict[key]):
            sim_weights[w] = 1 / math.log(1 + 1 + i)
    return sim_weights


"""AUTO result maker"""

# --------------------------------------------------------------------------
# Word2vec model AUTO function
# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
# l1 level Functions
# --------------------------------------------------------------------------
"""word2vec model function"""


def l1_semi_supervise_w2v_dict(
        path_input_w2v_model,
        seed_words_dict,
        restrict_vocab_per: typing.Optional[float],
        model_dims: int,

):
    """word dictionary semi-supervised under word2vec model, auto function"""
    model = gensim.models.Word2Vec.load(path_input_w2v_model)

    vocab_number = len(model.wv.vocab)

    print("Vocab size in the w2v model: {}".format(vocab_number))

    # expand dictionary
    expanded_words_dict = expand_words_dimension_mean(
        word2vec_model=model,
        seed_words=seed_words_dict,
        restrict=restrict_vocab_per,
        n=model_dims,
    )

    # make sure that one word only loads to one dimension
    expanded_words_dict = deduplicate_keywords(
        word2vec_model=model,
        expanded_words=expanded_words_dict,
        seed_words=seed_words_dict,
    )

    # rank the words under each dimension by similarity to the seed words
    expanded_words_dict = rank_by_sim(
        expanded_words_dict, seed_words_dict, model
    )
    # output the dictionary
    return expanded_words_dict


# --------------------------------------------------------------------------
# l0 level Classes
# --------------------------------------------------------------------------

class DocScorer:

    def __init__(self, path_current_dict,
                 path_trainw2v_sentences_dataset_txt,
                 path_trainw2v_sentences_dataset_index_txt,
                 processes
                 ):
        """
        Args:
            path_current_dict: path of current trained dict, already finished
            path_trainw2v_sentences_dataset_txt: path of the dataset to train the word2vec model
            path_trainw2v_sentences_dataset_index_txt: path of the dataset's IDS to train the word2vec model
            processes: Ncores to run
        """

        self.mp_threads = processes

        self.current_dict_path = str(path_current_dict)

        self.current_dict, self.all_dict_words = _BasicKits.FileT.read_dict_dictname_from_csv_dictset(
            self.current_dict_path
        )
        # words weighted by similarity rank (optional)
        self.word_sim_weights = compute_word_sim_weights(self.current_dict_path)

        """create doc level data"""

        self.sent_corpus_file = path_trainw2v_sentences_dataset_txt
        self.sent_id_file = path_trainw2v_sentences_dataset_index_txt

        self.doc_corpus, self.doc_ids, self.N_doc = \
            _BasicKits.FileT.l1_sentence_to_doc_level_corpus(self.sent_corpus_file, self.sent_id_file)

        """create doc freq dict"""
        self.doc_freq_dict = _BasicKits.FileT.calculate_doc_freq_dict(self.doc_corpus)

    """pickle the data"""

    def pickle_doc_level_corpus(self, save_pickle_path):

        if not str(save_pickle_path).endswith('.pickle'):
            raise ValueError('must endswith .pickle')

        with open(
                save_pickle_path,
                "wb",
        ) as out_f:
            pickle.dump(copy.deepcopy(self.doc_corpus), out_f)

    def pickle_doc_level_ids(self, save_pickle_path):

        if not str(save_pickle_path).endswith('.pickle'):
            raise ValueError('must endswith .pickle')

        with open(
                save_pickle_path,
                "wb",
        ) as out_f:
            pickle.dump(copy.deepcopy(self.doc_ids), out_f)

    def pickle_doc_freq(self, save_pickle_path):

        if not str(save_pickle_path).endswith('.pickle'):
            raise ValueError('must endswith .pickle')

        with open(
                save_pickle_path,
                "wb",
        ) as out_f:
            pickle.dump(copy.deepcopy(self.doc_freq_dict), out_f)

    def score_tf_df(self):
        """
        :return : score_df
        """
        score_df = tf_scorer(
            documents=self.doc_corpus,
            document_ids=self.doc_ids,
            expanded_words=self.current_dict,
            n_core=self.mp_threads,
        )

        return score_df

    """Scorer at doc level"""

    def score_tfidf_dfdf(self, method, normalize=False):
        """Score documents using tf-idf and its variations

        :param method :
                TFIDF: conventional tf-idf
                WFIDF: use wf-idf log(1+count) instead of tf in the numerator
                TFIDF/WFIDF+SIMWEIGHT: using additional word weights given by the word_weights dict
            expanded_dict {dict[str, set(str)]} -- expanded dictionary
        :return : score_df, word_contributions_df in tuple
        """
        if method == "TF":

            raise ValueError("TF-IDF method could not compat with TF method")

        else:
            print("Scoring TF-IDF.")
            # score tf-idf
            score_df, contribution_dict = tf_idf_scorer(
                documents=self.doc_corpus,
                document_ids=self.doc_ids,
                expanded_words=self.current_dict,
                df_dict=self.doc_freq_dict,
                N_doc=self.N_doc,
                word_weights=self.word_sim_weights,
                method=method,
                normalize=normalize,
            )

            # save word contributions
            word_contributions_df = pd.DataFrame.from_dict(contribution_dict, orient="index")

            return score_df, word_contributions_df

    """score contribution"""

    def score_contribution_df(self):
        """output contribution dict to Excel file
        Arguments:
            contribution_dict {word:contribution} -- a pre-calculated contribution dict for each word in expanded dictionary
            out_file {str} -- file name (Excel)
        """

        warnings.warn('score_contribution_df IS a unstable function, its result is unknown and not explainable',
                      FutureWarning
                      )

        # PART1-> count for conrtibution_dict

        contribution_TF = defaultdict(int)
        contribution_WFIDF = defaultdict(int)
        contribution_TFIDF = defaultdict(int)
        contribution_TFIDF_SIMWEIGHT = defaultdict(int)
        contribution_WFIDF_SIMWEIGHT = defaultdict(int)
        for i, doc in enumerate(tqdm.tqdm(self.doc_corpus)):
            document = doc.split()
            c = Counter(document)
            for pair in c.items():
                if pair[0] in self.all_dict_words:
                    contribution_TF[pair[0]] += pair[1]
                    w_ij = (1 + math.log(pair[1])) * math.log(self.N_doc / self.doc_freq_dict[pair[0]])
                    contribution_WFIDF[pair[0]] += w_ij
                    w_ij = pair[1] * math.log(self.N_doc / self.doc_freq_dict[pair[0]])
                    contribution_TFIDF[pair[0]] += w_ij
                    w_ij = (
                            pair[1] * self.word_sim_weights[pair[0]] * math.log(
                        self.N_doc / self.doc_freq_dict[pair[0]])
                    )
                    contribution_TFIDF_SIMWEIGHT[pair[0]] += w_ij
                    w_ij = (
                            (1 + math.log(pair[1]))
                            * self.word_sim_weights[pair[0]]
                            * math.log(self.N_doc / self.doc_freq_dict[pair[0]])
                    )
                    contribution_WFIDF_SIMWEIGHT[pair[0]] += w_ij
        contribution_dict = {
            "TF": contribution_TF,
            "TFIDF": contribution_TFIDF,
            "WFIDF": contribution_WFIDF,
            "TFIDF+SIMWEIGHT": contribution_TFIDF_SIMWEIGHT,
            "WFIDF+SIMWEIGHT": contribution_WFIDF_SIMWEIGHT,
        }

        # PART2 -> output contribution dict to Excel file

        contribution_lst = []
        for dim in self.current_dict:
            for w in self.current_dict[dim]:
                w_dict = {}
                w_dict["dim"] = dim
                w_dict["word"] = w
                w_dict["contribution"] = contribution_dict[w]
                contribution_lst.append(w_dict)

        contribution_df = pd.DataFrame(contribution_lst)
        dim_dfs = []
        for dim in sorted(self.current_dict.keys()):
            dim_df = (
                contribution_df[contribution_df.dim == dim]
                .sort_values(by="contribution", ascending=False)
                .reset_index(drop=True)
            )
            dim_df["total_contribution"] = dim_df["contribution"].sum()
            dim_df["relative_contribuion"] = dim_df["contribution"].div(
                dim_df["total_contribution"]
            )
            dim_df["cumulative_contribution"] = dim_df["relative_contribuion"].cumsum()
            dim_df = dim_df.drop(["total_contribution"], axis=1)
            dim_dfs.append(dim_df)

        return pd.concat(dim_dfs, axis=1)
