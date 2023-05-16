from collections import defaultdict
import tqdm
import pandas as pd
import copy
import pickle
from . import _dictionary
from . import _file_util


def construct_doc_level_corpus(sent_corpus_file, sent_id_file):
    """Construct document level corpus from sentence level corpus and write to disk.
    Dump "corpus_doc_level.pickle" and "doc_ids.pickle" to Path(global_options.OUTPUT_FOLDER, "scores", "temp").

    Arguments:
        sent_corpus_file {str or Path} -- The sentence corpus after parsing and cleaning, each line is a sentence
        sent_id_file {str or Path} -- The sentence ID file, each line correspond to a line in the sent_co(docID_sentenceID)

    Returns:
        [str], [str], int -- a tuple of a list of documents, a list of document IDs, and the number of documents
    """
    # sentence level corpus
    sent_corpus = _file_util.file_to_list(sent_corpus_file)
    sent_IDs = _file_util.file_to_list(sent_id_file)
    assert len(sent_IDs) == len(sent_corpus)
    # doc id for each sentence
    doc_ids = [x.split("_")[0] for x in sent_IDs]
    # concat all text from the same doc
    id_doc_dict = defaultdict(lambda: "")
    for i, id in enumerate(doc_ids):
        id_doc_dict[id] += " " + sent_corpus[i]
    # create doc level corpus
    corpus = list(id_doc_dict.values())
    doc_ids = list(id_doc_dict.keys())
    assert len(corpus) == len(doc_ids)
    N_doc = len(corpus)

    return corpus, doc_ids, N_doc


def calculate_doc_freq(corpus):
    """Calcualte and dump a document-freq dict for all the words.

    Arguments:
        corpus {[str]} -- a list of documents

    Returns:
        {dict[str: int]} -- document freq for each word
    """
    print("Calculating document frequencies.")
    # document frequency
    doc_freq_dict = defaultdict(int)
    for doc in tqdm.tqdm(corpus):
        doc_splited = doc.split()
        words_in_doc = set(doc_splited)
        for word in words_in_doc:
            doc_freq_dict[word] += 1

    return doc_freq_dict


class DocScorer:

    def __init__(self, path_current_dict, path_trainw2v_dataset_txt, path_trainw2v_dataset_index_txt, mp_threads):
        """
        Args:
            path_current_dict: path of current trained dict, already finished
            path_trainw2v_dataset_txt: path of the dataset to train the word2vec model
            path_trainw2v_dataset_index_txt: path of the dataset's IDS to train the word2vec model
            mp_threads: Ncores to run
        """

        self.mp_threads = mp_threads

        self.current_dict_path = str(path_current_dict)

        self.current_dict, self.all_dict_words = _dictionary.read_dict_from_csv(
            self.current_dict_path
        )
        # words weighted by similarity rank (optional)
        self.word_sim_weights = _dictionary.compute_word_sim_weights(self.current_dict_path)

        """create doc level data"""

        self.sent_corpus_file = path_trainw2v_dataset_txt
        self.sent_id_file = path_trainw2v_dataset_index_txt

        self.doc_corpus, self.doc_ids, self.N_doc = \
            construct_doc_level_corpus(self.sent_corpus_file, self.sent_id_file)

        """create doc freq dict"""
        self.doc_freq_dict = calculate_doc_freq(self.doc_corpus)

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
        score_df = _dictionary.score_tf(
            documents=self.doc_corpus,
            document_ids=self.doc_ids,
            expanded_words=self.current_dict,
            n_core=self.mp_threads,
        )

        return score_df

    """Scorer at doc level"""

    def score_tfidf_tupledf(self, method, normalize=False):
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
            score_df, contribution_dict = _dictionary.score_tf_idf(
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
