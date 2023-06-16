import numpy as np

from . import _Models
from .. import _BasicKits
import gensim
import gensim.similarities



class SimilarityTfidf:

    def __init__(self,
                 path_sentences_dataset_txt,
                 path_sentences_dataset_index_txt
                 ):
        self.doc_corpus_list, self.doc_ids, self.N_doc = \
            _BasicKits.FileT.l1_sentence_to_doc_level_corpus(path_sentences_dataset_txt,
                                                             path_sentences_dataset_index_txt)

        self.dictionary, self.tfidf_model = _Models.train_tfidf_model_tuple(
            text_list=self.doc_corpus_list
        )

    @_BasicKits._BasicFuncT.timer_wrapper
    def similarity_matrix(self, x_text_list, y_text_list):
        """
        Args:
            x_text_list: the first text list to input
            y_text_list: the second text list to input for compare the similarity

        Returns: shape of [X,Y] for dim 0 is X and dim 1 is y. Cosine similarity

        """
        x_text_tokenize_list = [doc.split() for doc in x_text_list]
        y_text_tokenize_list = [doc.split() for doc in y_text_list]

        # Convert tokenized documents to bag-of-words vectors
        x_bow_corpus = [self.dictionary.doc2bow(text) for text in x_text_tokenize_list]
        y_bow_corpus = [self.dictionary.doc2bow(text) for text in y_text_tokenize_list]

        # Convert to tf-idf vectors
        x_tfidf_corpus = self.tfidf_model[x_bow_corpus]
        y_tfidf_corpus = self.tfidf_model[y_bow_corpus]

        # Create a Similarity object to compute pairwise similarities
        index = gensim.similarities.Similarity(output_prefix='sim',
                                               corpus=x_tfidf_corpus,
                                               num_features=len(self.dictionary))

        # Compute similarity matrix
        sim_matrix = index[y_tfidf_corpus]

        return np.array(sim_matrix)
