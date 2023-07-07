import numpy as np
import tqdm
import gensim
import gensim.similarities
from . import _Models
from .. import _BasicKits


class SimilarityTfidf:

    def __init__(self,
                 path_sentences_dataset_txt,
                 path_sentences_dataset_index_txt
                 ):
        self.doc_text_corpus_list, self.doc_ids_list, self.N_doc = \
            _BasicKits.FileT.l1_sentence_to_doc_level_corpus(path_sentences_dataset_txt,
                                                             path_sentences_dataset_index_txt)

        self.dictionary, self.tfidf_model = _Models.train_tfidf_model_dictmod(
            text_list=self.doc_text_corpus_list
        )

    @_BasicKits._BasicFuncT.timer_wrapper
    def similarity_matrix(self, x_text_list, y_text_list, y_chunksize=None, model_output_prefix='sim',
                          dtype=np.float32):
        """
        Args:
            x_text_list: the first text list to input
            y_text_list: the second text list to input for compare the similarity
            y_chunksize: chunksize which deal with onetime, It would only work with y
            model_output_prefix: the prefix of saved model

        Returns: shape of [X,Y] for dim 0 is X and dim 1 is y. Cosine similarity

        """
        x_text_tokenize_list = [doc.split() for doc in x_text_list]
        y_text_tokenize_list = [doc.split() for doc in y_text_list]

        # Convert tokenized documents to bag-of-words vectors (text corpus to num corpus...)
        x_bow_corpus = [self.dictionary.doc2bow(text) for text in x_text_tokenize_list]
        y_bow_corpus = [self.dictionary.doc2bow(text) for text in y_text_tokenize_list]

        # Convert to tf-idf vectors
        x_tfidf_corpus = self.tfidf_model[x_bow_corpus]
        y_tfidf_corpus = self.tfidf_model[y_bow_corpus]

        # Create a Similarity object to compute pairwise similarities
        index = gensim.similarities.Similarity(output_prefix=model_output_prefix,
                                               corpus=x_tfidf_corpus,
                                               num_features=len(self.dictionary))

        if not (y_chunksize is None):
            chunkslice_arr = np.arange(0, len(y_tfidf_corpus), y_chunksize)

            sim_mat_list = []
            for sli in tqdm.tqdm(range(len(chunkslice_arr))):
                sim_mat_list.append(
                    np.array(index[
                                 y_tfidf_corpus[
                                 chunkslice_arr[sli]:chunkslice_arr[sli] + y_chunksize
                                 ]
                             ], dtype=dtype).T
                )

            sim_matrix = np.concatenate(sim_mat_list, axis=1)  # concat axis 1

        else:
            # Compute similarity matrix
            sim_matrix = np.array(index[y_tfidf_corpus], dtype=dtype).T

        return sim_matrix
