import bitermplus as btm
import numpy as np
import pandas as pd


class BtmTopic:

    def __init__(self, train_text_list, **kwargs):
        """
        :param train_text_list: the train text list to build the BTM topic model
        :param kwargs: the kwargs of sklearn.feature_extraction.text.CountVectorizer (the sklearn's embedder)
            https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
        :return: None
        """
        # PREPROCESSING
        # Obtaining terms frequency in a sparse matrix and corpus vocabulary
        self.train_X, self.vocabulary, self.vocab_dict = btm.get_words_freqs(train_text_list, **kwargs)
        # Vectorizing documents
        self.train_docs_vec = btm.get_vectorized_docs(train_text_list, self.vocabulary)
        # Generating biterms
        self.train_biterms = btm.get_biterms(self.train_docs_vec)
        # if not have been fit/train, then _model should be None
        self._model = None

    @property
    def model(self):

        if self._model:
            raise ValueError('The model has not been made, try to have BtmTopic.fit and initiate')

        else:
            return self._model

    def fit(self, fit_iterations=20, **kwargs):
        """
        :param fit_iterations: the iteration of btm.BTM().fit()'s iteration times, fit's iter times
        :param kwargs: the kwarg arguments of bitermplus.BTM
            https://bitermplus.readthedocs.io/en/latest/bitermplus.html#bitermplus.BTM
        """
        self._model = btm.BTM(self.train_X, self.vocabulary, **kwargs)
        self._model.fit(self.train_biterms, iterations=fit_iterations)

    def transform(self, text_list: list):
        new_docs_vec = btm.get_vectorized_docs(text_list, self.vocabulary)
        new_p_zd = self._model.transform(new_docs_vec)
        return new_p_zd

    def fit_transform(self, fit_iterations=20, **kwargs):
        """
        :param fit_iterations: the iteration of btm.BTM().fit()'s iteration times, fit's iter times
        :param kwargs: the kwarg arguments of bitermplus.BTM
        """
        self.fit(fit_iterations=fit_iterations, **kwargs)
        return self._model.transform(self.train_docs_vec)

    """Next part is about the utility function to pick something you need:"""

    def get_top_topic_words(self, words_num, topic_idx=None) -> pd.DataFrame:
        """
        Select top topic words from a fitted model.
        autofunction of https://bitermplus.readthedocs.io/en/latest/bitermplus.util.html#bitermplus.get_top_topic_words
        :return: Words with highest probabilities per each selected topic.
        """
        return btm.get_top_topic_words(self._model, words_num=words_num, topics_idx=topic_idx)

    def get_top_topic_docs(self, text_list: list, docs_num=20, topics_idx=None) -> pd.DataFrame:
        """
        Select top topic docs from a fitted model.
        https://bitermplus.readthedocs.io/en/latest/bitermplus.util.html#bitermplus.get_top_topic_docs
        :returns: Documents with highest probabilities in all selected topics.
        """
        return btm.get_top_topic_docs(
            docs=text_list,
            p_zd=self.transform(text_list),
            docs_num=docs_num,
            topics_idx=topics_idx
        )

    def get_docs_top_topic(self, text_list: list) -> pd.DataFrame:
        """
        Select most probable topic for each document.
        https://bitermplus.readthedocs.io/en/latest/bitermplus.util.html#bitermplus.get_docs_top_topic
        :returns: Documents and the most probable topic for each of them.
        """

        return btm.get_docs_top_topic(
            docs=text_list,
            p_zd=self.transform(text_list)
        )
