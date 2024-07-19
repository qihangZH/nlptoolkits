import re
import typing
from ..StanzaKits import CoreNLPServerPack
import nltk
import nltk.tokenize
import pathos
import tqdm
from .. import _BasicKits
import math


class NgramScorer:

    def __init__(self,
                 processes: int,
                 token_sep_string: str = CoreNLPServerPack._GlobalArgs.DEFAULT_TOKEN_SEP_STRING,
                 n: int = 2,
                 window_size: int = 20
                 ):
        self.processes = processes
        self.token_sep_string = token_sep_string
        self._N = n
        self.window_size = window_size
        self.windows_center = math.floor(self.window_size / 2)

    def _ngram_window(self, texture: str):

        words = texture.split(self.token_sep_string)
        # ngrams
        # ngrams = [' '.join(x) for x in zip(words[0:], words[1:])]
        ngrams = list(nltk.ngrams(words, n=self._N))

        # Window of +/- 10 consecutive ngrams
        windows = list(zip(*[ngrams[i:] for i in range(self.window_size + 1)]))

        return windows, words

    def scorer(self,
               texture: str,
               topic_ngram_tuple_list: typing.Optional[typing.List[tuple]],
               topic_ngram_weighted_list: typing.Optional[typing.List[str]],
               subject_word_set: typing.Optional[set],
               is_scale_by_totalwords: bool
               ):
        """
        :param texture: [str] texture that cleaned by the PreprocessT
        :param topic_ngram_tuple_list: typing.Optional[typing.List[str]] list of N-grams about a topic. For example, Political.
            Firstly you have to collect a list of N-grams from DictionaryT.NgramDictionaryBuilder.
            Like [('tax', 'haven')]
        :param topic_ngram_weighted_list: typing.Optional[typing.List[str]] list of weighted score of each N-gram
            that about a topic. If could be count by tfidf or tf
        :param subject_word_set: the subject you need to dig in under the restriction/conditional of the topic.
            This should be work with a set of subject words. Like positive words. Negative words, Risk words, etc.
        :param is_scale_by_totalwords: is or not scale the output other than totalwords to <X * 1/totalwords>
        :return: total_words, subject_count, topic_ngram_count, topic_ngram_count_weighted, topic_ngram_subject_score

        """
        assert len(topic_ngram_tuple_list) == len(topic_ngram_weighted_list)

        windows, words = self._ngram_window(texture)

        total_words = len(words)

        subject_count = len([word for word in words if word in subject_word_set])

        # conditional_score
        topic_ngram_count = 0

        topic_ngram_count_weighted = 0

        topic_ngram_subject_score = 0

        for i, window in enumerate(windows):

            # Find middle topic_ngram and check whether a "political" topic_ngram
            middle_topic_ngram = window[self.windows_center]
            if middle_topic_ngram not in topic_ngram_tuple_list:
                continue
            topic_ngram_weighted = topic_ngram_weighted_list[i]

            # Create word list for easy and quick access
            window_words = set([y for x in window for y in x])

            # If yes, check whether risk synonym in window
            topic_ngram_subject_count = (len([word for word in window_words
                                              if word in subject_word_set]) > 0)

            # Collect results
            topic_ngram_count += 1
            topic_ngram_count_weighted += topic_ngram_weighted
            topic_ngram_subject_score += topic_ngram_subject_count * topic_ngram_weighted

        if is_scale_by_totalwords:
            return total_words, \
                subject_count / total_words, \
                topic_ngram_count / total_words, \
                topic_ngram_count_weighted / total_words, \
                topic_ngram_subject_score / total_words
        else:
            return total_words, subject_count, topic_ngram_count, topic_ngram_count_weighted, topic_ngram_subject_score

    def list_scorer(self,
                    texture_list: typing.List[str],
                    topic_ngram_tuple_list: typing.Optional[typing.List[str]],
                    topic_ngram_weighted_list: typing.Optional[typing.List[str]],
                    subject_word_set: typing.Optional[set],
                    is_scale_by_totalwords: bool
                    ):
        """
        :param texture_list: [str] texture list that cleaned by the PreprocessT
        :param topic_ngram_tuple_list: typing.Optional[typing.List[str]] list of N-grams about a topic. For example, Political.
            Firstly you have to collect a list of N-grams from DictionaryT.NgramDictionaryBuilder.
        :param topic_ngram_weighted_list: typing.Optional[typing.List[str]] list of weighted score of each N-gram
            that about a topic. If could be count by tfidf or tf
        :param subject_word_set: the subject you need to dig in under the restriction/conditional of the topic.
            This should be work with a set of subject words. Like positive words. Negative words, Risk words, etc.
        :param is_scale_by_totalwords: is or not scale the output other than totalwords to <X * 1/totalwords>
        :return: total_words, subject_count, topic_ngram_count, topic_ngram_count_weighted, topic_ngram_subject_score
        """

        def _worker(texture):
            return self.scorer(
                texture=texture,
                topic_ngram_tuple_list=topic_ngram_tuple_list,
                topic_ngram_weighted_list=topic_ngram_weighted_list,
                subject_word_set=subject_word_set,
                is_scale_by_totalwords=is_scale_by_totalwords
            )

        total_words_l = []
        subject_count_l = []
        topic_ngram_count_l = []
        topic_ngram_count_weighted_l = []
        topic_ngram_subject_score_l = []

        with pathos.multiprocessing.Pool(
                initializer=_BasicKits._BasicFuncT.processes_interrupt_initiator,
                processes=self.processes
        ) as pool:
            for rsts in tqdm.tqdm(
                    pool.imap(_worker, texture_list),
                    total=len(texture_list)
            ):
                total_words_l.append(rsts[0])
                subject_count_l.append(rsts[1])
                topic_ngram_count_l.append(rsts[2])
                topic_ngram_count_weighted_l.append(rsts[3])
                topic_ngram_subject_score_l.append(rsts[4])

        return {
            'total_words': total_words_l,
            'subject_count': subject_count_l,
            'topic_ngram_count': topic_ngram_count_l,
            'topic_ngram_count_weighted': topic_ngram_count_weighted_l,
            'topic_ngram_subject_score': topic_ngram_subject_score_l
        }
