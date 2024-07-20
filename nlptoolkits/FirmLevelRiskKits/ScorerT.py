import copy
import re
import typing
from ..StanzaKits import CoreNLPServerPack
import nltk
import nltk.tokenize
import pathos
import tqdm
from .. import _BasicKits
import math
import collections


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
               topic_ngram_weighted_map_dict: dict,
               subject_word_set_dict: dict,
               is_scale_by_totalwords: bool,
               binary_transformation_subjects: typing.Optional[typing.Iterable],
               scale_multiplier: int
               ):
        """
        :param texture: [str] texture that cleaned by the PreprocessT
        :param topic_ngram_weighted_map_dict: typing.Dict[typing.Dict] For example, Political.
            Firstly you have to collect a list of N-grams from DictionaryT.NgramDictionaryBuilder.
            Then for each topic, you save them to a dictionary.
            The format is strict. you should input data format like this
            {'politics':{('politic', 'lobby'):0.121, ('president', 'trump'): 0.002}}
        :param subject_word_set_dict: typing.Optional[set]
            the subject(s) you need to dig in under the restriction/conditional of the topic.
            The format is strict. you should input data format like this
            {'risk': set('risk', 'endanger', 'imperil')}
        :param is_scale_by_totalwords:<THIS VAR IS NOT APPLY TO total_words>
        is or not scale the output other than totalwords to <X * 1/totalwords>
        :param binary_transformation_subjects: the subject names that will be weighted when count for the
            topic_subject_score_{tk}_{sk}.
            For example. in a window that contains more than 1 subjects, like X, are found,
            the window will count the score by X * bigram-weighted-score and add to total score.
            But when binary_transformation_subjects include it, X will always be 1 if it is not 0.
        :param scale_multiplier: <THIS VAR IS NOT APPLY TO total_words>
            scale up and down the result by multiply this number with the value. The total number will not be applicable
        :return: total_words, subject_count, topic_ngram_count, topic_ngram_count_weighted, topic_ngram_subject_score

        """

        # check input
        if binary_transformation_subjects:
            assert isinstance(binary_transformation_subjects, typing.Iterable) and \
                   (not isinstance(binary_transformation_subjects, str))

            assert set(binary_transformation_subjects).issubset(set(subject_word_set_dict.keys()))
        else:
            binary_transformation_subjects = set()

        windows, words = self._ngram_window(texture)

        _result_dict = dict()

        _result_dict['total_words'] = len(words)

        for sk, svalue in subject_word_set_dict.items():
            _result_dict[f'subject_count_{sk}'] = len(
                [word for word in words if word in svalue]
            )

        # The number of
        for tk in topic_ngram_weighted_map_dict.keys():
            # Collect results
            _result_dict[f'topic_ngram_count_{tk}'] = 0
            _result_dict[f'topic_ngram_count_weighted_{tk}'] = 0

            for sk in subject_word_set_dict.keys():
                _result_dict[f'topic_subject_score_{tk}_{sk}'] = 0

        for i, window in enumerate(windows):

            # Find middle topic_ngram and check whether a "political" topic_ngram
            middle_topic_ngram = window[self.windows_center]

            # should be
            for tk, tvaluedict in topic_ngram_weighted_map_dict.items():

                topic_ngram_tuple_list = list(tvaluedict.keys())

                topic_ngram_weighted_list = list(tvaluedict.values())

                if middle_topic_ngram not in topic_ngram_tuple_list:
                    continue
                topic_ngram_weighted = topic_ngram_weighted_list[i]

                # Create word list for easy and quick access
                window_words = set([y for x in window for y in x])

                _result_dict[f'topic_ngram_count_{tk}'] += 1
                _result_dict[f'topic_ngram_count_weighted_{tk}'] += topic_ngram_weighted

                for sk, svalue in subject_word_set_dict.items():
                    # If yes, check whether risk synonym in window

                    # binary transformation
                    if sk in set(binary_transformation_subjects):
                        topic_ngram_subject_count = (len([word for word in window_words if word in svalue]) > 0)
                        _result_dict[
                            f'topic_subject_score_{tk}_{sk}'] += topic_ngram_subject_count * topic_ngram_weighted
                    else:
                        # else go through
                        topic_ngram_subject_count = len([word for word in window_words if word in svalue])
                        _result_dict[
                            f'topic_subject_score_{tk}_{sk}'] += topic_ngram_subject_count * topic_ngram_weighted

        if is_scale_by_totalwords:
            _result_dict_weighted = dict()
            for k, v in _result_dict.items():
                if k == 'total_words':
                    _result_dict_weighted.update({k: v})
                else:
                    _result_dict_weighted.update({k: v / _result_dict['total_words']})

        else:
            _result_dict_weighted = _result_dict

        # scale the value
        _result_dict_weighted_scaled = dict()

        for k, v in _result_dict_weighted.items():
            if k == 'total_words':
                _result_dict_weighted_scaled.update({k: v})
            else:
                _result_dict_weighted_scaled.update({k: v * scale_multiplier})

        return _result_dict_weighted_scaled

    def list_scorer(self,
                    texture_list: typing.List[str],
                    topic_ngram_weighted_map_dict: dict,
                    subject_word_set_dict: dict,
                    is_scale_by_totalwords: bool,
                    binary_transformation_subjects: typing.Optional[typing.Iterable],
                    scale_multiplier: int = 100000
                    ):
        """
        :param texture_list: List[str] texture that cleaned by the PreprocessT
        :param topic_ngram_weighted_map_dict: typing.Dict[typing.Dict] For example, Political.
            Firstly you have to collect a list of N-grams from DictionaryT.NgramDictionaryBuilder.
            Then for each topic, you save them to a dictionary.
            The format is strict. you should input data format like this
            {'politics':{('politic', 'lobby'):0.121, ('president', 'trump'): 0.002}}
        :param subject_word_set_dict: typing.Optional[set]
            the subject(s) you need to dig in under the restriction/conditional of the topic.
            The format is strict. you should input data format like this
            {'risk': set('risk', 'endanger', 'imperil')}
        :param is_scale_by_totalwords:<THIS VAR IS NOT APPLY TO total_words>
        is or not scale the output other than totalwords to <X * 1/totalwords>
        :param binary_transformation_subjects: the subject names that will be weighted when count for the
            topic_subject_score_{tk}_{sk}.
            For example. in a window that contains more than 1 subjects, like X, are found,
            the window will count the score by X * bigram-weighted-score and add to total score.
            But when binary_transformation_subjects include it, X will always be 1 if it is not 0.
        :param scale_multiplier: <THIS VAR IS NOT APPLY TO total_words>
            scale up and down the result by multiply this number with the value. The total number will not be applicable
        :return: total_words, subject_count, topic_ngram_count, topic_ngram_count_weighted, topic_ngram_subject_score

        """

        def _worker(texture):
            return self.scorer(
                texture=texture,
                topic_ngram_weighted_map_dict=topic_ngram_weighted_map_dict,
                subject_word_set_dict=subject_word_set_dict,
                is_scale_by_totalwords=is_scale_by_totalwords,
                binary_transformation_subjects=binary_transformation_subjects,
                scale_multiplier=scale_multiplier
            )

        _rst_l = []

        with pathos.multiprocessing.Pool(
                initializer=_BasicKits._BasicFuncT.processes_interrupt_initiator,
                processes=self.processes
        ) as pool:
            for rsts in tqdm.tqdm(
                    pool.imap(_worker, texture_list),
                    total=len(texture_list)
            ):
                _rst_l.append(rsts)

        return _rst_l
