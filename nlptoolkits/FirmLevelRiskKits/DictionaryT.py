"""This package is to help to build your own dictionary from Hassan et al.(2019)"""
import warnings

from ..StanzaKits import CoreNLPServerPack
from . import _Resources
import typing
import nltk
import nltk.tokenize
import re
import pathos
import tqdm
from collections import Counter
from .. import _BasicKits
from .. import resources


class NgramDictionaryBuilder:

    def __init__(self,
                 processes: int = 1,
                 pos_tag_label: str = CoreNLPServerPack._GlobalArgs.DEFAULT_POS_TAG_LABEL,
                 ner_tag_label: str = CoreNLPServerPack._GlobalArgs.DEFAULT_NER_TAG_LABEL,
                 sentiment_tag_label: str = CoreNLPServerPack._GlobalArgs.DEFAULT_SENTIMENT_TAG_LABEL,
                 compounding_sep_string: str = CoreNLPServerPack._GlobalArgs.DEFAULT_COMPOUNDING_SEP_STRING,
                 token_sep_string: str = CoreNLPServerPack._GlobalArgs.DEFAULT_TOKEN_SEP_STRING,
                 lower_case: bool = True,
                 full_token_compose_restriction: typing.Optional[str] = 'contains_alphabet_only',
                 token_remove_ner_tags_to_lessequal_then_num: typing.Optional[int] = 1,
                 remove_stopwords_set: typing.Optional[set] = resources.SET_STOPWORDS,
                 remove_punctuations_set: typing.Optional[set] = None,
                 remove_token_lessequal_then_length: typing.Optional[int] = 1,
                 remove_ner_options_dict: typing.Optional[dict] = {'removes_tags': 'all'},
                 ):
        """
        This is the Ngram-Dictionary of Hassan et.al(2019),
        it required a cleaned texture of AnnotationT.LineAnnotatorParallel/AnnotationT.LineAnnotator
        :param processes: use how many processes to clean the document, default be 1
        :param pos_tag_label: str, the pos tag label
        :param ner_tag_label: str, the ner tag label
        :param sentiment_tag_label: str, the sentiment tag label of sentence
        :param compounding_sep_string: str, the compounding sep string
        :param token_sep_string: str, the token sep string
        :param lower_case: bool, if True then make the text lower case
        :param full_token_compose_restriction: str or None, the restriction of full token composition, if None then do nothing
            The full token means the mwe/compounding/phrase, the restriction is:
            'contains_alphabet_only': the full token must contains alphabet only
            'contains_alphabet_and_number_only': the full token must contains alphabet and number only
            'contains_number_only': the full token must contains number only
            'contains_alphabet': the full token must contains alphabet
            'contains_number': the full token must contains number
            'contains_alphabet_and_number': the full token must contains alphabet and number
            ELSE, the full token will be ignore and passed.
        :param token_remove_ner_tags_to_lessequal_then_num: int or None, the restriction of full token NER tags number, if None then do nothing
            This option if for situation when several NERs are in one token: [ner:duration]_[ner:duration]
            Actually the first one delegate the meaning so we can remove the second one. Vice versa.
            <It is recommend to use 1>
        :param remove_stopwords_set: set or None, the stopwords set to be removed
        :param remove_punctuations_set: set or None, the punctuations set to be removed
        :param remove_token_lessequal_then_length: int or None, the tokens which length is less equal than this length will be removed,
            the token total length included all sub-words inside a NER phrases
        :param remove_ner_options_dict: dict or None, the options of remove ner, if None then do nothing
            The keys of dict must be in ['removes_tags', 'removes_original_text', 'removes_tags_and_original_text']
            The values of dict SHOULD be in CoreNLPServerPack._GlobalArgs.STANFORD_CORENLP_NER_TAGS_UPPER_SET or CoreNLPServerPack._GlobalArgs.ALL_TAGS_FLAG
            The usage example be: {'removes_tags': ['PERSON', 'LOCATION'], 'removes_original_text': 'all'}
        """
        # cleaning is after the line resolver
        self._pre_line_clearner_cls = CoreNLPServerPack.AnnotatedResolverT.AnnotatedLineCleaner(
            pos_tag_label=pos_tag_label,
            ner_tag_label=ner_tag_label,
            sentiment_tag_label=sentiment_tag_label,
            compounding_sep_string=compounding_sep_string,
            token_sep_string=token_sep_string,
            lower_case=lower_case,
            # HERE: the prestep's restruct should be same as before, consistency
            restruct_compounding_sep_string=compounding_sep_string,
            # HERE: the prestep's restruct should be same as before, consistency
            restruct_token_sep_string=token_sep_string,
            full_token_compose_restriction=full_token_compose_restriction,
            token_remove_ner_tags_to_lessequal_then_num=token_remove_ner_tags_to_lessequal_then_num,
            remove_stopwords_set=remove_stopwords_set,
            remove_punctuations_set=remove_punctuations_set,
            remove_token_lessequal_then_length=remove_token_lessequal_then_length,
            remove_ner_options_dict=remove_ner_options_dict,
            remove_pos_options_dict=None,  # It must be None for it could not be removed
            clean_flag=CoreNLPServerPack._GlobalArgs.FLAG_ANNOTATED_LINE_CLEANER_CLEANEDLINE
        )

        self._after_line_resolver_cls = CoreNLPServerPack.AnnotatedResolverT._AnnotatedLineResolver(
            pos_tag_label=pos_tag_label,
            ner_tag_label=ner_tag_label,
            sentiment_tag_label=sentiment_tag_label,
            compounding_sep_string=compounding_sep_string,
            token_sep_string=token_sep_string,
            lower_case=lower_case
        )

        self.token_sep_string = token_sep_string

        self.processes = processes

    def _clean_dict_training_texture_list(self,
                                          texture_list,
                                          processes: int
                                          ) -> typing.List[str]:
        """
        The pre-work cleaner to clean texture list to texture that available for Hassan et.al(2019) and DictionaryT
        It do not have a final cleaner for it actually is not recommend to use outside for it is a limited cleaner.
        :param texture_list: the list of texture, typing.List[str]
        :param processes: numbers of processes, int
        Returns: list of cleaned texture
        """

        assert isinstance(texture_list, list), 'the texture_list must be list!'

        def _sent_cleaner(s):
            return self._pre_line_clearner_cls.line_annotated_tokens_cleaner(
                re.sub(r'\s+',
                       self.token_sep_string,
                       s,
                       flags=re.IGNORECASE | re.DOTALL)
            )

        if processes <= 1:
            return [
                _sent_cleaner(s)
                for s in tqdm.tqdm(texture_list)
            ]

        else:

            _result_sent_l = []

            with pathos.multiprocessing.Pool(
                    initializer=_BasicKits._BasicFuncT.processes_interrupt_initiator,
                    processes=processes
            ) as pool:
                for rsts in tqdm.tqdm(
                        pool.imap(_sent_cleaner, texture_list),
                        total=len(texture_list)
                ):
                    _result_sent_l.append(rsts)

            return _result_sent_l

    def n_gramizer_dictionary_builder(
            self,
            corenlp_annotated_texture: typing.Union[str, typing.List[str]],
            n: int = 2,
            scorer: typing.Literal['tf', 'count'] = 'tf',
            remove_ngram_postags_combinations: typing.Optional[
                list] = _Resources.library_remove_postags_bigram_combinations,
            remove_ngram_contain_any_words_list: typing.Optional[list] = _Resources.library_remove_single_words_lower,
            final_remove_token_lessequal_then_length: typing.Optional[int] = None,
            texture_split_by: typing.Literal['sentence', 'all'] = 'all'
    ):
        """
        Args:
            corenlp_annotated_texture: The annotated texture,
                it could be a list of sentences(documents) or a string of document
            n: numbers of n-gram, bigrams default
            scorer: tf or count, the score of n-gram. tf mean weighted by total number of n-grams
            remove_ngram_postags_combinations: the postags combinations to be removed, like ('PRP', 'PRP')
            remove_ngram_contain_any_words_list: the words to be removed, like ['i', 'ive', 'youve']
            final_remove_token_lessequal_then_length: the final remove token length less equal than this length
            texture_split_by: split the texture by sentence or all to count the n-gram scores

        Returns:

        """
        def _lambda_texture_bigramdict_builder(texture, remove_ngram_postags_comb_lower,
                                               remove_ngram_contain_anywordsl_lower):
            if not isinstance(texture, (str, list)):
                raise ValueError('Input texture must be a string or list of sentences.')
            if not texture:
                return 0, {}

            if isinstance(texture, list):
                texture = self.token_sep_string.join(texture)

            texture = self.token_sep_string.join(
                self._clean_dict_training_texture_list([texture], processes=self.processes)
            )

            texture_list, postags_list = [], []
            for _d in self._after_line_resolver_cls.line_resolver(texture)['resolved_tokens']:
                for _subd in _d.values():
                    texture_list.append(_subd['original_text'])
                    postags_list.append(_subd['pos'].lower())

            if final_remove_token_lessequal_then_length:
                try:
                    texture_list, postags_list = zip(*[
                        (t, p) for t, p in zip(texture_list, postags_list)
                        if len(t) > final_remove_token_lessequal_then_length
                    ])
                except Exception:
                    return 0, {}

            ngram_texture_list = list(nltk.ngrams(texture_list, n=n))
            ngram_postags_list = list(nltk.ngrams(postags_list, n=n))

            if remove_ngram_postags_comb_lower or remove_ngram_contain_anywordsl_lower:
                ngram_texture_list = [
                    ngram_texture_list[i]
                    for i in range(len(ngram_postags_list))
                    if ngram_postags_list[i] not in remove_ngram_postags_comb_lower and
                       not set(w.lower() for w in ngram_texture_list[i]).intersection(
                           remove_ngram_contain_anywordsl_lower)
                ]

            tl = len(ngram_texture_list)
            if scorer == 'count':
                bg_score_dict = dict(Counter(ngram_texture_list))
            elif scorer == 'tf':
                bg_score_dict = {w: count / tl for w, count in Counter(ngram_texture_list).items()}
            else:
                raise ValueError(f"Invalid scorer '{scorer}'. Must be either 'tf' or 'count'.")

            return tl, bg_score_dict

        assert scorer in ['tf', 'count'], 'Scorer must be "tf" or "count".'

        remove_ngram_postags_combinations_lower = [
            tuple(s.lower() for s in tp) for tp in (remove_ngram_postags_combinations or [])
        ]
        remove_ngram_contain_any_words_list_lower = [
            s.lower() for s in (remove_ngram_contain_any_words_list or [])
        ]

        print('Building bigram dictionary...')

        if texture_split_by == 'all':
            texture_len, bigram_score_dict = _lambda_texture_bigramdict_builder(
                corenlp_annotated_texture,
                remove_ngram_postags_combinations_lower,
                remove_ngram_contain_any_words_list_lower
            )
            return bigram_score_dict

        elif texture_split_by == 'sentence':
            if not isinstance(corenlp_annotated_texture, list):
                raise ValueError('Input must be a list of sentences if texture_split_by="sentence".')

            combined_dict = Counter()
            sentence_lengths = []

            for _sentence in tqdm.tqdm(corenlp_annotated_texture):
                tl, bg_dict = _lambda_texture_bigramdict_builder(
                    _sentence,
                    remove_ngram_postags_combinations_lower,
                    remove_ngram_contain_any_words_list_lower
                )
                combined_dict.update(bg_dict)
                sentence_lengths.append(tl)

            if scorer == 'tf':
                total_len = sum(sentence_lengths)
                if total_len == 0:
                    return {}
                combined_dict = {k: v / total_len for k, v in combined_dict.items()}

            return dict(combined_dict)

        else:
            raise ValueError(f'Invalid texture_split_by value "{texture_split_by}". Must be "all" or "sentence".')
