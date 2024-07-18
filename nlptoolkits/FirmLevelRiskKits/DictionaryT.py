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
            scorer: typing.Literal['tf', 'tfidf'] = 'tfidf',
            remove_ngram_postags_combinations: typing.Optional[list] = _Resources.library_remove_bigram_combinations,
            remove_ngram_contain_any_words_list: typing.Optional[list] = _Resources.library_remove_single_words_lower,
            final_remove_token_lessequal_then_length: typing.Optional[int] = None
    ):
        """
        :param corenlp_annotated_texture: a list of sentences to be processed, every one will be auto cleaned by arguments initialized
        :param n: the N-grams' N, for example, if 2, then it will be bi-gram, 3 will be tri-gram
        :param scorer: the scorer to give a score of each N-gram, could be 'tf' or 'tfidf'
        :param remove_ngram_postags_combinations: the combination of N-grams pos-tags to remove from the dictionary if match.
            The default,
        :param remove_ngram_contain_any_words_list: remvoe the N-gram pairs if one of them contains any of the words
        """

        assert scorer in ['tf', 'tfidf'], 'scorer are not in selectable choices!'

        if n != 2 and remove_ngram_postags_combinations == _Resources.library_remove_bigram_combinations:
            warnings.warn('Default Bigram remove postag combination maybe meaningless if N!=2', UserWarning)

        # we always lower the tags
        remove_ngram_postags_combinations_lower = [
            tuple([s.lower() for s in tp])
            for tp in remove_ngram_postags_combinations
        ] if remove_ngram_postags_combinations else []

        remove_ngram_contain_any_words_list_lower = [
            s.lower()
            for s in remove_ngram_contain_any_words_list
        ] if remove_ngram_contain_any_words_list else []

        if isinstance(corenlp_annotated_texture, list):
            texture = self.token_sep_string.join(
                corenlp_annotated_texture)

        elif isinstance(corenlp_annotated_texture, str):
            texture = corenlp_annotated_texture
        else:
            raise ValueError(
                'texture should be either string or list of sentences(The sentences itself will be concat)'
            )

        # use the cleaner to clean the texture(always keep POS tags, if have)
        texture = self.token_sep_string.join(
            self._clean_dict_training_texture_list([texture], processes=self.processes)
        )

        texture_list = []
        postags_list = []

        # extract both texture and pos tags
        for _d in self._after_line_resolver_cls.line_resolver(
                texture
        )['resolved_tokens']:

            for _subd in _d.values():
                texture_list.append(
                    _subd['original_text']
                )

                """It should be mention that the pos are always set to lower for better checkin"""
                postags_list.append(
                    _subd['pos'].lower()
                )

        """Add on: remove all texture that original text is less than the min length of final restriction"""
        _removed_texture_list = []
        _removed_postags_list = []
        if final_remove_token_lessequal_then_length is not None:
            assert len(texture_list) == len(postags_list)
            for i in range(len(texture_list)):
                if len(texture_list[i]) > final_remove_token_lessequal_then_length:
                    _removed_texture_list.append(texture_list[i])
                    _removed_postags_list.append(postags_list[i])

            texture_list = _removed_texture_list
            postags_list = _removed_postags_list


        ngram_texture_list = list(nltk.ngrams(texture_list, n=n))
        ngram_postags_list = list(nltk.ngrams(postags_list, n=n))

        if remove_ngram_postags_combinations_lower or remove_ngram_contain_any_words_list_lower:

            _cleaned_ngram_texture_list = []

            assert len(ngram_texture_list) == len(ngram_postags_list)
            # remove all combinations that postags match the removal:
            for i in range(len(ngram_postags_list)):
                if (ngram_postags_list[i] in remove_ngram_postags_combinations_lower) or \
                        (
                                # if any of N-grams same as remove words list(go lower)
                                not
                                set(
                                    [w.lower() for w in list(ngram_texture_list[i])]
                                ).isdisjoint(
                                    remove_ngram_contain_any_words_list_lower
                                )
                        ):
                    pass
                else:
                    _cleaned_ngram_texture_list.append(ngram_texture_list[i])

            ngram_texture_list = _cleaned_ngram_texture_list

        if scorer == 'tf':
            bigram_score_dict = {
                w: ngram_texture_list.count(w)
                for w in tqdm.tqdm(set(ngram_texture_list))
            }
        elif scorer == 'tfidf':
            bigram_score_dict = {
                w: ngram_texture_list.count(w) / len(ngram_texture_list)
                for w in tqdm.tqdm(set(ngram_texture_list))
            }
        else:
            raise

        return bigram_score_dict
