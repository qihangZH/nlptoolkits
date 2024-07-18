import re
import typing
import warnings
import nltk
import pathos
import tqdm
from .. import _BasicKits
from ..StanzaKits import CoreNLPServerPack
from .. import resources


# def cleaner_remove_token_lessequal_then_length(
#         texture: str,
#         token_sep_string: str = CoreNLPServerPack._GlobalArgs.DEFAULT_TOKEN_SEP_STRING,
#         remove_token_lessequal_then_length: typing.Optional[int] = 1
# ):
#     if remove_token_lessequal_then_length is not None:
#
#         token_list = texture.split(token_sep_string)
#
#         rst_l = []
#         for s in token_list:
#             if len(s) > remove_token_lessequal_then_length:
#                 rst_l.append(s)
#         return token_sep_string.join(
#             rst_l
#         )
#     else:
#         return texture


def naive_nltk_annotator(
        document_list,
        restruct_token_sep_string: str = CoreNLPServerPack._GlobalArgs.DEFAULT_TOKEN_SEP_STRING,
):
    warnings.warn("This function could only support the basic annotate needs. "
                  "It could be not reliable for complicate tasks, "
                  "I recommend to use AnnotationT.LineAnnotatorParallel/AnnotationT.LineAnnotator instead",
                  DeprecationWarning)

    return [
        restruct_token_sep_string.join(nltk.tokenize.word_tokenize(s))
        for s in document_list
    ]


class NgramDataPreprocessor:

    def __init__(self,
                 pos_tag_label: str = CoreNLPServerPack._GlobalArgs.DEFAULT_POS_TAG_LABEL,
                 ner_tag_label: str = CoreNLPServerPack._GlobalArgs.DEFAULT_NER_TAG_LABEL,
                 sentiment_tag_label: str = CoreNLPServerPack._GlobalArgs.DEFAULT_SENTIMENT_TAG_LABEL,
                 compounding_sep_string: str = CoreNLPServerPack._GlobalArgs.DEFAULT_COMPOUNDING_SEP_STRING,
                 token_sep_string: str = CoreNLPServerPack._GlobalArgs.DEFAULT_TOKEN_SEP_STRING,
                 lower_case: bool = True,
                 restruct_compounding_sep_string: str = " ",  # In restruct, all should be sep by " "
                 restruct_token_sep_string: str = " ",
                 full_token_compose_restriction: typing.Optional[str] = 'contains_alphabet_only',
                 token_remove_ner_tags_to_lessequal_then_num: typing.Optional[int] = 1,
                 remove_stopwords_set: typing.Optional[set] = resources.SET_STOPWORDS,
                 remove_punctuations_set: typing.Optional[set] = None,
                 remove_token_lessequal_then_length: typing.Optional[int] = 1,
                 remove_ner_options_dict: typing.Optional[dict] = {'removes_tags': 'all'},
                 remove_pos_options_dict: typing.Optional[dict] = {'removes_tags': 'all'},
                 clean_flag: int = CoreNLPServerPack._GlobalArgs.FLAG_ANNOTATED_LINE_CLEANER_CLEANEDLINE
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
        :param restruct_compounding_sep_string: str,
            the compounding sep string in restruct the tokens back after decompose.
            IT IS NOT INFLUENCED BY THE LOWER CASE!
        :param restruct_token_sep_string: str, the token sep string in restruct the tokens back after decompose.
            IT IS NOT INFLUENCED BY THE LOWER CASE!
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
        :param remove_token_lessequal_then_length: int or None, the tokens which length is less equal than this length will be removed
        :param remove_ner_options_dict: dict or None, the options of remove ner, if None then do nothing
            The keys of dict must be in ['removes_tags', 'removes_original_text', 'removes_tags_and_original_text']
            The values of dict SHOULD be in CoreNLPServerPack._GlobalArgs.STANFORD_CORENLP_NER_TAGS_UPPER_SET or CoreNLPServerPack._GlobalArgs.ALL_TAGS_FLAG
            The usage example be: {'removes_tags': ['PERSON', 'LOCATION'], 'removes_original_text': 'all'}
        :param remove_pos_options_dict: dict or None, the options of remove pos, if None then do nothing
            The keys of dict must be in ['removes_tags', 'removes_original_text', 'removes_tags_and_original_text']
            The values of dict SHOULD be in CoreNLPServerPack._GlobalArgs.POS_PENN_TREE_BANK_TAGS_UPPER_SET or CoreNLPServerPack._GlobalArgs.ALL_TAGS_FLAG
            The usage example be: {'removes_tags': ['NN', 'NNS'], 'removes_original_text': 'all'}
        :param clean_flag: int, the flag of clean->
            0 means only return cleaned line, or CoreNLPServerPack._GlobalArgs.ANNOTATED_LINE_CLEANER_CLEANEDLINE
            1 means only return sentiment or CoreNLPServerPack._GlobalArgs.ANNOTATED_LINE_CLEANER_SENTIMENT
            To be more specific, see StanzaKits/CoreNLPServerPack/CoreNLPServerPack._GlobalArgs.py For detail.
        """
        self.line_clearner_cls = CoreNLPServerPack.AnnotatedResolverT.AnnotatedLineCleaner(
            pos_tag_label=pos_tag_label,
            ner_tag_label=ner_tag_label,
            sentiment_tag_label=sentiment_tag_label,
            compounding_sep_string=compounding_sep_string,
            token_sep_string=token_sep_string,
            lower_case=lower_case,
            restruct_compounding_sep_string=restruct_compounding_sep_string,
            restruct_token_sep_string=restruct_token_sep_string,
            full_token_compose_restriction=full_token_compose_restriction,
            token_remove_ner_tags_to_lessequal_then_num=token_remove_ner_tags_to_lessequal_then_num,
            remove_stopwords_set=remove_stopwords_set,
            remove_punctuations_set=remove_punctuations_set,
            remove_token_lessequal_then_length=remove_token_lessequal_then_length,
            remove_ner_options_dict=remove_ner_options_dict,
            remove_pos_options_dict=remove_pos_options_dict,
            clean_flag=clean_flag
        )

        self.restruct_compounding_sep_string = restruct_compounding_sep_string
        self.restruct_token_sep_string = restruct_token_sep_string

        self.corenlp_cleaner_dict = {
            'pos_tag_label': pos_tag_label,
            'ner_tag_label': ner_tag_label,
            'sentiment_tag_label': sentiment_tag_label,
            'compounding_sep_string': compounding_sep_string,
            'token_sep_string': token_sep_string,
            'lower_case': lower_case,
            'restruct_compounding_sep_string': restruct_compounding_sep_string,
            'restruct_token_sep_string': restruct_token_sep_string,
            'full_token_compose_restriction': full_token_compose_restriction,
            'token_remove_ner_tags_to_lessequal_then_num': token_remove_ner_tags_to_lessequal_then_num,
            'remove_stopwords_set': remove_stopwords_set,
            'remove_punctuations_set': remove_punctuations_set,
            'remove_token_lessequal_then_length': remove_token_lessequal_then_length,
            'remove_ner_options_dict': remove_ner_options_dict,
            'remove_pos_options_dict': remove_pos_options_dict,
            'clean_flag': clean_flag
        }

        if (
                (self.restruct_compounding_sep_string != ' ') or
                (self.restruct_token_sep_string != ' ')
        ):
            warnings.warn(
                "The seperation token SHOULD be ' ' whatever, it may cause un-expected results!",
                UserWarning)

    """read the list of sentences"""

    def clean_annotated_texture_list(self,
                                     texture_list,
                                     processes: int,
                                     ) -> typing.List[str]:

        assert isinstance(texture_list, list), 'the texture_list must be list!'

        def _sent_cleaner(s):
            return self.line_clearner_cls.line_annotated_tokens_cleaner(
                re.sub(r'\s+',
                       self.restruct_token_sep_string,
                       s,
                       flags=re.IGNORECASE | re.DOTALL)
            )

        print('Cleaning the annotated data...')

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

    def alias_auto_clean_annotated_txt(self,
                                       path_in_parsed_txt,
                                       path_out_cleaned_txt,
                                       processes: int,
                                       chunk_size=200000,
                                       start_iloc=None
                                       ) -> None:
        """
        Use this alias to quickly replicate the same data of the class
        by other CoreNLP annotated data instead write new code and cause possible errors
        """
        CoreNLPServerPack.auto_clean_annotated_txt(
            path_in_parsed_txt=path_in_parsed_txt,
            path_out_cleaned_txt=path_out_cleaned_txt,
            processes=processes,
            chunk_size=chunk_size,
            start_iloc=start_iloc,
            **self.corenlp_cleaner_dict
        )
