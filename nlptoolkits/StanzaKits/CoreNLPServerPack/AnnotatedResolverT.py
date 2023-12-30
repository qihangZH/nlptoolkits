import functools
import re
import typing
import numpy as np
import warnings

from . import _GlobalArgs
# import _GlobalArgs


class _AnnotatedLineResolver:
    def __init__(self,
                 pos_tag_label: str,
                 ner_tag_label: str,
                 sentiment_tag_label: str,
                 compounding_sep_string: str,
                 token_sep_string: str,
                 lower_case: bool,
                 ):
        """
        Resolve the doc annotated by CoreNLP (preprocessor) in AnnotationT.py
        :param pos_tag_label: str, the pos tag label
        :param ner_tag_label: str, the ner tag label
        :param sentiment_tag_label: str, the sentiment tag label of sentence
        :param compounding_sep_string: str, the compounding sep string
        :param token_sep_string: str, the token sep string
        :param lower_case: bool, if True then make the text lower case
        """
        """WARNING: IF LOWER CASE SET, then the tokenize_sep and compounding_sep should be lower case too"""
        # Besides, the lowercase always run at first
        self.lower_case = lower_case

        self.pos_tag_label = pos_tag_label.lower() if self.lower_case else pos_tag_label
        self.ner_tag_label = ner_tag_label.lower() if self.lower_case else ner_tag_label

        self.sentiment_tag_label = sentiment_tag_label.lower() if self.lower_case else sentiment_tag_label

        self.token_sep_string = token_sep_string.lower() if self.lower_case else token_sep_string
        self.compounding_sep_string = compounding_sep_string.lower() if self.lower_case else compounding_sep_string

    # --------------------------------------------------------------------------------------------------
    # line cleaner
    # --------------------------------------------------------------------------------------------------
    def tokenize(self, line):
        """Tokenize the line by the tokenize_sep"""
        return line.split(self.token_sep_string)

    @staticmethod
    def line_lower_case(line):
        """Make the line lower case
        :param line: text processed_data by the preprocessor
        """
        return line.lower()

    # --------------------------------------------------------------------------------------------------
    # token cleaner(in line)
    # --------------------------------------------------------------------------------------------------

    def token_resolver(self, token) -> dict:
        """
        split the token to NER, POS and token by sep, and give each part a index.
        Then we have to make the fragments into a dict, which in sequence. \
        Each part should have such things:NER, POS, original_text, index.
        token_fragment and index is something we must have, while NER and POS is optional(based on parsed results).
        """
        fragments_list = token.split(self.compounding_sep_string)

        splitted_dict = dict()

        for i, fragment in enumerate(fragments_list):
            splitted_dict[i] = dict()

            # first find NER tags(in the start of the fragment), we also have to escape the special characters
            ner_match = re.search(rf"^\[{re.escape(self.ner_tag_label)}:(.+?)\]", fragment, flags=re.IGNORECASE)
            if ner_match:
                splitted_dict[i]['ner'] = ner_match.group(1)
                # get the end and start of match
                start, end = ner_match.span()
                # remove the searched NER tags
                fragment = fragment[:start] + fragment[end:]
            else:
                splitted_dict[i]['ner'] = ''

            # second find Non-POS tags(in the end of the fragment), if match nothing then means there are no pos tags
            # This tactic is use for non-greedy match, so that we can get the last match non-greedy
            pos_match = re.search(rf"(^.*)(?=\[{re.escape(self.pos_tag_label)}:(.+?)\]$)",
                                  fragment,
                                  flags=re.IGNORECASE
                                  )
            if pos_match:
                splitted_dict[i]['pos'] = pos_match.group(2)
                # if pos match exist then means there are pos tags
                start, end = pos_match.span()
                # remove the searched POS tags, actually we searched for the center part
                fragment = fragment[:end]
            else:
                splitted_dict[i]['pos'] = ''

            """
            TODO: IF ANY OTHER TAGS ADD ON, CONSIDER THE NER AND POS TAGS
            """
            splitted_dict[i]['original_text'] = fragment

        return splitted_dict

    def sentiment_resolver(self, token):
        sentiment_match = re.search(rf"^\[{re.escape(self.sentiment_tag_label)}:(.+?)\]$", token, flags=re.IGNORECASE)
        if sentiment_match:
            return sentiment_match.group(1)
        else:
            return None

    def line_resolver(self, line) -> dict:
        """
        Resolve the line annotated by CoreNLP (preprocessor) in AnnotationT.py
        :param line: text processed_data by the preprocessor
        :return: a list of dict, each dict is a token resolved by self.token_resolver
        """
        # flags of already find sentiment
        FLAG_SENTIMENT_FIND = False
        # container of resolved tokens
        line_resolved_dict = dict()
        line_resolved_dict['resolved_tokens'] = []
        # ----------------------------------------------
        # Lower case
        # ----------------------------------------------

        line = line.strip()

        if self.lower_case:
            line = self.line_lower_case(line)

        for t in self.tokenize(line):
            # ----------------------------------------------
            # detect and remove the token if it is sentiment
            # ----------------------------------------------
            if not FLAG_SENTIMENT_FIND:
                possible_sentiment = self.sentiment_resolver(t)
                if possible_sentiment:
                    line_resolved_dict['sentiment'] = possible_sentiment
                    # Change the flag to True
                    FLAG_SENTIMENT_FIND = True
                    # go to next loop
                    continue

            # ----------------------------------------------
            # resolve the token
            # ----------------------------------------------
            line_resolved_dict['resolved_tokens'].append(self.token_resolver(t))

        return line_resolved_dict


class AnnotatedLineCleaner(_AnnotatedLineResolver):

    def __init__(self,
                 pos_tag_label: str,
                 ner_tag_label: str,
                 sentiment_tag_label: str,
                 compounding_sep_string: str,
                 token_sep_string: str,
                 lower_case: bool,
                 restruct_compounding_sep_string: str,
                 restruct_token_sep_string: str,
                 full_token_compose_restriction: typing.Optional[str],
                 token_remove_ner_tags_to_lessequal_then_num: typing.Optional[int],
                 remove_stopwords_set: typing.Optional[set],
                 remove_punctuations_set: typing.Optional[set],
                 remove_token_lessequal_then_length: typing.Optional[int],
                 remove_ner_options_dict: typing.Optional[dict],
                 remove_pos_options_dict: typing.Optional[dict],
                 clean_flag: int,
                 ):
        """
        Clean the doc annotated by CoreNLP (preprocessor) in AnnotationT.py
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
            The keys of dict must be in ['remove_tags', 'removes_original_text', 'remove_tags_and_original_text']
            The values of dict SHOULD be in _GlobalArgs.STANFORD_CORENLP_NER_TAGS_UPPER_SET or _GlobalArgs.ALL_TAGS_FLAG
            The usage example be: {'remove_tags': ['PERSON', 'LOCATION'], 'removes_original_text': 'all'}
        :param remove_pos_options_dict: dict or None, the options of remove pos, if None then do nothing
            The keys of dict must be in ['remove_tags', 'removes_original_text', 'remove_tags_and_original_text']
            The values of dict SHOULD be in _GlobalArgs.POS_PENN_TREE_BANK_TAGS_UPPER_SET or _GlobalArgs.ALL_TAGS_FLAG
            The usage example be: {'remove_tags': ['NN', 'NNS'], 'removes_original_text': 'all'}
        :param clean_flag: int, the flag of clean->
            0 means only return cleaned line, or _GlobalArgs.ANNOTATED_LINE_CLEANER_CLEANEDLINE
            1 means only return sentiment or _GlobalArgs.ANNOTATED_LINE_CLEANER_SENTIMENT
            To be more specific, see StanzaKits/CoreNLPServerPack/_GlobalArgs.py For detail.
        """
        super().__init__(
            pos_tag_label=pos_tag_label,
            ner_tag_label=ner_tag_label,
            sentiment_tag_label=sentiment_tag_label,
            compounding_sep_string=compounding_sep_string,
            token_sep_string=token_sep_string,
            lower_case=lower_case
        )

        self.restruct_compounding_sep_string = restruct_compounding_sep_string
        self.restruct_token_sep_string = restruct_token_sep_string

        self.annotated_tokens_fields_arr = np.array(['ner', 'pos', 'original_text'])

        self.annotated_tokens_fields_iloc_dict = {
            'ner': np.where(self.annotated_tokens_fields_arr == 'ner')[0][0],
            'pos': np.where(self.annotated_tokens_fields_arr == 'pos')[0][0],
            'original_text': np.where(self.annotated_tokens_fields_arr == 'original_text')[0][0]
        }

        # remove all if detect this word in the option dict
        self._tag_remove_conserve_word = _GlobalArgs.ALL_TAGS_FLAG

        self.full_token_constitute_option_choices_map = {
            'contains_alphabet_only': lambda x: x.isalpha(),
            'contains_alphabet_or_number_only': lambda x: x.isalnum(),
            'contains_number_only': lambda x: x.isnumeric(),
            'contains_alphabet': lambda x: any([c.isalpha() for c in x]),
            'contains_number': lambda x: any([c.isnumeric() for c in x]),
            'contains_alphabet_or_number': lambda x: any([c.isalnum() for c in x]),
        }

        self.remove_ner_map_arr_dict = {
            'remove_tags': ~(self.annotated_tokens_fields_arr == 'ner'),
            'removes_original_text': ~(self.annotated_tokens_fields_arr == 'original_text'),
            'remove_tags_and_original_text': ~np.logical_or(
                self.annotated_tokens_fields_arr == 'ner',
                self.annotated_tokens_fields_arr == 'original_text'
            )
        }

        self.remove_pos_map_arr_dict = {
            'remove_tags': ~(self.annotated_tokens_fields_arr == 'pos'),
            'removes_original_text': ~(self.annotated_tokens_fields_arr == 'original_text'),
            'remove_tags_and_original_text': ~np.logical_or(
                self.annotated_tokens_fields_arr == 'pos',
                self.annotated_tokens_fields_arr == 'original_text'
            )
        }

        # full_token_compose_restriction
        if isinstance(full_token_compose_restriction, str) and \
                full_token_compose_restriction in self.full_token_constitute_option_choices_map.keys():
            self.full_token_compose_restriction = full_token_compose_restriction
        elif not full_token_compose_restriction:
            self.full_token_compose_restriction = None
        else:
            raise ValueError(
                f'full_token_constitute_option must be str and in {self.full_token_constitute_option_choices_map.keys()}'
            )

        # token_ner_tags_num_restriction
        if isinstance(token_remove_ner_tags_to_lessequal_then_num, bool):
            raise ValueError('token_ner_tags_num_restriction must be int or None')
        elif isinstance(token_remove_ner_tags_to_lessequal_then_num, int):
            self.token_ner_tags_num_restriction = token_remove_ner_tags_to_lessequal_then_num
        elif not token_remove_ner_tags_to_lessequal_then_num:
            self.token_ner_tags_num_restriction = None
        else:
            raise ValueError('token_ner_tags_num_restriction must be int or None')

        # remove_stopwords_set
        if isinstance(remove_stopwords_set, set):
            self.remove_stopwords_set = remove_stopwords_set
        elif not remove_stopwords_set:
            self.remove_stopwords_set = None
        else:
            raise ValueError('remove_stopwords_set must be set or None')

        # remove_punctuations_set
        if isinstance(remove_punctuations_set, set):
            self.remove_punctuations_set = remove_punctuations_set
        elif not remove_punctuations_set:
            self.remove_punctuations_set = None
        else:
            raise ValueError('remove_punctuations_set must be set or None')

        # remove_token_lessequal_then_length
        if isinstance(remove_token_lessequal_then_length, bool):
            raise ValueError('remove_token_lessequal_then_length must be int or None')
        if isinstance(remove_token_lessequal_then_length, int):
            self.remove_token_lessequal_then_length = remove_token_lessequal_then_length
        elif not remove_token_lessequal_then_length:
            self.remove_token_lessequal_then_length = None
        else:
            raise ValueError('remove_token_lessequal_then_length must be int or None')

        # remove_ner_options_dict
        if isinstance(remove_ner_options_dict, dict):
            self.remove_ner_options_dict = remove_ner_options_dict

            for k, v in self.remove_ner_options_dict.items():
                if k not in self.remove_ner_map_arr_dict.keys():
                    raise ValueError(f'remove_ner_options_dict keys must be in {self.remove_ner_map_arr_dict.keys()}')

                # Try to iter, if not iterable then raise error
                iter(v)

                if isinstance(v, str):
                    if v != self._tag_remove_conserve_word:
                        raise ValueError(f'the values of remove_ner_options_dict must be '
                                         f'{self._tag_remove_conserve_word} or ArrayLike')
                    else:
                        continue

                if not set([i.upper() for i in v]).issubset(_GlobalArgs.STANFORD_CORENLP_NER_TAGS_UPPER_SET):
                    warnings.warn(f'remove_ner_options_dict values SHOULD be in '
                                  f'{_GlobalArgs.STANFORD_CORENLP_NER_TAGS_UPPER_SET}, '
                                  f'going on if you know what you are doing.')

                if len(set([i.upper() for i in v]).intersection(_GlobalArgs.STANFORD_CORENLP_NER_TAGS_UPPER_SET)) != \
                        len(set([i for i in v]).intersection(_GlobalArgs.STANFORD_CORENLP_NER_TAGS_UPPER_SET)):
                    warnings.warn(f'Some of remove_ner_options_dict values ARE IN'
                                  f'{_GlobalArgs.STANFORD_CORENLP_NER_TAGS_UPPER_SET}, '
                                  f'BUT SOME ARE NOT IN SAME CASE AS it(may lower or camel, snake case, etc.'
                                  f'Be aware, This class always run in CASE INSENSITIVE MODE,'
                                  f'going on if you know what you are doing.')

        elif not remove_ner_options_dict:
            self.remove_ner_options_dict = None
        else:
            raise ValueError('remove_ner_options_dict must be dict or None')

        # remove_pos_options_dict
        if isinstance(remove_pos_options_dict, dict):
            self.remove_pos_options_dict = remove_pos_options_dict

            for k, v in self.remove_pos_options_dict.items():
                if k not in self.remove_pos_map_arr_dict.keys():
                    raise ValueError(f'remove_pos_options_dict keys must be in {self.remove_pos_map_arr_dict.keys()}')

                # Try to iter, if not iterable then raise error
                iter(v)

                if isinstance(v, str):
                    if v != self._tag_remove_conserve_word:
                        raise ValueError(f'the values of remove_pos_options_dict must be '
                                         f'{self._tag_remove_conserve_word} or ArrayLike')
                    else:
                        continue

                if not set([i.upper() for i in v]).issubset(_GlobalArgs.POS_PENN_TREE_BANK_TAGS_UPPER_SET):
                    warnings.warn(f'remove_pos_options_dict values SHOULD be in '
                                  f'{_GlobalArgs.POS_PENN_TREE_BANK_TAGS_UPPER_SET}, '
                                  f'going on if you know what you are doing.')

                if len(set([i.upper() for i in v]).intersection(_GlobalArgs.POS_PENN_TREE_BANK_TAGS_UPPER_SET)) != \
                        len(set([i for i in v]).intersection(_GlobalArgs.POS_PENN_TREE_BANK_TAGS_UPPER_SET)):
                    warnings.warn(f'Some of remove_pos_options_dict values ARE IN'
                                  f'{_GlobalArgs.POS_PENN_TREE_BANK_TAGS_UPPER_SET}, '
                                  f'BUT SOME ARE NOT IN SAME CASE AS it(may lower or camel, snake case, etc.'
                                  f'Be aware, This class always run in CASE INSENSITIVE MODE,'
                                  f'going on if you know what you are doing.')

        elif not remove_pos_options_dict:
            self.remove_pos_options_dict = None
        else:
            raise ValueError('remove_pos_options_dict must be dict or None')

        # clean_flag
        if isinstance(clean_flag, int) and \
                (clean_flag in _GlobalArgs.ANNOTATED_LINE_CLEANER_FLAGS.values()) and \
                (not isinstance(clean_flag, bool)):
            self.clean_flag = clean_flag
        else:
            raise ValueError(f'clean_flag must be {_GlobalArgs.ANNOTATED_LINE_CLEANER_FLAGS.values()}')

    # --------------------------------------------------------------------------------------------------
    # Token cleaner
    # --------------------------------------------------------------------------------------------------

    def _checker_zeros_matrix(self, token_splitted_dict):
        """
        Use a matrix to detect which token should be removed or keep
        give a pure matrix to do so, start with all False

        True mean keep, False mean remove
        """
        return np.zeros(
            (len(token_splitted_dict),
             len(self.annotated_tokens_fields_arr)
             ),
            dtype=bool)

    def _checker_ones_matrix(self, token_splitted_dict):
        """
        Use a matrix to detect which token should be removed or keep
        give a pure matrix to do so, start with all True

        True mean keep, False mean remove
        """
        return np.ones(
            (len(token_splitted_dict),
             len(self.annotated_tokens_fields_arr)
             ),
            dtype=bool)

    def full_token_compose_restriction_checker(self, token_splitted_dict):

        # start with an all False matrix
        _mask_matrix = self._checker_zeros_matrix(token_splitted_dict)

        # check the token restriction once the full token
        full_token = ''.join([token_splitted['original_text'] for token_splitted in token_splitted_dict.values()])

        # decide if to remove the full tokens or not
        if self.full_token_compose_restriction:

            _mask_matrix[:, :] = self.full_token_constitute_option_choices_map[
                self.full_token_compose_restriction
            ](full_token)

        else:
            _mask_matrix[:, :] = True

        return _mask_matrix

    def token_ner_tags_num_restriction_checker(self, token_splitted_dict):

        # start with an all True matrix
        _mask_matrix = self._checker_ones_matrix(token_splitted_dict)

        # decide if to remove the full tokens or not
        if self.token_ner_tags_num_restriction is not None:

            ner_count = 0

            for index in range(len(token_splitted_dict)):

                if ner_count >= self.token_ner_tags_num_restriction:
                    # Break if find the ner tags num is bigger than the restriction inside the token
                    _mask_matrix[index:, :] = self.remove_ner_map_arr_dict['remove_tags']
                    break

                token_splitted = token_splitted_dict[index]

                # decide if to remove the full tokens or not.
                # Count for the num of NER tags and make all following tags False
                if token_splitted['ner']:
                    ner_count += 1
                    continue

        else:
            _mask_matrix[:, :] = True

        return _mask_matrix

    def remove_punct_stopwords_checker(self, token_splitted_dict,
                                       removetype: typing.Literal['punct', 'stopwords']
                                       ):

        _mask_matrix = self._checker_zeros_matrix(token_splitted_dict)

        if removetype == 'punct':
            _remove_set = self.remove_punctuations_set
        elif removetype == 'stopwords':
            _remove_set = self.remove_stopwords_set
        else:
            raise ValueError(f'removetype must be punct or stopwords')

        # decide if to remove the full tokens or not
        if _remove_set and (len(token_splitted_dict) == 1):
            # if the full token is in the remove_stopwords_set then remove it
            _mask_matrix[:, :] = not (token_splitted_dict[0]['original_text'].upper() in
                                      [s.upper() for s in _remove_set]
                                      )
        else:
            _mask_matrix[:, :] = True

        return _mask_matrix

    def remove_token_lessequal_then_length_checker(self, token_splitted_dict):

        _mask_matrix = self._checker_zeros_matrix(token_splitted_dict)

        # decide if to remove the full tokens or not
        full_token = ''.join([token_splitted['original_text'] for token_splitted in token_splitted_dict.values()])

        """Integer consider 0 have to be not None! for not 0 = True"""
        if self.remove_token_lessequal_then_length is not None:
            _mask_matrix[:, :] = not (len(full_token) <= self.remove_token_lessequal_then_length)
        else:
            _mask_matrix[:, :] = True

        return _mask_matrix

    def remove_ner_checker(self, token_splitted_dict):
        """
        Use a matrix to detect which token should be removed or keep
        NER different from POS, for NER always remove all the tokens after it if it is a NER tag
        :param token_splitted_dict: the splitted dict of token by self.token_splitter
        """
        # -----------------------------------------
        # remove the token if it is full token and not meet the restriction
        # -----------------------------------------

        if not self.remove_ner_options_dict:
            return self._checker_ones_matrix(token_splitted_dict)

        else:

            _mask_m_list = list()

            # make a initial matrix of ones
            _mask_m_list.append(self._checker_ones_matrix(token_splitted_dict))

            for k, v in self.remove_ner_options_dict.items():
                _mask_matrix = self._checker_ones_matrix(token_splitted_dict)
                # from 0~max index to detect the token if be removed or not
                for index in range(len(token_splitted_dict)):
                    token_splitted = token_splitted_dict[index]

                    # if v is all
                    if v == self._tag_remove_conserve_word:
                        # IF it is an NER tag, not None or ''
                        if token_splitted['ner']:
                            _mask_matrix[index:, :] = self.remove_ner_map_arr_dict[k]

                            _mask_m_list.append(_mask_matrix)
                            continue

                    else:
                        # ignore case
                        if token_splitted['ner'].upper() in [i.upper() for i in v]:
                            _mask_matrix[index:, :] = self.remove_ner_map_arr_dict[k]

                            _mask_m_list.append(_mask_matrix)
                            continue

            _mask_matrix = functools.reduce(lambda x, y: x & y, _mask_m_list)

            return _mask_matrix

    def remove_pos_checker(self, token_splitted_dict):
        """
        Use a matrix to detect which token should be removed or keep
        POS only deal with the splitted part only, not all lines after it
        """
        # -----------------------------------------
        # remove the token if it is full token and not meet the restriction
        # -----------------------------------------

        if not self.remove_pos_options_dict:
            return self._checker_ones_matrix(token_splitted_dict)
        else:
            _mask_m_list = list()

            # make a initial matrix of ones
            _mask_m_list.append(self._checker_ones_matrix(token_splitted_dict))

            for k, v in self.remove_pos_options_dict.items():
                _mask_matrix = self._checker_ones_matrix(token_splitted_dict)
                # from 0~max index to detect the token if be removed or not
                for index in range(len(token_splitted_dict)):
                    token_splitted = token_splitted_dict[index]

                    # if v is all
                    if v == self._tag_remove_conserve_word:
                        # IF it is an POS tag, not None or ''
                        if token_splitted['pos']:
                            _mask_matrix[index, :] = self.remove_pos_map_arr_dict[k]

                    else:
                        # ignore case
                        if token_splitted['pos'].upper() in [i.upper() for i in v]:
                            _mask_matrix[index, :] = self.remove_pos_map_arr_dict[k]

                    _mask_m_list.append(_mask_matrix)

            _mask_matrix = functools.reduce(lambda x, y: x & y, _mask_m_list)

            return _mask_matrix

    def annotated_tokens_checker(self, token_splitted_dict):
        """
        Use a matrix to detect which token should be removed or keep
        :param token_splitted_dict: the splitted dict of token by self.token_splitter
        """
        checker_func_list = [
            self.full_token_compose_restriction_checker,
            self.token_ner_tags_num_restriction_checker,
            lambda l: self.remove_punct_stopwords_checker(l, removetype='punct'),
            lambda l: self.remove_punct_stopwords_checker(l, removetype='stopwords'),
            self.remove_token_lessequal_then_length_checker,
            self.remove_ner_checker,
            self.remove_pos_checker,
        ]

        _mask_m_list = [
            f(token_splitted_dict)
            for f in checker_func_list
        ]

        _mask_matrix = functools.reduce(lambda x, y: x & y, _mask_m_list)

        return _mask_matrix

    def annotated_cleaned_tokens_restruct(self, token_splitted_dict, mask_matrix,
                                          restruct_compounding_sep_string: str):
        """
        :param token_splitted_dict: the splitted dict of token by self.token_splitter
        :param mask_matrix: the mask matrix to detect which token should be removed or keep
        :param restruct_compounding_sep_string: the compounding sep string in restruct
        """

        restruct_token_splitted_list = list()
        for i in range(len(token_splitted_dict)):

            token_splitted = token_splitted_dict[i]

            for field in self.annotated_tokens_fields_arr:

                field_bool = mask_matrix[i, self.annotated_tokens_fields_iloc_dict[field]]

                if field_bool:
                    # when this field are not '' or None
                    if token_splitted[field]:

                        if field == 'pos':
                            token_splitted[field] = \
                                f'[POS:{token_splitted[field]}]'.lower() \
                                    if self.lower_case else f'[POS:{token_splitted[field]}]'
                        elif field == 'ner':
                            token_splitted[field] = \
                                f'[NER:{token_splitted[field]}]'.lower() \
                                    if self.lower_case else f'[NER:{token_splitted[field]}]'
                else:
                    token_splitted[field] = ''

            restruct_token_splitted = f'{token_splitted["ner"]}' \
                                      f'{token_splitted["original_text"]}' \
                                      f'{token_splitted["pos"]}'

            restruct_token_splitted_list.append(
                restruct_token_splitted
            )

        return restruct_compounding_sep_string.join([
            s for s in restruct_token_splitted_list if s
        ])

    # --------------------------------------------------------------------------------------------------
    # line cleaner
    # --------------------------------------------------------------------------------------------------
    def line_annotated_tokens_cleaner(self,
                                      line
                                      ):
        """
        run when flags is _GlobalArgs.ANNOTATED_LINE_CLEANER_CLEANEDLINE
        :param line: text processed_data by the preprocessor
        :return: the cleaned line
        """

        resolved_line = self.line_resolver(line)

        # -----------------------------------------
        # loop using the method and use reduce to get the result
        # -----------------------------------------
        restructed_tokens_list = []

        for token_splitted_dict in resolved_line['resolved_tokens']:
            # -----------------------------------------
            # remove the token if it is full token and not meet the restriction
            # -----------------------------------------
            mask_matrix = self.annotated_tokens_checker(token_splitted_dict)

            # -----------------------------------------
            # restruct the tokens
            # -----------------------------------------
            restructed_tokens_list.append(
                self.annotated_cleaned_tokens_restruct(
                    token_splitted_dict,
                    mask_matrix,
                    self.restruct_compounding_sep_string
                )
            )

        return self.restruct_token_sep_string.join([
            s for s in restructed_tokens_list if s
        ])

    def line_sentiment(self, line):
        """
        run when flags is _GlobalArgs.ANNOTATED_LINE_CLEANER_SENTIMENT
        :param line: text processed_data by the preprocessor
        :return: the sentiment of the line
        """
        resolved_line = self.line_resolver(line)
        if 'sentiment' in resolved_line:
            return resolved_line['sentiment']
        else:
            return ''

    def clean(self, line, index):
        """
        main function that chains all filters together and applies to a string.
        :param line: text processed_data by the preprocessor
        :param index: the index of the line
        :return: the cleaned line
        """
        if self.clean_flag == _GlobalArgs.FLAG_ANNOTATED_LINE_CLEANER_CLEANEDLINE:
            return self.line_annotated_tokens_cleaner(line), index
        elif self.clean_flag == _GlobalArgs.FLAG_ANNOTATED_LINE_CLEANER_SENTIMENT:
            return self.line_sentiment(line), index
        else:
            raise ValueError(f'clean_flag must be {_GlobalArgs.ANNOTATED_LINE_CLEANER_FLAGS.values()}')

# class LineTextCleaner:
#     """Clean the text parsed by CoreNLP (preprocessor)
#     """
#
#     def __init__(self,
#                  stopwords_set: set,
#                  ner_keep_types_origin_list: typing.Optional[list] = None,
#                  token_minlength: typing.Optional[int] = 2,
#                  punctuations_set: set = set(["-lrb-", "-rrb-", "-lsb-", "-rsb-", "'s"]),
#                  is_remove_no_alphabet_contains: bool = True,
#                  ):
#         """
#         :param stopwords_set: stop word set to be remove
#         :param ner_keep_types_origin_list: a name list corenlp NER types which should be keep,
#                                             or will remove origin name and only keep NER types,
#                                             should input None or list
#         :param token_minlength: default 2 the minimal length of each token, else remove,
#                                 remove all the tokens which length is less than this length
#                                 if None then not remove
#         :param punctuations_set: punctuation set to be remove, especially
#         :param is_remove_no_alphabet_contains: is remove words(token) contains no alphabetic
#
#         """
#         if not isinstance(stopwords_set, set):
#             raise ValueError('stopwords_set must be set')
#
#         if not (
#                 isinstance(ner_keep_types_origin_list, list) or
#                 (ner_keep_types_origin_list is None)
#         ):
#             raise ValueError('ner_keep_types_origin_list must be list or None')
#
#         if not (
#                 isinstance(token_minlength, int) or
#                 (token_minlength is None)
#         ):
#             raise ValueError('token_minlength must be int or None')
#
#         if not isinstance(punctuations_set, set):
#             raise ValueError('punctuations_set must be set')
#
#         if not isinstance(is_remove_no_alphabet_contains, bool):
#             raise ValueError('is_removenum must be bool')
#
#         self.stopwords = stopwords_set
#
#         self.ner_keep_types_origin_list = ner_keep_types_origin_list if ner_keep_types_origin_list else list()
#
#         self.token_minlength = token_minlength
#
#         self.punctuations = punctuations_set if punctuations_set else set()
#
#         self.is_removenum = is_remove_no_alphabet_contains
#
#     def remove_ner(self, line):
#         """Remove the named entity and only leave the tag
#
#         Arguments:
#             line {str} -- text processed_data by the preprocessor
#
#         Returns:
#             str -- text with NE replaced by NE tags,
#             e.g. [NER:PERCENT]16_% becomes [NER:PERCENT]
#         """
#         # always make the line lower case
#         line = line.lower()
#         # remove ner for words of specific types:
#         if self.ner_keep_types_origin_list:  # have a loop if it is not None
#             for i in self.ner_keep_types_origin_list:
#                 line = re.sub(rf"(\[ner:{i.lower()}\])(\S+)", r"\2", line, flags=re.IGNORECASE)
#
#         # update for deeper search, remove the entity name
#         NERs = re.compile(r"(\[ner:\w+\])(\S+)", flags=re.IGNORECASE)
#         line = re.sub(NERs, r"\1", line)
#         return line
#
#     def remove_puct_num(self, line):
#         """Remove tokens that are only numerics and puctuation marks
#
#         Arguments:
#             line {str} -- text processed_data by the preprocessor
#
#         Returns:
#             str -- text with stopwords, numerics, 1-letter words removed
#         """
#         tokens = line.strip().lower().split(" ")  # do not use nltk.tokenize here
#         tokens = [re.sub(r"\[pos:.*?\]", "", t, flags=re.IGNORECASE) for t in tokens]
#
#         # these are tagged bracket and parenthesises
#         if self.punctuations or self.stopwords:
#             puncts_stops = (self.punctuations | self.stopwords)
#             # filter out numerics and 1-letter words as recommend by
#             # https://sraf.nd.edu/textual-analysis/resources/#StopWords
#         else:
#             puncts_stops = set()
#
#         def _lambda_filter_token_bool(t):
#             """
#             the judegement after the function is help to give
#             """
#             contain_alphabet = any(c.isalpha() for c in t) if self.is_removenum else True
#             is_not_punctuation_stopwords = t not in puncts_stops
#             is_biggerthan_minlength = len(t) >= self.token_minlength if self.token_minlength else True
#
#             return all([contain_alphabet, is_not_punctuation_stopwords, is_biggerthan_minlength])
#
#         tokens = filter(
#             # lambda t: any(c.isalpha() for c in t)
#             #           and (t not in puncts_stops)
#             #           and (len(t) > 1),
#             _lambda_filter_token_bool,
#             tokens,
#         )
#         return " ".join(tokens)
#
#     def clean(self, line, index):
#         """Main function that chains all filters together and applies to a string.
#         """
#         return (
#             functools.reduce(
#                 lambda obj, func: func(obj),
#                 [self.remove_ner, self.remove_puct_num],
#                 line,
#             ),
#             index,
#         )


# if __name__ == '__main__':
#     line_resolver = AnnotatedLineCleaner(
#         pos_tag_label='POS',
#         ner_tag_label='NER',
#         compounding_sep_string='[SEP]',
#         token_sep_string=' ',
#         sentiment_tag_label='SENTIMENT',
#         lower_case=True,
#         restruct_compounding_sep_string='_',
#         restruct_token_sep_string=' ',
#         full_token_compose_restriction=None,
#         token_remove_ner_tags_to_lessequal_then_num=None,
#         remove_stopwords_set=None,
#         remove_punctuations_set=None,
#         remove_token_lessequal_then_length=None,
#         remove_ner_options_dict=None,
#         remove_pos_options_dict=None,
#         clean_flag=1
#     )
#
#     demoline = """[SENTIMENT:Neutral] -[POS::] Dr.[POS:NNP][SEP][NER:PERSON]Martin[POS:NNP][SEP][NER:PERSON]Luther[POS:NNP][SEP][NER:PERSON]King[POS:NNP][SEP]Jr[POS:NNP][SEP]pop[POS:NN] on[POS:IN] by[POS:IN] any[POS:DT] of[POS:IN] we[POS:PRP$] location[POS:NNS] [NER:DATE]tomorrow[POS:NN] ,[POS:,] Wed.[POS:NNP][SEP]1/19[POS:NNP] ,[POS:,] as[POS:IN] we[POS:PRP] celebrate[POS:VBP] #NationalPopcornDay[POS:NNP] with[POS:IN] locally[POS:RB] grow[POS:VBN] popcorn[POS:NN] from[POS:IN] [NER:ORGANIZATION]Rickey[POS:NNP][SEP]&[POS:CC][SEP]Charlie[POS:NNP] 's[POS:POS] Popcorn[POS:NNP] ![POS:.] 125[POS:POS]""".strip().replace(
#         '\n', '')
#
#     print(line_resolver.clean(
#         demoline, 1
#     )
#     )
