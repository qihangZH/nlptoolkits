import functools
import re
import typing


class LineTextCleaner:
    """Clean the text parsed by CoreNLP (preprocessor)
    """

    def __init__(self,
                 stopwords_set: set,
                 ner_keep_types_origin_list: typing.Optional[list] = None,
                 token_minlength: typing.Optional[int] = 2,
                 punctuations_set: set = set(["-lrb-", "-rrb-", "-lsb-", "-rsb-", "'s"]),
                 is_remove_no_alphabet_contains: bool = True,
                 ):
        """
        :param stopwords_set: stop word set to be remove
        :param ner_keep_types_origin_list: a name list corenlp NER types which should be keep,
                                            or will remove origin name and only keep NER types,
                                            should input None or list
        :param token_minlength: default 2 the minimal length of each token, else remove,
                                remove all the tokens which length is less than this length
                                if None then not remove
        :param punctuations_set: punctuation set to be remove, especially
        :param is_remove_no_alphabet_contains: is remove words(token) contains no alphabetic

        """
        if not isinstance(stopwords_set, set):
            raise ValueError('stopwords_set must be set')

        if not (
                isinstance(ner_keep_types_origin_list, list) or
                (ner_keep_types_origin_list is None)
        ):
            raise ValueError('ner_keep_types_origin_list must be list or None')

        if not (
                isinstance(token_minlength, int) or
                (token_minlength is None)
        ):
            raise ValueError('token_minlength must be int or None')

        if not isinstance(punctuations_set, set):
            raise ValueError('punctuations_set must be set')

        if not isinstance(is_remove_no_alphabet_contains, bool):
            raise ValueError('is_removenum must be bool')

        self.stopwords = stopwords_set

        self.ner_keep_types_origin_list = ner_keep_types_origin_list if ner_keep_types_origin_list else list()

        self.token_minlength = token_minlength

        self.punctuations = punctuations_set if punctuations_set else set()

        self.is_removenum = is_remove_no_alphabet_contains

    def remove_ner(self, line):
        """Remove the named entity and only leave the tag

        Arguments:
            line {str} -- text processed_data by the preprocessor

        Returns:
            str -- text with NE replaced by NE tags,
            e.g. [NER:PERCENT]16_% becomes [NER:PERCENT]
        """
        # always make the line lower case
        line = line.lower()
        # remove ner for words of specific types:
        if self.ner_keep_types_origin_list:  # have a loop if it is not None
            for i in self.ner_keep_types_origin_list:
                line = re.sub(rf"(\[ner:{i.lower()}\])(\S+)", r"\2", line, flags=re.IGNORECASE)

        # update for deeper search, remove the entity name
        NERs = re.compile(r"(\[ner:\w+\])(\S+)", flags=re.IGNORECASE)
        line = re.sub(NERs, r"\1", line)
        return line

    def remove_puct_num(self, line):
        """Remove tokens that are only numerics and puctuation marks

        Arguments:
            line {str} -- text processed_data by the preprocessor

        Returns:
            str -- text with stopwords, numerics, 1-letter words removed
        """
        tokens = line.strip().lower().split(" ")  # do not use nltk.tokenize here
        tokens = [re.sub("\[pos:.*?\]", "", t, flags=re.IGNORECASE) for t in tokens]

        # these are tagged bracket and parenthesises
        if self.punctuations or self.stopwords:
            puncts_stops = (self.punctuations | self.stopwords)
            # filter out numerics and 1-letter words as recommend by
            # https://sraf.nd.edu/textual-analysis/resources/#StopWords
        else:
            puncts_stops = set()

        def _lambda_filter_token_bool(t):
            """
            the judegement after the function is help to give
            """
            contain_alphabet = any(c.isalpha() for c in t) if self.is_removenum else True
            is_not_punctuation_stopwords = t not in puncts_stops
            is_biggerthan_minlength = len(t) >= self.token_minlength if self.token_minlength else True

            return all([contain_alphabet, is_not_punctuation_stopwords, is_biggerthan_minlength])

        tokens = filter(
            # lambda t: any(c.isalpha() for c in t)
            #           and (t not in puncts_stops)
            #           and (len(t) > 1),
            _lambda_filter_token_bool,
            tokens,
        )
        return " ".join(tokens)

    def clean(self, line, index):
        """Main function that chains all filters together and applies to a string.
        """
        return (
            functools.reduce(
                lambda obj, func: func(obj),
                [self.remove_ner, self.remove_puct_num],
                line,
            ),
            index,
        )
