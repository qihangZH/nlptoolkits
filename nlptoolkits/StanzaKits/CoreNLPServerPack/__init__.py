import functools
import typing
import stanza.server
from . import AnnotatedResolverT, AnnotationT, _GlobalArgs
from ... import _BasicKits

# --------------------------------------------------------------------------
# L0 Preprocessing AUTO functions
# --------------------------------------------------------------------------

"""Preprocessing: parser"""


def auto_doc_to_sentences_annotator(
        endpoint,
        memory,
        processes: int,
        path_input_txt,
        path_output_txt,
        input_index_list,
        path_output_index_txt,
        chunk_size=100,
        start_iloc=None,
        mwe_dep_types: set = set(["mwe", "compound", "compound:prt"]),
        annotation_choices: typing.Iterable[str] = [
            'Lemmatize', 'POStags', 'NERtags', 'DepParseMWECompounds'
        ],
        pos_tag_label: str = _GlobalArgs.DEFAULT_POS_TAG_LABEL,
        ner_tag_label: str = _GlobalArgs.DEFAULT_NER_TAG_LABEL,
        sentiment_tag_label: str = _GlobalArgs.DEFAULT_SENTIMENT_TAG_LABEL,
        compounding_sep_string: str = _GlobalArgs.DEFAULT_COMPOUNDING_SEP_STRING,
        token_sep_string: str = _GlobalArgs.DEFAULT_TOKEN_SEP_STRING,
        properties: typing.Optional[dict] = {
            "ner.applyFineGrained": "false",
            "annotators": "tokenize, ssplit, pos, lemma, ner, depparse",
        },
        **kwargs):
    """
    parse the sentence with NER tagging and MWEs concatenated, etc
    For example: Rickey Hall ->
    [{self.ner_tag_label}:PERSON]Rickey[{self.pos_tag_label}:NNP]_Hall[{self.pos_tag_label}:NNP]
    Moreover, if sentiment is choosed, then the output will be:
    [{self.sentiment_tag_label}:2] [{self.ner_tag_label}:PERSON]Rickey[{self.pos_tag_label}:NNP]_Hall[{self.pos_tag_label}:NNP]
    :param memory: memory using, should be str like "\d+G"
    :param processes: how much processes does nlp use.
    :param endpoint: endpoint in stanfordnlp.server.CoreNLPClient, should be address of port
    :param path_input_txt:  {str or Path} path to a text file, each line is a document
    :param path_output_txt: {str or Path} processed_data linesentence file (remove if exists)
    :param input_index_list: {str} -- a list of input_data line ids
    :param path_output_index_txt: {str or Path} -- path to the index file of the output
    :param chunk_size: {int} -- number of lines to process each time, increasing the default may increase performance
    :param start_iloc: {int} -- line number to start from (index starts with 0)
    :param mwe_dep_types: the set of mwe dep types a list of MWEs in Universal Dependencies v1
            (default: s{set(["mwe", "compound", "compound:prt"])})
            see: http://universaldependencies.org/docsv1/u/dep/compound.html
            and http://universaldependencies.org/docsv1/u/dep/mwe.html
    :param annotation_choices: a list of annotation choices. The order is not important.
        Here is the explaination of each element:
        **'Lemmatize'**: lemmatize the word, for example, wanted -> want
        **'POStags'**: add POS tags to the word **IN SUFFIX**, for example, want -> want[pos:VB]
            see https://erwinkomen.ruhosting.nl/eng/2014_Longdale-Labels.htm for detail.
            Besides, other 12 kinds of punctuations are not list in penn treebank pos. Like `,:#
        **'NERtags'**: add NER tags to the word *IN PREFIX*,
            for example, Stanford University -> [NER:ORGANIZATION]Stanford_University
            For English, by default, this annotator recognizes named (PERSON, LOCATION, ORGANIZATION, MISC),
            numerical (MONEY, NUMBER, ORDINAL, PERCENT), and temporal (DATE, TIME, DURATION, SET) entities (12 classes).
        **'DepParseMWECompounds'**: use dep parsing to concatenate MWEs and compounds, for example, go to -> go_to.
                                If you want use this function, you have to set mwe_dep_types.
                                like mwe_dep_types: set = set(["mwe", "compound", "compound:prt"])
        **SentenceSentiment**: add sentence sentiment to the sentence, if use, then 'sentiment' have to be added to
            stanford corenlp client pipeline choices.
    :param pos_tag_label: the POS tag used in the output, default "POS", finally will become "[POS:...]"
    :param ner_tag_label: the NER tag used in the output, default "NER", finally will become "[NER:...]"
    :param sentiment_tag_label: the sentiment tag used in the output, default "SENTIMENT",
        finally will become "[SENTIMENT:...] <other part of the annotated tokens in that line(sentence)...>"
    :param compounding_sep_string: the separator string used to concatenate MWEs and compounds, default "[SEP]"
    :param token_sep_string: the separator string used to separate tokens, default " "
    :param properties: the properties of stanfordnlp.server.CoreNLPClient, properties['annotators'] should contain
                        PIPELINE like 'tokenize', 'ssplit', 'pos', 'lemma', 'ner', 'depparse', 'sentiment', etc.
                        if loss some important components, the function will raise error.
    :param kwargs: the other arguments of stanfordnlp.server.CoreNLPClient
    """
    # supply for arguments
    kwargs['timeout'] = kwargs['timeout'] if 'timeout' in kwargs else 12000000
    kwargs['max_char_length'] = kwargs['max_char_length'] if 'max_char_length' in kwargs else 1000000
    kwargs['start_server'] = kwargs['start_server'] \
        if 'start_server' in kwargs else stanza.server.StartServer.FORCE_START
    kwargs['be_quiet'] = kwargs['be_quiet'] if 'be_quiet' in kwargs else True

    def _lambda_process_line(line, lineID, corpus_processor):
        """Process each line and return a tuple of sentences, sentence_IDs,

        Arguments:
            line {str} -- a document
            lineID {str} -- the document ID

        Returns:
            str, str -- processed_data document with each sentence in a line,
                        sentence IDs with each in its own line: lineID_0 lineID_1 ...
        """
        try:
            sentences_processed, doc_sent_ids = corpus_processor.parse_line_to_sentences(
                line, lineID
            )
            return "\n".join(sentences_processed), "\n".join(doc_sent_ids)
        except Exception as e:
            print(e)
            print("Exception in line: {}".format(lineID))

    with stanza.server.CoreNLPClient(
            memory=memory,
            threads=processes,
            endpoint=endpoint,  # must type in
            properties=properties,
            **kwargs
    ) as client:

        if processes > 1:
            """you must make corenlp and mp.Pool's port are same!!!"""
            corpus_preprocessor = AnnotationT.LineAnnotatorParallel(
                annotation_choices=annotation_choices,
                mwe_dep_types=mwe_dep_types,
                pos_tag_label=pos_tag_label,
                ner_tag_label=ner_tag_label,
                sentiment_tag_label=sentiment_tag_label,
                compounding_sep_string=compounding_sep_string,
                token_sep_string=token_sep_string
            )
            _BasicKits.FileT.l1_mp_process_largefile(
                path_input_txt=path_input_txt,
                path_output_txt=path_output_txt,
                input_index_list=input_index_list,
                path_output_index_txt=path_output_index_txt,
                # you must make corenlp and mp.Pool's port are same
                process_line_func=lambda x, y: corpus_preprocessor.parse_line_to_sentences(x, y, endpoint),
                processes=processes,
                chunk_size=chunk_size,
                start_iloc=start_iloc
            )
        else:
            corpus_preprocessor = AnnotationT.LineAnnotator(client=client,
                                                            annotation_choices=annotation_choices,
                                                            mwe_dep_types=mwe_dep_types,
                                                            pos_tag_label=pos_tag_label,
                                                            ner_tag_label=ner_tag_label,
                                                            sentiment_tag_label=sentiment_tag_label,
                                                            compounding_sep_string=compounding_sep_string,
                                                            token_sep_string=token_sep_string
                                                            )

            _BasicKits.FileT.l1_process_largefile(
                path_input_txt=path_input_txt,
                path_output_txt=path_output_txt,
                input_index_list=input_index_list,
                path_output_index_txt=path_output_index_txt,
                process_line_func=lambda x, y: _lambda_process_line(x, y, corpus_preprocessor),
                chunk_size=chunk_size,
                start_iloc=start_iloc
            )


"""Preprocessing: clean the parsed file"""


def auto_clean_annotated_txt(path_in_parsed_txt,
                             path_out_cleaned_txt,
                             processes: int,
                             pos_tag_label: str = _GlobalArgs.DEFAULT_POS_TAG_LABEL,
                             ner_tag_label: str = _GlobalArgs.DEFAULT_NER_TAG_LABEL,
                             sentiment_tag_label: str = _GlobalArgs.DEFAULT_SENTIMENT_TAG_LABEL,
                             compounding_sep_string: str = _GlobalArgs.DEFAULT_COMPOUNDING_SEP_STRING,
                             token_sep_string: str = _GlobalArgs.DEFAULT_TOKEN_SEP_STRING,
                             lower_case: bool = True,
                             restruct_compounding_sep_string: str = "_",
                             restruct_token_sep_string: str = _GlobalArgs.DEFAULT_TOKEN_SEP_STRING,
                             full_token_compose_restriction: typing.Optional[str] = None,
                             token_remove_ner_tags_to_lessequal_then_num: typing.Optional[int] = 1,
                             remove_stopwords_set: typing.Optional[set] = None,
                             remove_punctuations_set: typing.Optional[set] = None,
                             remove_token_lessequal_then_length: typing.Optional[int] = None,
                             remove_ner_options_dict: typing.Optional[dict] = {'removes_original_text': 'all'},
                             remove_pos_options_dict: typing.Optional[dict] = {'remove_tags': 'all',
                                                                               'remove_tags_and_original_text': _GlobalArgs.POS_PENN_TREE_BANK_TAGS_PUNCT_UPPER_SET
                                                                               },
                             clean_flag: int = _GlobalArgs.FLAG_ANNOTATED_LINE_CLEANER_CLEANEDLINE,
                             ):
    """
    clean the entire corpus (output from CoreNLP)
    see more info, see AnnotatedResolverT.AnnotatedLineCleaner
    :param path_in_parsed_txt: {str or Path} path to the parsed_data file
    :param path_out_cleaned_txt: {str or Path} path to the cleaned_data file (remove if exists)
    :param processes: {int} -- number of processes to use
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
    line_clearner_cls = AnnotatedResolverT.AnnotatedLineCleaner(
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
    if processes > 1:
        _BasicKits.FileT.l1_process_largefile(
            path_input_txt=path_in_parsed_txt,
            path_output_txt=path_out_cleaned_txt,
            input_index_list=[
                str(i) for i in range(_BasicKits.FileT._line_counter(path_in_parsed_txt))
            ],  # fake IDs (do not need IDs for this function).
            path_output_index_txt=None,
            process_line_func=functools.partial(line_clearner_cls.clean),
            chunk_size=200000,
        )

    else:
        _BasicKits.FileT.l1_mp_process_largefile(
            path_input_txt=path_in_parsed_txt,
            path_output_txt=path_out_cleaned_txt,
            input_index_list=[
                str(i) for i in range(_BasicKits.FileT._line_counter(path_in_parsed_txt))
            ],  # fake IDs (do not need IDs for this function).
            path_output_index_txt=None,
            process_line_func=functools.partial(line_clearner_cls.clean),
            processes=processes,
            chunk_size=200000,
        )
