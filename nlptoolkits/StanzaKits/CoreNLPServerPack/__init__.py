import functools
import typing
import stanza.server
from . import CleanT, ParserT
from ... import _BasicKits


# --------------------------------------------------------------------------
# L0 Preprocessing AUTO functions
# --------------------------------------------------------------------------

"""Preprocessing: parser"""


def auto_doc_to_sentences_parser(
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
        parsing_choices: typing.Iterable[str] = ['Lemmatize', 'POStags', 'NERtags', 'DepParseMWECompounds'],
        **kwargs):
    """
    :param memory: memory using, should be str like "\d+G"
    :param processes: how much processes does nlp use.
    :param endpoint: endpoint in stanfordnlp.server.CoreNLPClient, should be address of port
    :param path_input_txt:  {str or Path} path to a text file, each line is a document
    :param path_output_txt: {str or Path} processed_data linesentence file (remove if exists)
    :param input_index_list: {str} -- a list of input_data line ids
    :param path_output_index_txt: {str or Path} -- path to the index file of the output
    :param chunk_size: {int} -- number of lines to process each time, increasing the default may increase performance
    :param start_iloc: {int} -- line number to start from (index starts with 0)
    :param kwargs: the other arguments of stanfordnlp.server.CoreNLPClient
    :param mwe_dep_types: the set of mwe dep types a list of MWEs in Universal Dependencies v1
            (default: s{set(["mwe", "compound", "compound:prt"])})
            see: http://universaldependencies.org/docsv1/u/dep/compound.html
            and http://universaldependencies.org/docsv1/u/dep/mwe.html
    :param parsing_choices: a list of parsing choices. The order is not important.
        Here is the explaination of each element:
        **'Lemmatize'**: lemmatize the word, for example, wanted -> want
        **'POStags'**: add POS tags to the word **IN SUFFIX**, for example, want -> want[pos:VB]
            see https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html for detail.
            Besides, other 12 kinds of punctuations are not list in penn treebank pos. Like `,:#
        **'NERtags'**: add NER tags to the word *IN PREFIX*,
            for example, Stanford University -> [NER:ORGANIZATION]Stanford_University
        **'DepParseMWECompounds'**: use dep parsing to concatenate MWEs and compounds, for example, go to -> go_to.
                                If you want use this function, you have to set mwe_dep_types.
                                like mwe_dep_types: set = set(["mwe", "compound", "compound:prt"])

    Writes:
        Write the ouput_file and output_index_file

    """
    # supply for arguments
    kwargs['properties'] = kwargs['properties'] if 'properties' in kwargs else {
        "ner.applyFineGrained": "false",
        "annotators": "tokenize, ssplit, pos, lemma, ner, depparse",
    }
    kwargs['timeout'] = kwargs['timeout'] if 'timeout' in kwargs else 12000000
    kwargs['max_char_length'] = kwargs['max_char_length'] if 'max_char_length' in kwargs else 1000000
    kwargs['start_server'] = kwargs['start_server'] \
        if 'start_server' in kwargs else stanza.server.StartServer.TRY_START
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
            **kwargs
    ) as client:

        if processes > 1:
            """you must make corenlp and mp.Pool's port are same!!!"""
            corpus_preprocessor = ParserT.DocParserParallel(
                parsing_choices=parsing_choices,
                mwe_dep_types=mwe_dep_types)
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
            corpus_preprocessor = ParserT.DocParser(client=client,
                                                    parsing_choices=parsing_choices,
                                                    mwe_dep_types=mwe_dep_types
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


def auto_clean_parsed_txt(path_in_parsed_txt, path_out_cleaned_txt, stopwords_set, processes: int, **kwargs):
    """
    clean the entire corpus (output from CoreNLP)
    see more info, see PreprocessT.TextCleaner

    :param path_in_parsed_txt: the parsed file(txt) which has be dealed by stanford corenlp
    :param path_out_cleaned_txt: the path of cleaned file to be output, will be tagged and some words are removed
    :param processes: how much processes to be used
    :param stopwords_set: the stopwords, should be removed.
    :param kwargs: the arguments which would be passed to PreprocessT.TextCleaner(stopwords_set, **kwargs)
        stopwords_set/ner_keep_types_origin_list/token_minlength/punctuations_set/is_remove_no_alphabet_contains,

    """
    a_text_clearner = CleanT.LineTextCleaner(stopwords_set, **kwargs)
    if processes > 1:
        _BasicKits.FileT.l1_process_largefile(
            path_input_txt=path_in_parsed_txt,
            path_output_txt=path_out_cleaned_txt,
            input_index_list=[
                str(i) for i in range(_BasicKits.FileT._line_counter(path_in_parsed_txt))
            ],  # fake IDs (do not need IDs for this function).
            path_output_index_txt=None,
            process_line_func=functools.partial(a_text_clearner.clean),
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
            process_line_func=functools.partial(a_text_clearner.clean),
            processes=processes,
            chunk_size=200000,
        )
