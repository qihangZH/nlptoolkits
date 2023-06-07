from . import _BasicT
from . import PreprocessT
from . import Wrd2vScorerT
from . import qihangfuncs
# out source pack

import os
import stanfordnlp.server
import shutil
import functools

# --------------------------------------------------------------------------
# l0 level functions
# --------------------------------------------------------------------------

"""Other functions"""


def delete_whole_dir(directory):
    """delete the whole dir..."""
    if os.path.exists(directory) and os.path.isdir(directory):
        shutil.rmtree(directory)


"""function aliases"""


def alias_file_to_list(*args, **kwargs):
    """alias to _BasicT.file_to_list"""
    return _BasicT.file_to_list(*args, **kwargs)


def alias_write_dict_to_csv(*args, **kwargs):
    """alias to _BasicT.write_dict_to_csv"""
    return _BasicT.write_dict_to_csv(*args, **kwargs)


# --------------------------------------------------------------------------
# L0 Preprocessing AUTO functions
# --------------------------------------------------------------------------


"""Preprocessing: parser"""


def auto_parser(
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
            sentences_processed, doc_sent_ids = corpus_processor.process_document(
                line, lineID
            )
            return "\n".join(sentences_processed), "\n".join(doc_sent_ids)
        except Exception as e:
            print(e)
            print("Exception in line: {}".format(lineID))

    with stanfordnlp.server.CoreNLPClient(
            memory=memory,
            threads=processes,
            endpoint=endpoint,  # must type in
            **kwargs
    ) as client:

        if processes > 1:
            """you must make corenlp and mp.Pool's port are same!!!"""
            corpus_preprocessor = PreprocessT.DocParserParallel(mwe_dep_types=mwe_dep_types)
            _BasicT.l1_mp_process_largefile(
                path_input_txt=path_input_txt,
                path_output_txt=path_output_txt,
                input_index_list=input_index_list,
                path_output_index_txt=path_output_index_txt,
                # you must make corenlp and mp.Pool's port are same
                process_line_func=lambda x, y: corpus_preprocessor.process_document(x, y, endpoint),
                processes=processes,
                chunk_size=chunk_size,
                start_iloc=start_iloc
            )
        else:
            corpus_preprocessor = PreprocessT.DocParser(client=client, mwe_dep_types=mwe_dep_types)

            _BasicT.l1_process_largefile(
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
    a_text_clearner = PreprocessT.TextCleaner(stopwords_set, **kwargs)
    if processes > 1:
        _BasicT.l1_process_largefile(
            path_input_txt=path_in_parsed_txt,
            path_output_txt=path_out_cleaned_txt,
            input_index_list=[
                str(i) for i in range(_BasicT._line_counter(path_in_parsed_txt))
            ],  # fake IDs (do not need IDs for this function).
            path_output_index_txt=None,
            process_line_func=functools.partial(a_text_clearner.clean),
            chunk_size=200000,
        )

    else:
        _BasicT.l1_mp_process_largefile(
            path_input_txt=path_in_parsed_txt,
            path_output_txt=path_out_cleaned_txt,
            input_index_list=[
                str(i) for i in range(_BasicT._line_counter(path_in_parsed_txt))
            ],  # fake IDs (do not need IDs for this function).
            path_output_index_txt=None,
            process_line_func=functools.partial(a_text_clearner.clean),
            processes=processes,
            chunk_size=200000,
        )


"""Preprocessing: train and transform the bigram model, concat two words into one"""


def auto_bigram_fit_transform_txt(path_input_clean_txt,
                                  path_output_transformed_txt,
                                  path_output_model_mod,
                                  phrase_min_length: int,
                                  stopwords_set,
                                  threshold=None,
                                  scoring="original_scorer"
                                  ):
    """
    transform the sep two length words to concat in a word which join by '_'
    which means uni-gram -> bi-gram words.
    you can recursive this function to get the target tri-gram or bigger phrases.

    Args:
        path_input_clean_txt:
        path_output_transformed_txt:
        path_output_model_mod:
        phrase_min_length:
        stopwords_set:
        threshold:
        scoring:

    Returns:

    """

    # precheck
    if not str(path_output_model_mod).endswith('.mod'):
        raise ValueError('Model must end with .mod')

    # train and apply a phrase model to detect 3-word phrases ----------------
    PreprocessT.train_bigram_model(
        input_path=path_input_clean_txt,
        model_path=path_output_model_mod,
        phrase_min_length=phrase_min_length,
        phrase_threshold=threshold,
        stopwords_set=stopwords_set
    )
    PreprocessT.file_bigramer(
        input_path=path_input_clean_txt,
        output_path=path_output_transformed_txt,
        model_path=path_output_model_mod,
        scoring=scoring,
        threshold=threshold,
    )
