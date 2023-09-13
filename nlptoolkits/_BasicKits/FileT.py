import datetime
import itertools
import os
import pathos
import pandas as pd
import re
import tqdm
import base64
import gzip
from collections import defaultdict
from . import _BasicFuncT


# --------------------------------------------------------------------------
# l(-1) level functions/classes
# --------------------------------------------------------------------------
def _line_counter(a_file):
    """Count the number of lines in a text file

    Arguments:
        a_file {str or Path} -- input_data text file

    Returns:
        int -- number of lines in the file
    """
    n_lines = 0
    with open(a_file, "rb") as f:
        n_lines = sum(1 for _ in f)
    return n_lines


# --------------------------------------------------------------------------
# l0 level functions/classes
# --------------------------------------------------------------------------

def file_to_list(a_file):
    """Read a text file to a list, each line is an element
    
    Arguments:
        a_file {str or path} -- path to the file
    
    Returns:
        [str] -- list of lines in the input_data file, can be empty
    """
    file_content = []
    with open(a_file, "rb") as f:
        for l in f:
            file_content.append(l.decode(encoding="utf-8").strip())
    return file_content


def list_to_file(input_list, a_file, plain_validate=True, mode="w", encoding="utf-8", compress=False):
    """Write a list to a file, each element in a line
    The strings need to have no line break "\n" or they will be removed

    Keyword Arguments:
        plain_validate {bool} -- check if the number of lines in the file
            equals the length of the list (default: {True}), Only Useful when not compress
        mode {str} -- the argument of open()
        compress {bool} -- whether to compress the output file as a gz file (default: {False})
    """
    open_mode = mode
    if compress:
        a_file += '.gz'
        open_mode = 'wb'

    if compress:
        with gzip.open(a_file, open_mode) as f:
            for e in input_list:
                e = str(e).replace("\n", " ").replace("\r", " ")
                f.write("{}\n".format(e).encode(encoding))
    else:
        with open(a_file, mode, 8192000, encoding=encoding, newline="\n") as f:
            for e in input_list:
                e = str(e).replace("\n", " ").replace("\r", " ")
                f.write("{}\n".format(e))

        if plain_validate:
            assert _line_counter(a_file) == len(input_list)


def base64_to_file(base64_string, file_path, **kwargs):
    """
    Args:
        base64_string: encoded base64
        file_path: the path to save the file
        **kwargs: the arguments of open()

    Returns: None

    """
    data_bytes = base64.b64decode(base64_string)

    with open(file_path, mode='wb', **kwargs) as text_file:
        text_file.write(data_bytes)


def read_large_file(a_file, block_size=10000):
    """A generator to read text files into blocks
    Usage: 
    for block in read_large_file(filename):
        do_something(block)
    
    Arguments:
        a_file {str or path} -- path to the file
    
    Keyword Arguments:
        block_size {int} -- [number of lines in a block] (default: {10000})
    """
    block = []
    with open(a_file) as file_handler:
        for line in file_handler:
            block.append(line)
            if len(block) == block_size:
                yield block
                block = []
    # yield the last block
    if block:
        yield block


def write_dict_to_csv(culture_dict, file_name):
    """write the expanded dictionary to a csv file, each dimension is a column, the header includes dimension names

    Arguments:
        culture_dict {dict[str, list[str]]} -- an expanded dictionary {dimension: [words]}
        file_name {str} -- where to save the csv file?
    """
    pd.DataFrame.from_dict(culture_dict, orient="index").transpose().to_csv(
        file_name, index=None
    )


def read_dict_dictname_from_csv_dictset(file_name):
    """Read nlptoolkits dict from a csv file

    Arguments:
        file_name {str} -- expanded dictionary file

    Returns:
        culture_dict {dict{str: set(str)}} -- a nlptoolkits dict, dim name as key, set of expanded words as value
        all_dict_words {set(str)} -- a set of all words in the dict
    """
    print("Importing dict: {}".format(file_name))
    culture_dict_df = pd.read_csv(file_name, index_col=None)
    culture_dict = culture_dict_df.to_dict("list")
    for k in culture_dict.keys():
        culture_dict[k] = set([x for x in culture_dict[k] if x == x])  # remove nan

    all_dict_words = set()
    for key in culture_dict:
        all_dict_words |= culture_dict[key]

    for dim in culture_dict.keys():
        print("Number of words in {} dimension: {}".format(dim, len(culture_dict[dim])))

    return culture_dict, all_dict_words


"""sentence level -> doc level function"""


def calculate_doc_freq_dict(corpus):
    """Calcualte and dump a document-freq dict for all the words.

    Arguments:
        corpus {[str]} -- a list of documents

    Returns:
        {dict[str: int]} -- document freq for each word
    """
    print("Calculating document frequencies.")
    # document frequency
    doc_freq_dict = defaultdict(int)
    for doc in tqdm.tqdm(corpus):
        doc_splited = doc.split()
        words_in_doc = set(doc_splited)
        for word in words_in_doc:
            doc_freq_dict[word] += 1

    return doc_freq_dict


# --------------------------------------------------------------------------
# l1 level functions/classes
# --------------------------------------------------------------------------

def l1_sentence_to_doc_level_corpus(sent_corpus_file, sent_id_file,
                                    path_save_sent_corpus_file=None,
                                    path_save_sent_id_file=None
                                    ):
    """Construct document level corpus from sentence level corpus and write to disk.
    Dump "corpus_doc_level.pickle" and "doc_ids.pickle" to Path(global_options.OUTPUT_FOLDER, "scores", "temp").

    Arguments:
        sent_corpus_file {str or Path} -- The sentence corpus after parsing and cleaning, each line is a sentence
        sent_id_file {str or Path} -- The sentence ID file, each line correspond to a line in the sent_co(docID_sentenceID)

    Returns:
        [str], [str], int -- a tuple of a list of documents, a list of document IDs, and the number of documents
    """
    # sentence level corpus
    sent_corpus = file_to_list(sent_corpus_file)
    sent_IDs = file_to_list(sent_id_file)
    assert len(sent_IDs) == len(sent_corpus)

    # doc id for each sentence
    """old-version:"""
    # doc_ids = [x.split("_")[0] for x in sent_IDs]
    """new-version:"""
    doc_ids = pd.Series(sent_IDs).str.extract(pat=r'^(.*)_[^_]*$',  # pick the group which before the last underscore
                                              expand=False, flags=re.IGNORECASE).to_list()

    # concat all text from the same doc
    id_doc_dict = defaultdict(lambda: "")
    for i, id in enumerate(doc_ids):
        id_doc_dict[id] += " " + sent_corpus[i]
    # create doc level corpus
    corpus = list(id_doc_dict.values())
    doc_ids = list(id_doc_dict.keys())
    assert len(corpus) == len(doc_ids)
    N_doc = len(corpus)

    if not (path_save_sent_corpus_file is None):
        with open(path_save_sent_corpus_file, 'w', encoding='utf-8') as f:
            for sentence in corpus:
                f.write(sentence + '\n')  # add a newline character to separate sentences

    if not (path_save_sent_id_file is None):
        with open(path_save_sent_id_file, 'w', encoding='utf-8') as f:
            for ids in doc_ids:
                f.write(ids + '\n')  # add a newline character to separate sentences

    return corpus, doc_ids, N_doc


"""Process file"""


def l1_process_largefile(
        path_input_txt,
        path_output_txt,
        input_index_list,
        path_output_index_txt,
        process_line_func,
        chunk_size=100,
        start_iloc=None,
):
    """ A helper function that transforms an input_data file + a list of IDs of each line (documents + document_IDs)
    to two output files (processed_data documents + processed_data document IDs)
    by calling function_name on chunks of the input_data files.
      Each document can be decomposed into multiple processed_data documents (e.g. sentences).
      Not support Multiprocessor


    Arguments:
    :param path_input_txt:  {str or Path} path to a text file, each line is a document
    :param path_output_txt: {str or Path} processed_data linesentence file (remove if exists)
    :param input_index_list: {str} -- a list of input_data line ids
    :param path_output_index_txt: {str or Path} -- path to the index file of the output
    :param process_line_func: {callable} -- A function that processes a list of strings, list of ids and return
        a list of processed_data strings and ids. func(line_text, line_ids)
    :param chunk_size: {int} -- number of lines to process each time, increasing the default may increase performance
    :param start_iloc: {int} -- line number to start from (index starts with 0)

    Writes:
        Write the ouput_file and output_index_file

      Not support Multiprocessor
    """

    # check data must save in txt and dim be 1:

    try:
        if start_iloc is None:
            # if start from the first line, remove existing output file
            # else append to existing output file
            os.remove(str(path_output_txt))
            os.remove(str(path_output_index_txt))
    except OSError:
        pass
    assert _line_counter(path_input_txt) == len(
        input_index_list
    ), "Make sure the input_data file has the same number of rows as the input_data ID file. "

    with open(path_input_txt, newline="\n", encoding="utf-8", errors="ignore") as f_in:
        line_i = 0
        # jump to index
        if start_iloc is not None:
            # start at start_index line
            for _ in range(start_iloc):
                next(f_in)
            input_index_list = input_index_list[start_iloc:]
            line_i = start_iloc
        for next_n_lines, next_n_line_ids in zip(
                itertools.zip_longest(*[f_in] * chunk_size),
                itertools.zip_longest(*[iter(input_index_list)] * chunk_size),
        ):
            line_i += chunk_size
            print(datetime.datetime.now())
            print(f"Processing line: {line_i}.")
            next_n_lines = list(filter(None.__ne__, next_n_lines))
            next_n_line_ids = list(filter(None.__ne__, next_n_line_ids))
            output_lines = []
            output_line_ids = []
            # Use parse_parallel.py to speed things up
            for output_line, output_line_id in map(
                    process_line_func, next_n_lines, next_n_line_ids
            ):
                output_lines.append(output_line)
                output_line_ids.append(output_line_id)
            output_lines = "\n".join(output_lines) + "\n"
            output_line_ids = "\n".join(output_line_ids) + "\n"
            with open(path_output_txt, "a", newline="\n", encoding='utf-8') as f_out:
                f_out.write(output_lines)
            if path_output_index_txt is not None:
                with open(path_output_index_txt, "a", newline="\n", encoding="utf-8") as f_out:
                    f_out.write(output_line_ids)


def l1_mp_process_largefile(
        path_input_txt,
        path_output_txt,
        input_index_list,
        path_output_index_txt,
        process_line_func,
        processes: int,
        chunk_size=100,
        start_iloc=None,

):
    """ A helper function that transforms an input_data file + a list of IDs of each line (documents + document_IDs)
    to two output files (processed_data documents + processed_data document IDs)
    by calling function_name on chunks of the input_data files.
      Each document can be decomposed into multiple processed_data documents (e.g. sentences).



    Arguments:
    :param path_input_txt:  {str or Path} path to a text file, each line is a document
    :param path_output_txt: {str or Path} processed_data linesentence file (remove if exists)
    :param input_index_list: {str} -- a list of input_data line ids
    :param path_output_index_txt: {str or Path} -- path to the index file of the output
    :param process_line_func: {callable} -- A function that processes a list of strings, list of ids and return
        a list of processed_data strings and ids. func(line_text, line_ids)
    :param processes: {int} -- the core to use, should be same as the argument of your project,
        like globaloption.NCORES
    :param chunk_size: {int} -- number of lines to process each time, increasing the default may increase performance
    :param start_iloc: {int} -- line number to start from (index starts with 0)

    Writes:
        Write the ouput_file and output_index_file
    """
    try:
        if start_iloc is None:
            # if start from the first line, remove existing output file
            # else append to existing output file
            os.remove(str(path_output_txt))
            os.remove(str(path_output_index_txt))
    except OSError:
        pass
    assert _line_counter(path_input_txt) == len(
        input_index_list
    ), "Make sure the input_data file has the same number of rows as the input_data ID file. "

    with open(path_input_txt, newline="\n", encoding="utf-8", errors="ignore") as f_in:
        line_i = 0
        # jump to index
        if start_iloc is not None:
            # start at start_index line
            for _ in range(start_iloc):
                next(f_in)
            input_index_list = input_index_list[start_iloc:]
            line_i = start_iloc
        for next_n_lines, next_n_line_ids in zip(
                itertools.zip_longest(*[f_in] * chunk_size),
                itertools.zip_longest(*[iter(input_index_list)] * chunk_size),
        ):
            line_i += chunk_size
            print(datetime.datetime.now())
            print(f"Processing line: {line_i}.")
            next_n_lines = list(filter(None.__ne__, next_n_lines))
            next_n_line_ids = list(filter(None.__ne__, next_n_line_ids))
            output_lines = []
            output_line_ids = []
            with pathos.multiprocessing.Pool(processes=processes,
                                             initializer=_BasicFuncT.processes_interrupt_initiator
                                             ) as pool:
                for output_line, output_line_id in pool.starmap(
                        process_line_func, zip(next_n_lines, next_n_line_ids)
                ):
                    output_lines.append(output_line)
                    output_line_ids.append(output_line_id)
            output_lines = "\n".join(output_lines) + "\n"
            output_line_ids = "\n".join(output_line_ids) + "\n"
            with open(path_output_txt, "a", newline="\n", encoding='utf-8') as f_out:
                f_out.write(output_lines)
            if path_output_index_txt is not None:
                with open(path_output_index_txt, "a", newline="\n", encoding='utf-8') as f_out:
                    f_out.write(output_line_ids)
