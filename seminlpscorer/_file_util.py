import datetime
import itertools
import os
import pathos
from . import qihangfuncs


# --------------------------------------------------------------------------
# l0 level functions/classes
# --------------------------------------------------------------------------
def line_counter(a_file):
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


def list_to_file(list, a_file, validate=True):
    """Write a list to a file, each element in a line
    The strings needs to have no line break "\n" or they will be removed
    
    Keyword Arguments:
        validate {bool} -- check if number of lines in the file
            equals to the length of the list (default: {True})
    """
    with open(a_file, "w", 8192000, encoding="utf-8", newline="\n") as f:
        for e in list:
            e = str(e).replace("\n", " ").replace("\r", " ")
            f.write("{}\n".format(e))
    if validate:
        assert line_counter(a_file) == len(list)


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


# --------------------------------------------------------------------------
# l1 level functions/classes
# --------------------------------------------------------------------------

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
    assert line_counter(path_input_txt) == len(
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
    assert line_counter(path_input_txt) == len(
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
                                             initializer=qihangfuncs.threads_interrupt_initiator
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
