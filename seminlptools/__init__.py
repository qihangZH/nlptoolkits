import culture_dictionary
import culture_models
import file_util
import preprocess
import preprocess_parallel
# out source pack
import datetime
import itertools
import os

"""parse"""


def parse_largefile(
        path_input_file_txt,
        path_output,
        path_input_file_ids,
        path_output_index,
        parse_line_func,
        chunk_size=100,
        start_index=None,
):
    """ A helper function that transforms an input file + a list of IDs of each line (documents + document_IDs) to two
     output files (processed documents + processed document IDs) by calling function_name on chunks of the input files.
      Each document can be decomposed into multiple processed documents (e.g. sentences).
    Supports parallel with Pool.

    Arguments:
        path_input_file_txt {str or Path} -- path to a text file, each line is a document
        path_output {str or Path} -- processed linesentence file (remove if exists)
        path_input_file_ids {str]} -- a list of input line ids
        path_output_index {str or Path} -- path to the index file of the output
        parse_line_func {callable} -- A function that processes a list of strings, list of ids and return a list of
        processed strings and ids.
        chunk_size {int} -- number of lines to process each time, increasing the default may increase performance
        start_index {int} -- line number to start from (index starts with 0)

    Writes:
        Write the ouput_file and output_index_file
    """
    try:
        if start_index is None:
            # if start from the first line, remove existing output file
            # else append to existing output file
            os.remove(str(path_output))
            os.remove(str(path_output_index))
    except OSError:
        pass
    assert file_util.line_counter(path_input_file_txt) == len(
        path_input_file_ids
    ), "Make sure the input file has the same number of rows as the input ID file. "

    with open(path_input_file_txt, newline="\n", encoding="utf-8", errors="ignore") as f_in:
        line_i = 0
        # jump to index
        if start_index is not None:
            # start at start_index line
            for _ in range(start_index):
                next(f_in)
            path_input_file_ids = path_input_file_ids[start_index:]
            line_i = start_index
        for next_n_lines, next_n_line_ids in zip(
                itertools.zip_longest(*[f_in] * chunk_size),
                itertools.zip_longest(*[iter(path_input_file_ids)] * chunk_size),
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
                    parse_line_func, next_n_lines, next_n_line_ids
            ):
                output_lines.append(output_line)
                output_line_ids.append(output_line_id)
            output_lines = "\n".join(output_lines) + "\n"
            output_line_ids = "\n".join(output_line_ids) + "\n"
            with open(path_output, "a", newline="\n", encoding='utf-8') as f_out:
                f_out.write(output_lines)
            if path_output_index is not None:
                with open(path_output_index, "a", newline="\n") as f_out:
                    f_out.write(output_line_ids)
