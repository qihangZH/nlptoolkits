import typing

from . import BitermplusT
from . import WordninjaT
from . import LangDetectT
from . import IOHandlerT
from . import LexRankT

import pathos
import os
import math
import numpy as np
import pandas as pd
from .. import _BasicKits


def auto_filelist_reader(filepath_list, file_mime_list: list, processes=os.cpu_count(),
                         is_trim_text_wordninja=True, mime_readerfunc_dict: typing.Optional[dict] = None,
                         ):
    """
    Auto return the file-Text(If contains any), the function use imap+loop, so result is definitely in sequence.
    return the tuple of result list and the error list. The readers are given in defaults and all ..._to_single_line_str
    Series. However, You may update new config-function by yourself in mime_readerfunc_dict.
    For example {'pdf': lambda x: IOHandlerT.convert_pdf_to_single_line_str(x, start_index=2, end_index=3)}
    This will cause the pdf reader only read the pdf at page index 2(or page num 3).
    :param filepath_list: the path list of file to be read
    :param file_mime_list: the mime, like rtf, doc, actually the suffix of file
    :param processes: Use Howmuch Processes to run
    :param is_trim_text_wordninja: Is trim the data use wordninja?
    :param mime_readerfunc_dict: Optional, update the function map, like {'pdf':func}
    :return : the tuple(result_list, error_list), any error one should result in result_list as None.

    """
    # default value should be False for suppress the warning.

    func_map = {
        'rtf': lambda x: IOHandlerT.convert_rtf_to_single_line_str(x, suppress_warn=True),
        'doc': lambda x: IOHandlerT.convert_doc_to_single_line_str(x, suppress_warn=True),
        'pdf': lambda x: IOHandlerT.convert_pdf_to_single_line_str(x, suppress_warn=True),
        'html': lambda x: IOHandlerT.convert_html_to_single_line_str(x, suppress_warn=True)
    }

    if isinstance(mime_readerfunc_dict, dict):
        func_map.update(mime_readerfunc_dict)

    def _lambda_reader_loop(task_list_tuple):

        err_list = []
        rst_list = []

        filepaths, file_mimes = task_list_tuple

        readers = [func_map[m] for m in file_mimes]

        for pos in range(len(filepaths)):

            try:
                if is_trim_text_wordninja:
                    rst_list.append(
                        WordninjaT.replace_sequence_letters_to_words_str(readers[pos](filepaths[pos]))
                    )
                else:
                    rst_list.append(
                        readers[pos](filepaths[pos])
                    )
            except Exception as e:
                print(e)
                rst_list.append(
                    None
                )
                # also append error list
                err_list.append(
                    filepaths[pos]
                )

        return rst_list, err_list

    # precheck if the data is suitable
    assert set(file_mime_list).issubset(set(func_map.keys())), \
        'the file_mime_list contains file mime type which could not be read'
    assert len(filepath_list) == len(file_mime_list), \
        'The path of filepath and its mime must be same'
    assert not (np.any(pd.isna(filepath_list)) or np.any(pd.isna(file_mime_list))), \
        'The filepath-list/mime should not contains any Null'

    tasks_chunks = [
        (filepath_list[s: s + math.ceil(len(filepath_list) / processes)],
         file_mime_list[s: s + math.ceil(len(file_mime_list) / processes)]
         )
        for s in range(0, len(filepath_list), math.ceil(len(filepath_list) / processes))
    ]

    text_list = []
    error_list = []

    with pathos.multiprocessing.Pool(
            processes=processes, initializer=_BasicKits._BasicFuncT.processes_interrupt_initiator
    ) as pool:
        for rsterr in pool.imap(_lambda_reader_loop, tasks_chunks):
            text_list.extend(rsterr[0])
            if rsterr[1]:  # [] -> None extend if not contain anything(list)
                error_list.extend(rsterr[1])

    return text_list, error_list
