# %%
from pathlib import Path
import global_options
import seminlpscorer
import pandas as pd
import os

# %%
if __name__ == "__main__":

    PATH_FILE = os.path.abspath(os.path.dirname(__file__)) + '/'
    PATH_TEXT = PATH_FILE + 'input_data/tweets_origin.txt'
    PATH_INDEX = PATH_FILE + 'input_data/ids_origin.txt'

    with open(PATH_TEXT, 'r',encoding='utf-8') as f:
        lines = f.read().splitlines()

    max_text_len = pd.Series(lines).apply(len).max()
    
    if max_text_len > 1000000:
        raise ValueError(
            f'max number of input, {max_text_len} is bigger than limitation 1000000'
        )

    """Arguments"""
    seminlpscorer.l1_auto_parser(
        endpoint=global_options.ADDRESS_CORENLP,
        memory=global_options.RAM_CORENLP,
        nlp_threads=global_options.N_CORES,
        path_input_txt=Path(global_options.INPUT_DATA_FOLDER, "tweets_origin.txt"),
        input_index_list=seminlpscorer._file_util.file_to_list(
            Path(global_options.INPUT_DATA_FOLDER, "ids_origin.txt")
        ),
        path_output_txt=Path(
            global_options.PROCESSED_DATA_FOLDER, "parsed", "documents.txt"
        ),
        path_output_index_txt=Path(
            global_options.PROCESSED_DATA_FOLDER, "parsed", "document_sent_ids.txt",
        ),
        use_multicores=True,
        chunk_size=global_options.PARSE_CHUNK_SIZE
    )
