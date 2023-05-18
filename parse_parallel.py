from pathlib import Path
import global_options
import seminlpclassify

if __name__ == "__main__":
    """Arguments"""
    seminlpclassify.l1_auto_parser(
        endpoint=global_options.ADDRESS_CORENLP,
        memory=global_options.RAM_CORENLP,
        nlp_threads=global_options.N_CORES,
        path_input_txt=Path(global_options.INPUT_DATA_FOLDER, "tweets_origin.txt"),
        input_index_list=seminlpclassify._file_util.file_to_list(
            Path(global_options.INPUT_DATA_FOLDER, "id_origin.txt")
        ),
        path_output_txt=Path(
            global_options.PROCESSED_DATA_FOLDER, "parsed", "documents.txt"
        ),
        path_output_index=Path(
            global_options.PROCESSED_DATA_FOLDER, "parsed", "document_sent_ids.txt",
        ),
        use_multicores=True,
        chunk_size=global_options.PARSE_CHUNK_SIZE
    )
