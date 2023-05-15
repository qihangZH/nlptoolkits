from pathlib import Path
import global_options
import seminlpclassify

if __name__ == "__main__":
    """Arguments"""
    seminlpclassify.l1_auto_parser(
        endpoint=global_options.ADDRESS_CORENLP,
        memory=global_options.RAM_CORENLP,
        nlp_threads=global_options.N_CORES,
        path_input_txt=Path(global_options.DATA_FOLDER, "input", "documents.txt"),
        input_index_list=seminlpclassify.file_util.file_to_list(
            Path(global_options.DATA_FOLDER, "input", "document_ids.txt")
        ),
        path_output_txt=Path(
            global_options.DATA_FOLDER, "processed", "parsed", "documents.txt"
        ),
        path_output_index=Path(
            global_options.DATA_FOLDER, "processed", "parsed", "document_sent_ids.txt",
        ),
        use_multicores=True
    )