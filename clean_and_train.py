import os
from pathlib import Path
import global_options
import seminlpscorer

# logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


if __name__ == '__main__':
    # check make directory if exist.

    # clean the parsed text (remove POS tags, stopwords, etc.) ----------------
    seminlpscorer.auto_clean_parsed_txt(
        path_in_parsed_txt=Path(global_options.PROCESSED_DATA_FOLDER, "parsed", "documents.txt"),
        path_out_cleaned_txt=Path(global_options.PROCESSED_DATA_FOLDER, "unigram", "documents.txt"),
        stopwords=global_options.STOPWORDS,
        processes=os.cpu_count()
    )

    # train and apply a phrase model to detect 2-word phrases ----------------
    seminlpscorer.auto_bigram_fit_transform_txt(
        path_input_clean_txt=Path(
            global_options.PROCESSED_DATA_FOLDER, "unigram", "documents.txt"
        ),
        path_output_transformed_txt=Path(
            global_options.PROCESSED_DATA_FOLDER, "bigram", "documents.txt"
        ),
        path_output_model_mod=Path(global_options.MODEL_FOLDER, "phrases", "bigram.mod"),
        phrase_min_length=global_options.PHRASE_MIN_COUNT,
        stopwords_set=global_options.STOPWORDS,
        threshold=global_options.PHRASE_THRESHOLD,
        scoring="original_scorer"
    )

    # train and apply a phrase model to detect 3-word phrases ----------------

    seminlpscorer.auto_bigram_fit_transform_txt(
        path_input_clean_txt=Path(global_options.PROCESSED_DATA_FOLDER, "bigram", "documents.txt"),
        path_output_transformed_txt=Path(
            global_options.PROCESSED_DATA_FOLDER, "trigram", "documents.txt"
        ),
        path_output_model_mod=Path(global_options.MODEL_FOLDER, "phrases", "trigram.mod"),
        phrase_min_length=global_options.PHRASE_MIN_COUNT,
        stopwords_set=global_options.STOPWORDS,
        threshold=global_options.PHRASE_THRESHOLD,
        scoring="original_scorer"
    )
