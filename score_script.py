import global_options
import pathlib
from seminlpclassify.scorer import DocScorer

if __name__ == "__main__":
    scorer_class = DocScorer(
        path_current_dict=pathlib.Path(global_options.OUTPUT_FOLDER, "dict", "expanded_dict.csv"),
        path_trainw2v_dataset_txt=pathlib.Path(
            global_options.PROCESSED_DATA_FOLDER, "trigram", "documents.txt"
        ),
        path_trainw2v_dataset_index_txt=pathlib.Path(
            global_options.PROCESSED_DATA_FOLDER, "parsed", "document_sent_ids.txt"
        ),
        mp_threads=global_options.N_CORES
    )

    """however, you do not need this part"""
    scorer_class.pickle_doc_freq(
        pathlib.Path(global_options.OUTPUT_FOLDER, "scores", "temp", "doc_freq.pickle")
    )

    scorer_class.pickle_doc_level_ids(
        pathlib.Path(global_options.OUTPUT_FOLDER, "scores", "temp", "doc_ids.pickle")
    )

    scorer_class.pickle_doc_level_corpus(
        pathlib.Path(global_options.OUTPUT_FOLDER, "scores", "temp", "corpus_doc_level.pickle")
    )

    scorer_class.score_tf_df().to_csv(
        pathlib.Path(global_options.OUTPUT_FOLDER, "scores", "scores_TF.csv"), index=False
    )
    methods = ["TFIDF", "WFIDF"]
    for method in methods:
        score, contribution = scorer_class.score_tfidf_tupledf(
            method=method,
            normalize=False,
        )

        score.to_csv(
            str(
                pathlib.Path(
                    global_options.OUTPUT_FOLDER,
                    "scores",
                    "scores_{}.csv".format(method),
                )
            ),
            index=False,
        )
        # save word contributions
        contribution.to_csv(
            pathlib.Path(
                global_options.OUTPUT_FOLDER,
                "scores",
                "word_contributions",
                "word_contribution_{}.csv".format(method),
            )
        )
