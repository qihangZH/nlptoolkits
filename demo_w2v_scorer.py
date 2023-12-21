"""initialize the project and insure the project can run"""
import global_options
from pathlib import Path
import os
import pandas as pd
import datetime
import pathlib
import nlptoolkits

if __name__ == '__main__':

    # --------------------------------------------------------------------------------------------------
    # initializer
    # --------------------------------------------------------------------------------------------------

    print("""REFRESHING THE PROJECT FOLDERS,PLEASE WAIT...""")
    # delete the directory if they already exist
    for outputdir in [global_options.PROCESSED_DATA_FOLDER,
                      global_options.MODEL_FOLDER,
                      global_options.OUTPUT_FOLDER

                      ]:
        nlptoolkits.delete_whole_dir(directory=outputdir)

    """root level dir make"""
    Path(global_options.PROCESSED_DATA_FOLDER).mkdir(parents=False, exist_ok=True)
    Path(global_options.MODEL_FOLDER).mkdir(parents=False, exist_ok=True)
    Path(global_options.OUTPUT_FOLDER).mkdir(parents=False, exist_ok=True)

    """model dir make"""
    Path(global_options.MODEL_FOLDER, "phrases").mkdir(parents=False, exist_ok=True)
    Path(global_options.MODEL_FOLDER, "w2v").mkdir(parents=False, exist_ok=True)

    """processed data dir make"""
    Path(global_options.PROCESSED_DATA_FOLDER, "parsed").mkdir(parents=False, exist_ok=True)
    Path(global_options.PROCESSED_DATA_FOLDER, "unigram").mkdir(parents=False, exist_ok=True)
    Path(global_options.PROCESSED_DATA_FOLDER, "bigram").mkdir(parents=False, exist_ok=True)
    Path(global_options.PROCESSED_DATA_FOLDER, "trigram").mkdir(parents=False, exist_ok=True)

    """output result dir make"""
    Path(global_options.OUTPUT_FOLDER, "dict").mkdir(parents=False, exist_ok=True)
    Path(global_options.OUTPUT_FOLDER, "scores").mkdir(parents=False, exist_ok=True)
    Path(global_options.OUTPUT_FOLDER, "scores", "temp").mkdir(parents=False, exist_ok=True)
    Path(global_options.OUTPUT_FOLDER, "scores", "word_contributions").mkdir(parents=False, exist_ok=True)

    print("""...DONE""")

    # --------------------------------------------------------------------------------------------------
    # parse parallel
    # --------------------------------------------------------------------------------------------------

    PATH_FILE = os.path.abspath(os.path.dirname(__file__)) + '/'
    PATH_TEXT = PATH_FILE + 'input_data/tweets_origin.txt'
    PATH_INDEX = PATH_FILE + 'input_data/ids_origin.txt'

    with open(PATH_TEXT, 'r', encoding='utf-8') as f:
        lines = f.read().splitlines()

    max_text_len = pd.Series(lines).apply(len).max()

    if max_text_len > 1000000:
        raise ValueError(
            f'max number of input, {max_text_len} is bigger than limitation 1000000'
        )

    """Arguments"""
    nlptoolkits.StanzaKits.CoreNLPServerPack.auto_doc_to_sentences_parser(
        endpoint=global_options.ADDRESS_CORENLP,
        memory=global_options.RAM_CORENLP,
        processes=global_options.N_CORES,
        path_input_txt=Path(global_options.INPUT_DATA_FOLDER, "tweets_origin.txt"),
        input_index_list=nlptoolkits.SmallKits.IOHandlerT.file_to_list(
            Path(global_options.INPUT_DATA_FOLDER, "ids_origin.txt")
        ),
        path_output_txt=Path(
            global_options.PROCESSED_DATA_FOLDER, "parsed", "documents.txt"
        ),
        path_output_index_txt=Path(
            global_options.PROCESSED_DATA_FOLDER, "parsed", "document_sent_ids.txt",
        ),
        chunk_size=global_options.PARSE_CHUNK_SIZE,
        be_quite=True,
        parsing_choices=['Lemmatize', 'POStags', 'NERtags', 'DepParseMWECompounds']
    )

    # --------------------------------------------------------------------------------------------------
    # clean and train
    # --------------------------------------------------------------------------------------------------

    # # clean the parsed text (remove POS tags, stopwords, etc.) ----------------
    # nlptoolkits.StanzaKits.CoreNLPServerPack.auto_clean_parsed_txt(
    #     path_in_parsed_txt=Path(global_options.PROCESSED_DATA_FOLDER, "parsed", "documents.txt"),
    #     path_out_cleaned_txt=Path(global_options.PROCESSED_DATA_FOLDER, "unigram", "documents.txt"),
    #     stopwords_set=global_options.STOPWORDS,
    #     processes=os.cpu_count()
    # )
    #
    # # train and apply a phrase model to detect 2-word phrases ----------------
    # nlptoolkits.GensimKits.BigramT.sentence_bigram_fit_transform_txt(
    #     path_input_clean_txt=Path(
    #         global_options.PROCESSED_DATA_FOLDER, "unigram", "documents.txt"
    #     ),
    #     path_output_transformed_txt=Path(
    #         global_options.PROCESSED_DATA_FOLDER, "bigram", "documents.txt"
    #     ),
    #     path_output_model_mod=Path(global_options.MODEL_FOLDER, "phrases", "bigram.mod"),
    #     phrase_min_length=global_options.PHRASE_MIN_COUNT,
    #     stopwords_set=global_options.STOPWORDS,
    #     threshold=global_options.PHRASE_THRESHOLD,
    #     scoring="original_scorer"
    # )
    #
    # # --------------------------------------------------------------------------------------------------
    # # train and apply a phrase model to detect 3-word phrases
    # # --------------------------------------------------------------------------------------------------
    #
    # nlptoolkits.GensimKits.BigramT.sentence_bigram_fit_transform_txt(
    #     path_input_clean_txt=Path(global_options.PROCESSED_DATA_FOLDER, "bigram", "documents.txt"),
    #     path_output_transformed_txt=Path(
    #         global_options.PROCESSED_DATA_FOLDER, "trigram", "documents.txt"
    #     ),
    #     path_output_model_mod=Path(global_options.MODEL_FOLDER, "phrases", "trigram.mod"),
    #     phrase_min_length=global_options.PHRASE_MIN_COUNT,
    #     stopwords_set=global_options.STOPWORDS,
    #     threshold=global_options.PHRASE_THRESHOLD,
    #     scoring="original_scorer"
    # )
    #
    # # --------------------------------------------------------------------------------------------------
    # # # train the word2vec model
    # # --------------------------------------------------------------------------------------------------
    # print(datetime.datetime.now())
    # print("Training w2v model...")
    # nlptoolkits.GensimKits._Models.train_w2v_model(
    #     path_input_sentence_txt=Path(
    #         global_options.PROCESSED_DATA_FOLDER, "trigram", "documents.txt"
    #     ),
    #     path_output_model=Path(global_options.MODEL_FOLDER, "w2v", "w2v.mod"),
    #     size=global_options.W2V_DIM,
    #     window=global_options.W2V_WINDOW,
    #     workers=global_options.N_CORES,
    #     iter=global_options.W2V_ITER,
    # )
    #
    # result_dict = nlptoolkits.GensimKits.Wrd2vScorerT.l1_semi_supervise_w2v_dict(
    #     path_input_w2v_model=str(Path(global_options.MODEL_FOLDER, "w2v", "w2v.mod")),
    #     seed_words_dict=global_options.SEED_WORDS,
    #     restrict_vocab_per=global_options.DICT_RESTRICT_VOCAB,
    #     model_dims=global_options.N_WORDS_DIM
    # )
    #
    # # --------------------------------------------------------------------------------------------------
    # # scorer
    # # --------------------------------------------------------------------------------------------------
    #
    # # output the dictionary
    # nlptoolkits._BasicKits.FileT.write_dict_to_csv(
    #     culture_dict=result_dict,
    #     file_name=str(Path(global_options.OUTPUT_FOLDER, "dict", "expanded_dict.csv")),
    # )
    # print(f'Dictionary saved at {str(Path(global_options.OUTPUT_FOLDER, "dict", "expanded_dict.csv"))}')
    #
    # scorer_class = nlptoolkits.GensimKits.Wrd2vScorerT.DocScorer(
    #     path_current_dict=pathlib.Path(global_options.OUTPUT_FOLDER, "dict", "expanded_dict.csv"),
    #     path_trainw2v_sentences_dataset_txt=pathlib.Path(
    #         global_options.PROCESSED_DATA_FOLDER, "trigram", "documents.txt"
    #     ),
    #     path_trainw2v_sentences_dataset_index_txt=pathlib.Path(
    #         global_options.PROCESSED_DATA_FOLDER, "parsed", "document_sent_ids.txt"
    #     ),
    #     processes=global_options.N_CORES
    # )
    #
    # """however, you do not need this part"""
    # scorer_class.pickle_doc_freq(
    #     pathlib.Path(global_options.OUTPUT_FOLDER, "scores", "temp", "doc_freq.pickle")
    # )
    #
    # scorer_class.pickle_doc_level_ids(
    #     pathlib.Path(global_options.OUTPUT_FOLDER, "scores", "temp", "doc_ids.pickle")
    # )
    #
    # scorer_class.pickle_doc_level_corpus(
    #     pathlib.Path(global_options.OUTPUT_FOLDER, "scores", "temp", "corpus_doc_level.pickle")
    # )
    #
    # scorer_class.score_tf_df().to_csv(
    #     pathlib.Path(global_options.OUTPUT_FOLDER, "scores", "scores_TF.csv"), index=False
    # )
    # methods = ["TFIDF", "WFIDF"]
    # for method in methods:
    #     score, contribution = scorer_class.score_tfidf_dfdf(
    #         method=method,
    #         normalize=False,
    #     )
    #
    #     score.to_csv(
    #         str(
    #             pathlib.Path(
    #                 global_options.OUTPUT_FOLDER,
    #                 "scores",
    #                 "scores_{}.csv".format(method),
    #             )
    #         ),
    #         index=False,
    #     )
    #     # save word contributions
    #     contribution.to_csv(
    #         pathlib.Path(
    #             global_options.OUTPUT_FOLDER,
    #             "scores",
    #             "word_contributions",
    #             "word_contribution_{}.csv".format(method),
    #         )
    #     )
