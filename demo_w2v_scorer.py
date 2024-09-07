"""initialize the project and insure the project can run"""
import __glob_opts
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
    for outputdir in [__glob_opts.PROCESSED_DATA_FOLDER,
                      __glob_opts.MODEL_FOLDER,
                      __glob_opts.OUTPUT_FOLDER

                      ]:
        nlptoolkits.delete_whole_dir(directory=outputdir)

    """root level dir make"""
    Path(__glob_opts.PROCESSED_DATA_FOLDER).mkdir(parents=False, exist_ok=True)
    Path(__glob_opts.MODEL_FOLDER).mkdir(parents=False, exist_ok=True)
    Path(__glob_opts.OUTPUT_FOLDER).mkdir(parents=False, exist_ok=True)

    """model dir make"""
    Path(__glob_opts.MODEL_FOLDER, "phrases").mkdir(parents=False, exist_ok=True)
    Path(__glob_opts.MODEL_FOLDER, "w2v").mkdir(parents=False, exist_ok=True)

    """processed data dir make"""
    Path(__glob_opts.PROCESSED_DATA_FOLDER, "parsed").mkdir(parents=False, exist_ok=True)
    Path(__glob_opts.PROCESSED_DATA_FOLDER, "unigram").mkdir(parents=False, exist_ok=True)
    Path(__glob_opts.PROCESSED_DATA_FOLDER, "bigram").mkdir(parents=False, exist_ok=True)
    Path(__glob_opts.PROCESSED_DATA_FOLDER, "trigram").mkdir(parents=False, exist_ok=True)

    """output result dir make"""
    Path(__glob_opts.OUTPUT_FOLDER, "dict").mkdir(parents=False, exist_ok=True)
    Path(__glob_opts.OUTPUT_FOLDER, "scores").mkdir(parents=False, exist_ok=True)
    Path(__glob_opts.OUTPUT_FOLDER, "scores", "temp").mkdir(parents=False, exist_ok=True)
    Path(__glob_opts.OUTPUT_FOLDER, "scores", "word_contributions").mkdir(parents=False, exist_ok=True)

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
    nlptoolkits.StanzaKits.CoreNLPServerPack.auto_doc_to_sentences_annotator(
        endpoint=__glob_opts.ADDRESS_CORENLP,
        memory=__glob_opts.RAM_CORENLP,
        processes=__glob_opts.N_CORES,
        path_input_txt=Path(__glob_opts.INPUT_DATA_FOLDER, "tweets_origin.txt"),
        input_index_list=nlptoolkits.SmallKits.IOHandlerT.file_to_list(
            Path(__glob_opts.INPUT_DATA_FOLDER, "ids_origin.txt"),
            charset_error_encoding=__glob_opts.DEFAULT_ENCODING
        ),
        path_output_txt=Path(
            __glob_opts.PROCESSED_DATA_FOLDER, "parsed", "documents.txt"
        ),
        path_output_index_txt=Path(
            __glob_opts.PROCESSED_DATA_FOLDER, "parsed", "document_sent_ids.txt",
        ),
        chunk_size=__glob_opts.PARSE_CHUNK_SIZE,
        be_quite=True,
        # annotation_choices=['Lemmatize', 'NERtags', 'DepParseMWECompounds', 'POStags', 'SentenceSentiment'],
        # properties={
        #     "ner.applyFineGrained": "false",
        #     "annotators": "tokenize, ssplit, pos, lemma, ner, depparse, sentiment",
        # }
        annotation_choices=['Lemmatize', 'NERtags', 'DepParseMWECompounds', 'POStags'],
        properties={
            "ner.applyFineGrained": "false",
            "annotators": "tokenize, ssplit, pos, lemma, ner, depparse",
        }
    )

    # --------------------------------------------------------------------------------------------------
    # clean and train
    # --------------------------------------------------------------------------------------------------

    # clean the parsed text (remove POS tags, stopwords, etc.) ----------------
    nlptoolkits.StanzaKits.CoreNLPServerPack.auto_clean_annotated_txt(
        path_in_parsed_txt=Path(__glob_opts.PROCESSED_DATA_FOLDER, "parsed", "documents.txt"),
        path_out_cleaned_txt=Path(__glob_opts.PROCESSED_DATA_FOLDER, "unigram", "documents.txt"),
        remove_stopwords_set=__glob_opts.STOPWORDS,
        token_remove_ner_tags_to_lessequal_then_num=1,
        processes=__glob_opts.N_CORES,
        clean_flag=0
    )

    # train and apply a phrase model to detect 2-word phrases ----------------
    nlptoolkits.GensimKits.BigramT.sentence_bigram_fit_transform_txt(
        path_input_clean_txt=Path(
            __glob_opts.PROCESSED_DATA_FOLDER, "unigram", "documents.txt"
        ),
        path_output_transformed_txt=Path(
            __glob_opts.PROCESSED_DATA_FOLDER, "bigram", "documents.txt"
        ),
        path_output_model_mod=Path(__glob_opts.MODEL_FOLDER, "phrases", "bigram.mod"),
        phrase_min_length=__glob_opts.PHRASE_MIN_COUNT,
        connection_words=__glob_opts.STOPWORDS,
        processes=1,
        chunk_size=200000,
        start_iloc=None,
        threshold=__glob_opts.PHRASE_THRESHOLD,
        scoring="original_scorer"
    )

    # --------------------------------------------------------------------------------------------------
    # train and apply a phrase model to detect 3-word phrases
    # --------------------------------------------------------------------------------------------------

    nlptoolkits.GensimKits.BigramT.sentence_bigram_fit_transform_txt(
        path_input_clean_txt=Path(__glob_opts.PROCESSED_DATA_FOLDER, "bigram", "documents.txt"),
        path_output_transformed_txt=Path(
            __glob_opts.PROCESSED_DATA_FOLDER, "trigram", "documents.txt"
        ),
        path_output_model_mod=Path(__glob_opts.MODEL_FOLDER, "phrases", "trigram.mod"),
        phrase_min_length=__glob_opts.PHRASE_MIN_COUNT,
        connection_words=__glob_opts.STOPWORDS,
        processes=1,
        chunk_size=200000,
        start_iloc=None,
        threshold=__glob_opts.PHRASE_THRESHOLD,
        scoring="original_scorer"
    )

    # --------------------------------------------------------------------------------------------------
    # # train the word2vec model
    # --------------------------------------------------------------------------------------------------

    '''
    NOTICE, if seed and hashfxn do not set in train_w2v_model, and if workers do not set to 1
    then the RESULT COULD BE UNSTABLE!
    seed should be set as a number like 42 while hashfxn can be set by _Model.gen_gensim_hash
    IGNORE it if you expect random result or your corpus is huge enough.
    see here for problem:
    https://stackoverflow.com/questions/34831551/ensure-the-gensim-generate-the-same-word2vec-model-for-different-runs-on-the-sam
    '''

    print(datetime.datetime.now())
    print("Training w2v model...")
    nlptoolkits.GensimKits._Models.train_w2v_model(
        path_input_sentence_txt=Path(
            __glob_opts.PROCESSED_DATA_FOLDER, "trigram", "documents.txt"
        ),
        path_output_model=Path(__glob_opts.MODEL_FOLDER, "w2v", "w2v.mod"),
        vector_size=__glob_opts.W2V_DIM,
        window=__glob_opts.W2V_WINDOW,
        workers=1,
        epochs=__glob_opts.W2V_ITER,
        seed=42,
        hashfxn=nlptoolkits.GensimKits._Models.gen_gensim_hash
    )

    result_dict = nlptoolkits.GensimKits.Wrd2vScorerT.l1_semi_supervise_w2v_dict(
        path_input_w2v_model=str(Path(__glob_opts.MODEL_FOLDER, "w2v", "w2v.mod")),
        seed_words_dict=__glob_opts.SEED_WORDS,
        restrict_vocab_per=__glob_opts.DICT_RESTRICT_VOCAB,
        model_dims=__glob_opts.N_WORDS_DIM
    )

    # --------------------------------------------------------------------------------------------------
    # scorer
    # --------------------------------------------------------------------------------------------------

    # output the dictionary
    nlptoolkits._BasicKits.FileT.write_dict_to_csv(
        culture_dict=result_dict,
        file_name=str(Path(__glob_opts.OUTPUT_FOLDER, "dict", "expanded_dict.csv")),
    )
    print(f'Dictionary saved at {str(Path(__glob_opts.OUTPUT_FOLDER, "dict", "expanded_dict.csv"))}')

    scorer_class = nlptoolkits.GensimKits.Wrd2vScorerT.DocScorer(
        path_current_dict=pathlib.Path(__glob_opts.OUTPUT_FOLDER, "dict", "expanded_dict.csv"),
        path_trainw2v_sentences_dataset_txt=pathlib.Path(
            __glob_opts.PROCESSED_DATA_FOLDER, "trigram", "documents.txt"
        ),
        path_trainw2v_sentences_dataset_index_txt=pathlib.Path(
            __glob_opts.PROCESSED_DATA_FOLDER, "parsed", "document_sent_ids.txt"
        ),
        charset_error_encoding=__glob_opts.DEFAULT_ENCODING
    )

    """however, you do not need this part"""
    scorer_class.pickle_doc_freq(
        pathlib.Path(__glob_opts.OUTPUT_FOLDER, "scores", "temp", "doc_freq.pickle")
    )

    scorer_class.pickle_doc_level_ids(
        pathlib.Path(__glob_opts.OUTPUT_FOLDER, "scores", "temp", "doc_ids.pickle")
    )

    scorer_class.pickle_doc_level_corpus(
        pathlib.Path(__glob_opts.OUTPUT_FOLDER, "scores", "temp", "corpus_doc_level.pickle")
    )

    scorer_class.score_tf_df().to_csv(
        pathlib.Path(__glob_opts.OUTPUT_FOLDER, "scores", "scores_TF.csv"), index=False
    )
    methods = ["TFIDF", "WFIDF"]
    for method in methods:
        score, contribution = scorer_class.score_tfidf_dfdf(
            method=method,
            normalize=False,
            vague=False
        )

        score.to_csv(
            str(
                pathlib.Path(
                    __glob_opts.OUTPUT_FOLDER,
                    "scores",
                    "scores_{}.csv".format(method),
                )
            ),
            index=False,
        )
        # save word contributions
        contribution.to_csv(
            pathlib.Path(
                __glob_opts.OUTPUT_FOLDER,
                "scores",
                "word_contributions",
                "word_contribution_{}.csv".format(method),
            )
        )
