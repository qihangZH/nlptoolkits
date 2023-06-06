import global_options
import seminlpscorer
import datetime
from pathlib import Path

if __name__ == '__main__':

    # # train the word2vec model ----------------
    print(datetime.datetime.now())
    print("Training w2v model...")
    seminlpscorer.Wrd2vScorerT.l1_train_w2v_model(
        path_input_cleaned_txt=Path(
            global_options.PROCESSED_DATA_FOLDER, "trigram", "documents.txt"
        ),
        path_output_model=Path(global_options.MODEL_FOLDER, "w2v", "w2v.mod"),
        size=global_options.W2V_DIM,
        window=global_options.W2V_WINDOW,
        workers=global_options.N_CORES,
        iter=global_options.W2V_ITER,
    )

    result_dict = seminlpscorer.Wrd2vScorerT.l1_semi_supervise_w2v_dict(
        path_input_w2v_model=str(Path(global_options.MODEL_FOLDER, "w2v", "w2v.mod")),
        seed_words_dict=global_options.SEED_WORDS,
        restrict_vocab_per=global_options.DICT_RESTRICT_VOCAB,
        model_dims=global_options.N_WORDS_DIM
    )

    # output the dictionary
    seminlpscorer._BasicT.write_dict_to_csv(
        culture_dict=result_dict,
        file_name=str(Path(global_options.OUTPUT_FOLDER, "dict", "expanded_dict.csv")),
    )
    print(f'Dictionary saved at {str(Path(global_options.OUTPUT_FOLDER, "dict", "expanded_dict.csv"))}')
