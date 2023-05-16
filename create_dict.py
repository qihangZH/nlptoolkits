import global_options
import seminlpclassify
from pathlib import Path

if __name__ == '__main__':
    result_dict = seminlpclassify.auto_w2v_semisup_dict(
        path_input_w2v_model=str(Path(global_options.MODEL_FOLDER, "w2v", "w2v.mod")),
        seed_words_dict=global_options.SEED_WORDS,
        restrict_vocab_per=global_options.DICT_RESTRICT_VOCAB,
        model_dims=global_options.N_WORDS_DIM
    )

    # output the dictionary
    seminlpclassify._dictionary.write_dict_to_csv(
        culture_dict=result_dict,
        file_name=str(Path(global_options.OUTPUT_FOLDER, "dict", "expanded_dict.csv")),
    )
    print(f'Dictionary saved at {str(Path(global_options.OUTPUT_FOLDER, "dict", "expanded_dict.csv"))}')
