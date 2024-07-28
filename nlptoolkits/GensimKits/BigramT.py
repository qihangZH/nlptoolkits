import typing

import gensim
import tqdm
from . import _Models
from .. import _BasicKits

# --------------------------------------------------------------------------
# l0 level functions
# --------------------------------------------------------------------------

"""bigram models to make seperate words to one word concat with _"""


def _sentence_file_bigram(input_path,
                          output_path,
                          model_path,
                          processes,
                          chunk_size,
                          start_iloc: typing.Optional[int],
                          threshold=None,
                          scoring=None
                          ):
    """ Transform an input_data text file into a file with 2-word phrases.
    Apply again to learn 3-word phrases.

    Arguments:
        input_path {str}: Each line is a sentence
        ouput_file {str}: Each line is a sentence with 2-word phraes concatenated
        model_path: the path to use the model
        processes: how much processes to deal with data
        chunk_size: {int} -- number of lines to process each time, increasing the default may increase performance
        start_iloc: {int} -- line number to start from (index starts with 0)
    """

    # Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    # Path(model_path).parent.mkdir(parents=True, exist_ok=True)

    def _lambda_bigram_transform(line, index, bigram_phraser):
        """ Helper file fore file_bigramer
        Note: Needs a phraser object or phrase model.

        Arguments:
            line {str}: a line

        return: a line with phrases joined using "_"
        """
        # return " ".join(bigram_phraser[line.split()])
        return " ".join(bigram_phraser[line.split()]), index

    bigram_model = gensim.models.phrases.Phrases.load(str(model_path))
    if scoring is not None:
        bigram_model.scoring = getattr(gensim.models.phrases, scoring)
    if threshold is not None:
        bigram_model.threshold = threshold

    """old version do not fit low memory using"""
    # # bigram_phraser = models.phrases.Phraser(bigram_model)
    # with open(input_path, "r", encoding='utf-8') as f:
    #     input_data = f.readlines()
    # data_bigram = [_lambda_bigram_transform(l, None, bigram_model) for l in tqdm.tqdm(input_data)]
    # with open(output_path, "w", encoding='utf-8') as f:
    #     f.write("\n".join(data_bigram) + "\n")
    # assert len(input_data) == _BasicKits.FileT._line_counter(output_path)

    if processes > 1:
        _BasicKits.FileT.l1_mp_process_largefile(
            path_input_txt=input_path,
            path_output_txt=output_path,
            input_index_list=[
                str(i) for i in range(_BasicKits.FileT._line_counter(input_path))
            ],  # fake IDs (do not need IDs for this function).
            path_output_index_txt=None,
            process_line_func=lambda line, i: _lambda_bigram_transform(line, i, bigram_model),
            processes=processes,
            chunk_size=chunk_size,
            start_iloc=start_iloc
        )

    else:

        _BasicKits.FileT.l1_process_largefile(
            path_input_txt=input_path,
            path_output_txt=output_path,
            input_index_list=[
                str(i) for i in range(_BasicKits.FileT._line_counter(input_path))
            ],  # fake IDs (do not need IDs for this function).
            path_output_index_txt=None,
            process_line_func=lambda line, i: _lambda_bigram_transform(line, i, bigram_model),
            chunk_size=chunk_size,
            start_iloc=start_iloc
        )


# --------------------------------------------------------------------------
# l1 level functions
# --------------------------------------------------------------------------
"""Preprocessing: train and transform the bigram model, concat two words into one"""


def sentence_bigram_fit_transform_txt(path_input_clean_txt,
                                      path_output_transformed_txt,
                                      path_output_model_mod,
                                      phrase_min_length: int,
                                      connection_words,
                                      processes: int,
                                      chunk_size: int,
                                      start_iloc: typing.Optional[int],
                                      threshold=None,
                                      scoring="original_scorer"
                                      ):
    """
    transform the sep two length words to concat in a word which join by '_'
    which means uni-gram -> bi-gram words.
    you can recursive this function to get the target tri-gram or bigger phrases.

    Args:
        path_input_clean_txt:
        path_output_transformed_txt:
        path_output_model_mod:
        phrase_min_length:
        connection_words: stop words
        processes: how much processes to deal with data
        threshold:
        scoring:
        chunk_size: {int} -- number of lines to process each time, increasing the default may increase performance
        start_iloc: {int} -- line number to start from (index starts with 0)

    Returns:

    """

    # precheck
    if not str(path_output_model_mod).endswith('.mod'):
        raise ValueError('Model must end with .mod')

    # train and apply a phrase model to detect 3-word phrases ----------------
    _Models.train_sentence_bigram_model(
        input_path=path_input_clean_txt,
        model_path=path_output_model_mod,
        phrase_min_length=phrase_min_length,
        phrase_threshold=threshold,
        connector_words=connection_words
    )
    _sentence_file_bigram(
        input_path=path_input_clean_txt,
        output_path=path_output_transformed_txt,
        model_path=path_output_model_mod,
        processes=processes,
        chunk_size=chunk_size,
        start_iloc=start_iloc,
        scoring=scoring,
        threshold=threshold,
    )
