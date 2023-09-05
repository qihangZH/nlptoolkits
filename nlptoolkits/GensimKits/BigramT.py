import gensim
import tqdm
from . import _Models
from .. import _BasicKits

# --------------------------------------------------------------------------
# l0 level functions
# --------------------------------------------------------------------------

"""bigram models to make seperate words to one word concat with _"""


def _sentence_file_bigram(input_path, output_path, model_path, threshold=None, scoring=None):
    """ Transform an input_data text file into a file with 2-word phrases.
    Apply again to learn 3-word phrases.

    Arguments:
        input_path {str}: Each line is a sentence
        ouput_file {str}: Each line is a sentence with 2-word phraes concatenated
    """

    # Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    # Path(model_path).parent.mkdir(parents=True, exist_ok=True)

    def _lambda_bigram_transform(line, bigram_phraser):
        """ Helper file fore file_bigramer
        Note: Needs a phraser object or phrase model.

        Arguments:
            line {str}: a line

        return: a line with phrases joined using "_"
        """
        return " ".join(bigram_phraser[line.split()])

    bigram_model = gensim.models.phrases.Phrases.load(str(model_path))
    if scoring is not None:
        bigram_model.scoring = getattr(gensim.models.phrases, scoring)
    if threshold is not None:
        bigram_model.threshold = threshold
    # bigram_phraser = models.phrases.Phraser(bigram_model)
    with open(input_path, "r", encoding='utf-8') as f:
        input_data = f.readlines()
    data_bigram = [_lambda_bigram_transform(l, bigram_model) for l in tqdm.tqdm(input_data)]
    with open(output_path, "w", encoding='utf-8') as f:
        f.write("\n".join(data_bigram) + "\n")
    assert len(input_data) == _BasicKits.FileT._line_counter(output_path)


# --------------------------------------------------------------------------
# l1 level functions
# --------------------------------------------------------------------------
"""Preprocessing: train and transform the bigram model, concat two words into one"""


def sentence_bigram_fit_transform_txt(path_input_clean_txt,
                                      path_output_transformed_txt,
                                      path_output_model_mod,
                                      phrase_min_length: int,
                                      stopwords_set,
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
        stopwords_set:
        threshold:
        scoring:

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
        stopwords_set=stopwords_set
    )
    _sentence_file_bigram(
        input_path=path_input_clean_txt,
        output_path=path_output_transformed_txt,
        model_path=path_output_model_mod,
        scoring=scoring,
        threshold=threshold,
    )
