from . import _dictionary
from . import nlp_models
from . import _file_util
from . import preprocess
from . import scorer
from . import qihangfuncs
# out source pack
import os
import gensim
import typing
import shutil

"""Other functions"""


def delete_whole_dir(directory):
    """delete the whole dir..."""
    if os.path.exists(directory) and os.path.isdir(directory):
        shutil.rmtree(directory)


def alias_file_to_list(a_file):
    return _file_util.file_to_list(a_file=a_file)


"""train and transform the model"""


def auto_bigram_fit_transform_txt(path_input_clean_txt,
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
    nlp_models.train_bigram_model(
        input_path=path_input_clean_txt,
        model_path=path_output_model_mod,
        phrase_min_length=phrase_min_length,
        phrase_threshold=threshold,
        stopwords_set=stopwords_set
    )
    nlp_models.file_bigramer(
        input_path=path_input_clean_txt,
        output_path=path_output_transformed_txt,
        model_path=path_output_model_mod,
        scoring=scoring,
        threshold=threshold,
    )


"""word2vec model function"""


@qihangfuncs.timer_wrapper
def train_w2v_model(path_input_cleaned_txt, path_output_model, *args, **kwargs):
    """ Train a word2vec model using the LineSentence file in input_path,
    save the model to model_path.count

    Arguments:
        input_path {str} -- Corpus for training, each line is a sentence
        model_path {str} -- Where to save the model?
    """
    # pathlib.Path(path_output_model).parent.mkdir(parents=True, exist_ok=True)
    corpus_confcall = gensim.models.word2vec.PathLineSentences(
        str(path_input_cleaned_txt), max_sentence_length=10000000
    )
    model = gensim.models.Word2Vec(corpus_confcall, *args, **kwargs)
    model.save(str(path_output_model))


"""word dictionary semi-supervised under word2vec model, auto function"""


def auto_w2v_semisup_dict(
        path_input_w2v_model,
        seed_words_dict,
        restrict_vocab_per: typing.Optional[float],
        model_dims: int,

):
    model = gensim.models.Word2Vec.load(path_input_w2v_model)

    vocab_number = len(model.wv.vocab)

    print("Vocab size in the w2v model: {}".format(vocab_number))

    # expand dictionary
    expanded_words_dict = _dictionary.expand_words_dimension_mean(
        word2vec_model=model,
        seed_words=seed_words_dict,
        restrict=restrict_vocab_per,
        n=model_dims,
    )

    # make sure that one word only loads to one dimension
    expanded_words_dict = _dictionary.deduplicate_keywords(
        word2vec_model=model,
        expanded_words=expanded_words_dict,
        seed_words=seed_words_dict,
    )

    # rank the words under each dimension by similarity to the seed words
    expanded_words_dict = _dictionary.rank_by_sim(
        expanded_words_dict, seed_words_dict, model
    )
    # output the dictionary
    return expanded_words_dict
