import datetime
import warnings

import gensim
import tqdm
from .. import _BasicKits


def gen_gensim_hash(astring):
    """
    For a fully deterministically reproducible run, next to defining a seed,
        you must also limit the model to a single worker thread (workers=1),
        to eliminate ordering jitter from OS thread scheduling.
        (In Python 3, reproducibility between interpreter launches also requires the use of the PYTHONHASHSEED
        environment variable to control hash randomization).

    Usage: model = gensim.models.Word2Vec (texts, workers=1, seed=1, hashfxn=hash)

    """
    return ord(astring[0])


def train_w2v_model(path_input_sentence_txt, path_output_model=None, *args, **kwargs):
    """ Train a word2vec model using the LineSentence file in input_path,
    save the model to model_path.count
    https://stackoverflow.com/questions/34831551/ensure-the-gensim-generate-the-same-word2vec-model-for-different-runs-on-the-sam

    Arguments:
        input_path {str} -- Corpus for training, each line is a sentence
        model_path {str} -- Where to save the model?
    """
    if not {'seed', 'hashfxn'}.issubset(set(kwargs.keys())):
        warnings.warn(
        '''
        NOTICE, if seed and hashfxn do not set in train_w2v_model, 
        then the RESULT COULD BE UNSTABLE!
        seed should be set as a number like 42 while hashfxn can be set by _Model.gen_gensim_hash
        IGNORE it if you expect random result or your corpus is huge enough.
        see here for problem:
        https://stackoverflow.com/questions/34831551/ensure-the-gensim-generate-the-same-word2vec-model-for-different-runs-on-the-sam
        '''
        )

    if 'workers' in kwargs:
        if kwargs['workers'] != 1:
            warnings.warn(
                '''
                NOTICE, if workers do not set to 1, 
                while seed and hashfxn do not set in train_w2v_model(if they are set as so)
                then the RESULT COULD BE UNSTABLE!
                IGNORE it if you expect random result or your corpus is huge enough.
                see here for problem:
                https://stackoverflow.com/questions/34831551/ensure-the-gensim-generate-the-same-word2vec-model-for-different-runs-on-the-sam
                '''
            )

    # pathlib.Path(path_output_model).parent.mkdir(parents=True, exist_ok=True)
    corpus_confcall = gensim.models.word2vec.PathLineSentences(
        # str(path_input_sentence_txt), max_sentence_length=10000000
        str(path_input_sentence_txt)
    )
    model = gensim.models.Word2Vec(corpus_confcall, *args, **kwargs)

    if not (path_output_model is None):
        model.save(str(path_output_model))

    return model


def train_tfidf_model_dictmod(text_list, path_output_dictionary=None, path_output_model=None):
    """
    Train a tf-idf model using the provided corpus,
    save the model and the dictionary to their respective paths.
    :return : tuple(dictionary, output_model)

    Arguments:
        text_list {list of list of str} -- each element of list is a whole text
        path_output_dictionary {str} -- Where to save the dictionary? A dict to change tokenize corpus->bow corpus
        path_output_model {str} -- Where to save the model?
    """

    # Tokenize each document in the corpus
    tokenized_corpus = [doc.split() for doc in text_list]

    # Create a dictionary
    dictionary = gensim.corpora.Dictionary(tokenized_corpus)

    # Create a BOW corpus
    bow_corpus = [dictionary.doc2bow(text) for text in tokenized_corpus]

    # Train the TF-IDF model
    tfidf_model = gensim.models.TfidfModel(bow_corpus)

    # Save the dictionary for future use
    if path_output_dictionary is not None:
        dictionary.save(path_output_dictionary)

    # Save the model for future use
    if path_output_model is not None:
        tfidf_model.save(path_output_model)

    return dictionary, tfidf_model


def train_sentence_bigram_model(input_path, model_path, phrase_min_length, phrase_threshold, connector_words):
    """ Train a phrase model and save it to the disk.

    Arguments:
        input_path {str or Path} -- input_data corpus
        model_path {str or Path} -- where to save the trained phrase model?

    Returns:
        gensim.models.phrases.Phrases -- the trained phrase model
    """
    # Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    print(datetime.datetime.now())
    print("Training phraser...")
    corpus = gensim.models.word2vec.PathLineSentences(
        str(input_path), max_sentence_length=10000000
    )
    n_lines = _BasicKits.FileT._line_counter(input_path)
    bigram_model = gensim.models.phrases.Phrases(
        tqdm.tqdm(corpus, total=n_lines),
        min_count=phrase_min_length,
        scoring="default",
        threshold=phrase_threshold,
        connector_words=connector_words,
    )
    bigram_model.save(str(model_path))
    return bigram_model
