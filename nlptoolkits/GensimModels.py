import gensim


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
