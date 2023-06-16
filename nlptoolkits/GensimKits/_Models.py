import gensim


def train_w2v_model(path_input_cleaned_txt, path_output_model=None, *args, **kwargs):
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

    if not (path_output_model is None):
        model.save(str(path_output_model))

    return model


def train_tfidf_model_tuple(path_input_cleaned_txt, path_output_dictionary=None, path_output_model=None):
    """
    Train a tf-idf model using the LineSentence file in input_path,
    save the model and the dictionary to their respective paths.
    :return : tuple(dictionary, output_model)

    Arguments:
        path_input_cleaned_txt {str} -- Corpus for training, each line is a sentence
        path_output_dictionary {str} -- Where to save the dictionary?
        path_output_model {str} -- Where to save the model?

    """

    # Load sentences from the text file
    corpus_confcall = gensim.models.word2vec.PathLineSentences(
        str(path_input_cleaned_txt), max_sentence_length=10000000
    )

    # Create a dictionary
    dictionary = gensim.corpora.Dictionary(corpus_confcall)

    # Create a BOW corpus
    corpus = [dictionary.doc2bow(text) for text in corpus_confcall]

    # Train the TF-IDF model
    tfidf_model = gensim.models.TfidfModel(corpus)

    # Save the dictionary for future use
    if not (path_output_dictionary is None):
        dictionary.save(path_output_dictionary)

    # Save the model for future use
    if not (path_output_model is None):
        tfidf_model.save(path_output_model)

    return dictionary, tfidf_model
