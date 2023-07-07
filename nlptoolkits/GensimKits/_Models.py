import gensim


def train_w2v_model(path_input_sentence_txt, path_output_model=None, *args, **kwargs):
    """ Train a word2vec model using the LineSentence file in input_path,
    save the model to model_path.count

    Arguments:
        input_path {str} -- Corpus for training, each line is a sentence
        model_path {str} -- Where to save the model?
    """
    # pathlib.Path(path_output_model).parent.mkdir(parents=True, exist_ok=True)
    corpus_confcall = gensim.models.word2vec.PathLineSentences(
        str(path_input_sentence_txt), max_sentence_length=10000000
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
