import datetime
from pathlib import Path
import gensim
import tqdm
from gensim import models
from . import file_util


def train_bigram_model(input_path, model_path, phrase_min_length, phrase_threshold, stopwords_set):
    """ Train a phrase model and save it to the disk. 
    
    Arguments:
        input_path {str or Path} -- input corpus
        model_path {str or Path} -- where to save the trained phrase model?
    
    Returns:
        gensim.models.phrases.Phrases -- the trained phrase model
    """
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    print(datetime.datetime.now())
    print("Training phraser...")
    corpus = gensim.models.word2vec.PathLineSentences(
        str(input_path), max_sentence_length=10000000
    )
    n_lines = file_util.line_counter(input_path)
    bigram_model = models.phrases.Phrases(
        tqdm.tqdm(corpus, total=n_lines),
        min_count=phrase_min_length,
        scoring="default",
        threshold=phrase_threshold,
        common_terms=stopwords_set,
    )
    bigram_model.save(str(model_path))
    return bigram_model


def bigram_transform(line, bigram_phraser):
    """ Helper file fore file_bigramer
    Note: Needs a phraser object or phrase model.

    Arguments:
        line {str}: a line 

    return: a line with phrases joined using "_"
    """
    return " ".join(bigram_phraser[line.split()])


def file_bigramer(input_path, output_path, model_path, threshold=None, scoring=None):
    """ Transform an input text file into a file with 2-word phrases. 
    Apply again to learn 3-word phrases. 

    Arguments:
        input_path {str}: Each line is a sentence
        ouput_file {str}: Each line is a sentence with 2-word phraes concatenated
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    bigram_model = gensim.models.phrases.Phrases.load(str(model_path))
    if scoring is not None:
        bigram_model.scoring = getattr(gensim.models.phrases, scoring)
    if threshold is not None:
        bigram_model.threshold = threshold
    # bigram_phraser = models.phrases.Phraser(bigram_model)
    with open(input_path, "r", encoding='utf-8') as f:
        input_data = f.readlines()
    data_bigram = [bigram_transform(l, bigram_model) for l in tqdm.tqdm(input_data)]
    with open(output_path, "w", encoding='utf-8') as f:
        f.write("\n".join(data_bigram) + "\n")
    assert len(input_data) == file_util.line_counter(output_path)



