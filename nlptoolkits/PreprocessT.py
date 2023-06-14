import pandas as pd
from stanfordnlp.server import CoreNLPClient
import re
import time
import functools
import typing
import gensim
import tqdm
from gensim import models
import datetime
from . import _BasicT


# Qihang Zhang modified in 6/6/2023, National university of singapore
# Modification for better usage, and for its flexible using
# --------------------------------------------------------------------------
# l (-1) level classes, this basic class is only for inheritance
# --------------------------------------------------------------------------

class _ParserBasic:

    def __init__(self, mwe_dep_types: set):
        """
        :param mwe_dep_types: a list of MWEs in Universal Dependencies v1: set(["mwe", "compound", "compound:prt"])
        """
        self.mwe_dep_types = mwe_dep_types
        if not isinstance(mwe_dep_types, set):
            raise ValueError('mwe dep types must be set!')

    def sentence_mwe_finder(self, sentence_ann):
        """Find the edges between words that are MWEs

        Arguments:
            sentence_ann {CoreNLP_pb2.Sentence} -- An annotated sentence

        Keyword Arguments:
            dep_types {[str]} -- a list of MWEs in Universal Dependencies v1
            (default: s{set(["mwe", "compound", "compound:prt"])})
            see: http://universaldependencies.org/docsv1/u/dep/compound.html
            and http://universaldependencies.org/docsv1/u/dep/mwe.html
        Returns:
            A list of edges: e.g. [(1, 2), (4, 5)]
        """
        WMEs = [
            x
            for x in sentence_ann.enhancedPlusPlusDependencies.edge
            if x.dep in self.mwe_dep_types
        ]
        wme_edges = []
        for wme in WMEs:
            edge = sorted([wme.target, wme.source])
            # Note: (-1) because edges in WMEs use indicies that indicate the end of a token (tokenEndIndex)
            # (+ sentence_ann.token[0].tokenBeginIndex) because
            # the edges indices are for current sentence, whereas tokenBeginIndex are for the document.
            wme_edges.append(
                [end - 1 + sentence_ann.token[0].tokenBeginIndex for end in edge]
            )
        return wme_edges

    @staticmethod
    def sentence_NE_finder(sentence_ann):
        """Find the edges between wordxs that are a named entity

        Arguments:
            sentence_ann {CoreNLP_pb2.Sentence} -- An annotated sentence

        Returns:
            A tuple NE_edges, NE_types
                NE_edges is a list of edges, e.g. [(1, 2), (4, 5)]
                NE_types is a list of NE types, e.g. ["ORGANIZATION", "LOCATION"]
                see https://stanfordnlp.github.io/CoreNLP/ner.html
        """
        NE_edges = []
        NE_types = []
        for m in sentence_ann.mentions:
            edge = sorted(
                [m.tokenStartInSentenceInclusive, m.tokenEndInSentenceExclusive]
            )
            # Note: edge in NEs's end index is at the end of the last token
            NE_edges.append([edge[0], edge[1] - 1])
            NE_types.append(m.entityType)
            # # alternative method:
            # NE_edges.append(sorted([field[1]
            #                         for field in m.ListFields()][1:3]))
        return NE_edges, NE_types

    @staticmethod
    def edge_simplifier(edges):
        """Simplify list of edges to a set of edge sources. Edges always points to the next word.
        Self-pointing edges are removed

        Arguments:
            edges {[[a,b], [c,d]...]} -- a list of edges using tokenBeginIndex; a <= b.

        Returns:
            [a, c, ...] -- a list of edge sources, edges always go from word_i to word_i+1
        """
        edge_sources = set([])  # edge that connects next token
        for e in edges:
            if e[0] + 1 == e[1]:
                edge_sources.add(e[0])
            else:
                for i in range(e[0], e[1]):
                    edge_sources.add(i)
        return edge_sources

    def process_sentence(self, sentence_ann):
        """Process a raw sentence

        Arguments:
            sentence_ann {CoreNLP_pb2.Sentence} -- An annotated sentence

        Returns:
            str -- sentence with NER tagging and MWEs concatenated
        """
        mwe_edge_sources = self.edge_simplifier(self.sentence_mwe_finder(sentence_ann))
        # NE_edges can span more than two words or self-pointing
        NE_edges, NE_types = self.sentence_NE_finder(sentence_ann)
        # For tagging NEs
        NE_BeginIndices = [e[0] for e in NE_edges]
        # Unpack NE_edges to two-word edges set([i,j],..)
        NE_edge_sources = self.edge_simplifier(NE_edges)
        # For concat MWEs, multi-words NEs are MWEs too
        mwe_edge_sources |= NE_edge_sources
        sentence_parsed = []

        NE_j = 0
        for i, t in enumerate(sentence_ann.token):
            token_lemma = "{}[pos:{}]".format(t.lemma, t.pos)
            # concate MWEs
            if t.tokenBeginIndex not in mwe_edge_sources:
                token_lemma = token_lemma + " "
            else:
                token_lemma = token_lemma + "_"
            # Add NE tags
            if t.tokenBeginIndex in NE_BeginIndices:
                if t.ner != "O":
                    # Only add tag if the word itself is an entity.
                    # (If a Pronoun refers to an entity, mention will also tag it.)
                    token_lemma = "[NER:{}]".format(NE_types[NE_j]) + token_lemma
                    NE_j += 1
            sentence_parsed.append(token_lemma)
        return "".join(sentence_parsed)


"""New script likely in next part->"""
# --------------------------------------------------------------------------
# l0 level functions
# --------------------------------------------------------------------------

"""bigram models to make seperate words to one word concat with _"""


def train_bigram_model(input_path, model_path, phrase_min_length, phrase_threshold, stopwords_set):
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
    n_lines = _BasicT._line_counter(input_path)
    bigram_model = models.phrases.Phrases(
        tqdm.tqdm(corpus, total=n_lines),
        min_count=phrase_min_length,
        scoring="default",
        threshold=phrase_threshold,
        common_terms=stopwords_set,
    )
    bigram_model.save(str(model_path))
    return bigram_model


def file_bigramer(input_path, output_path, model_path, threshold=None, scoring=None):
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
    assert len(input_data) == _BasicT._line_counter(output_path)


"""Tokenize the Sentences in each line"""
pass


# --------------------------------------------------------------------------
# l0 level classes
# --------------------------------------------------------------------------

class DocParser(_ParserBasic):
    def __init__(self, client, mwe_dep_types: set):
        super().__init__(mwe_dep_types=mwe_dep_types)
        self.client = client

    def process_document(self, doc, doc_id=None):
        """Main method: Annotate a document using CoreNLP client

        Arguments:
            doc {str} -- raw string of a document
            doc_id {str} -- raw string of a document ID

        Returns:
            sentences_processed {[str]} -- a list of processed_data sentences with NER tagged
                and MWEs concatenated
            doc_ids {[str]} -- a list of processed_data sentence IDs [docID1_1, docID1_2...]
            Example:
                Input: "When I was a child in Ohio,
                I always wanted to go to Stanford University with respect to higher education.
                But I had to go along with my parents."
                Output: 
                
                'when I be a child in ['when I be a child in [NER:LOCATION]Ohio ,
                'I always want to go to [NER:ORGANIZATION]Stanford_University with_respect_to higher education .'
                'but I have to go_along with my parent . '

                doc1_1
                doc1_2
        
        Note:
            When the doc is empty, both doc_id and sentences processed_data will be too.
        """
        doc_ann = self.client.annotate(doc)
        sentences_processed = []
        doc_ids = []
        for i, sentence in enumerate(doc_ann.sentence):
            sentences_processed.append(self.process_sentence(sentence))
            doc_ids.append(str(doc_id) + "_" + str(i))
        return sentences_processed, doc_ids


class DocParserParallel(_ParserBasic):
    def __init__(self, mwe_dep_types: set):
        super().__init__(mwe_dep_types=mwe_dep_types)

    def process_document(self, doc, doc_id=None, corenlp_endpoint: str = "http://localhost:9002"):
        """Main method: Annotate a document using CoreNLP client

        Arguments:
            doc {str} -- raw string of a document
            doc_id {str} -- raw string of a document ID
            corenlp_endpoint {str} -- core nlp port to deal with data, like 9001, 9002...

        Returns:
            sentences_processed {[str]} -- a list of processed_data sentences with NER tagged
                and MWEs concatenated
            doc_ids {[str]} -- a list of processed_data sentence IDs [docID1_1, docID1_2...]
            Example:
                Input: "When I was a child in Ohio,
                I always wanted to go to Stanford University with respect to higher education.
                But I had to go along with my parents."
                Output:

                'when I be a child in ['when I be a child in [NER:LOCATION]Ohio ,
                I always want to go to [NER:ORGANIZATION]Stanford_University with_respect_to higher education .
                'but I have to go_along with my parent . '

                doc1_1
                doc1_2

        Note:
            When the doc is empty, both doc_id and sentences processed_data will be too.
        """
        # if not nlptoolkits.qihangfuncs.check_server(corenlp_endpoint, timeout=2100000):
        #     raise ConnectionError(f'{corenlp_endpoint} is not running, reset the port and try again.')
        wait_seconds = 10
        while True:
            try:
                with CoreNLPClient(
                        endpoint=corenlp_endpoint, start_server=False, timeout=120000000
                ) as client:
                    doc_ann = client.annotate(doc)

                    break

            except Exception as e:
                print(e, f'occurs, \nwait for {wait_seconds} seconds')
                time.sleep(wait_seconds)
                wait_seconds = wait_seconds * 1.2

        sentences_processed = []
        doc_sent_ids = []
        for i, sentence in enumerate(doc_ann.sentence):
            sentences_processed.append(self.process_sentence(sentence))
            doc_sent_ids.append(str(doc_id) + "_" + str(i))
        return "\n".join(sentences_processed), "\n".join(doc_sent_ids)


class LineTextCleaner:
    """Clean the text parsed by CoreNLP (preprocessor)
    """

    def __init__(self,
                 stopwords_set: set,
                 ner_keep_types_origin_list: typing.Optional[list] = None,
                 token_minlength: typing.Optional[int] = 2,
                 punctuations_set: set = set(["-lrb-", "-rrb-", "-lsb-", "-rsb-", "'s"]),
                 is_remove_no_alphabet_contains: bool = True,
                 ):
        """
        :param stopwords_set: stop word set to be remove
        :param ner_keep_types_origin_list: a name list corenlp NER types which should be keep,
                                            or will remove origin name and only keep NER types,
                                            should input None or list
        :param token_minlength: default 2 the minimal length of each token, else remove,
                                remove all the tokens which length is less than this length
                                if None then not remove
        :param punctuations_set: punctuation set to be remove, especially
        :param is_remove_no_alphabet_contains: is remove words(token) contains no alphabetic

        """
        if not isinstance(stopwords_set, set):
            raise ValueError('stopwords_set must be set')

        if not (
                isinstance(ner_keep_types_origin_list, list) or
                (ner_keep_types_origin_list is None)
        ):
            raise ValueError('ner_keep_types_origin_list must be list or None')

        if not (
                isinstance(token_minlength, int) or
                (token_minlength is None)
        ):
            raise ValueError('token_minlength must be int or None')

        if not isinstance(punctuations_set, set):
            raise ValueError('punctuations_set must be set')

        if not isinstance(is_remove_no_alphabet_contains, bool):
            raise ValueError('is_removenum must be bool')

        self.stopwords = stopwords_set

        self.ner_keep_types_origin_list = ner_keep_types_origin_list if ner_keep_types_origin_list else list()

        self.token_minlength = token_minlength

        self.punctuations = punctuations_set if punctuations_set else set()

        self.is_removenum = is_remove_no_alphabet_contains

    def remove_ner(self, line):
        """Remove the named entity and only leave the tag
        
        Arguments:
            line {str} -- text processed_data by the preprocessor
        
        Returns:
            str -- text with NE replaced by NE tags, 
            e.g. [NER:PERCENT]16_% becomes [NER:PERCENT]
        """
        # always make the line lower case
        line = line.lower()
        # remove ner for words of specific types:
        if self.ner_keep_types_origin_list:  # have a loop if it is not None
            for i in self.ner_keep_types_origin_list:
                line = re.sub(rf"(\[ner:{i.lower()}\])(\S+)", r"\2", line, flags=re.IGNORECASE)

        # update for deeper search, remove the entity name
        NERs = re.compile(r"(\[ner:\w+\])(\S+)", flags=re.IGNORECASE)
        line = re.sub(NERs, r"\1", line)
        return line

    def remove_puct_num(self, line):
        """Remove tokens that are only numerics and puctuation marks

        Arguments:
            line {str} -- text processed_data by the preprocessor
        
        Returns:
            str -- text with stopwords, numerics, 1-letter words removed
        """
        tokens = line.strip().lower().split(" ")  # do not use nltk.tokenize here
        tokens = [re.sub("\[pos:.*?\]", "", t, flags=re.IGNORECASE) for t in tokens]

        # these are tagged bracket and parenthesises
        if self.punctuations or self.stopwords:
            puncts_stops = (self.punctuations | self.stopwords)
            # filter out numerics and 1-letter words as recommend by
            # https://sraf.nd.edu/textual-analysis/resources/#StopWords
        else:
            puncts_stops = set()

        def _lambda_filter_token_bool(t):
            """
            the judegement after the function is help to give
            """
            contain_alphabet = any(c.isalpha() for c in t) if self.is_removenum else True
            is_not_punctuation_stopwords = t not in puncts_stops
            is_biggerthan_minlength = len(t) >= self.token_minlength if self.token_minlength else True

            return all([contain_alphabet, is_not_punctuation_stopwords, is_biggerthan_minlength])

        tokens = filter(
            # lambda t: any(c.isalpha() for c in t)
            #           and (t not in puncts_stops)
            #           and (len(t) > 1),
            _lambda_filter_token_bool,
            tokens,
        )
        return " ".join(tokens)

    def clean(self, line, index):
        """Main function that chains all filters together and applies to a string. 
        """
        return (
            functools.reduce(
                lambda obj, func: func(obj),
                [self.remove_ner, self.remove_puct_num],
                line,
            ),
            index,
        )
