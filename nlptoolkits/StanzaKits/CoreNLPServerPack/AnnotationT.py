import stanza.server
from stanza.server import CoreNLPClient
import typing
import time

class _AnnotatorBasic:

    def __init__(self,
                 annotation_choices: typing.Iterable[str],
                 mwe_dep_types: set,
                 pos_tag_label: str,
                 ner_tag_label: str,
                 sentiment_tag_label: str,
                 compounding_sep_string: str,
                 token_sep_string: str
                 ):
        """
        parse the sentence with NER tagging and MWEs concatenated, etc
        For example: Rickey Hall ->
        [{self.ner_tag_label}:PERSON]Rickey[{self.pos_tag_label}:NNP]_Hall[{self.pos_tag_label}:NNP]
        Moreover, if sentiment is choosed, then the output will be:
        [{self.sentiment_tag_label}:2] [{self.ner_tag_label}:PERSON]Rickey[{self.pos_tag_label}:NNP]_Hall[{self.pos_tag_label}:NNP]
        :param mwe_dep_types: a list of MWEs in Universal Dependencies v1: set(["mwe", "compound", "compound:prt"])
        :param annotation_choices: a list of parsing choices. The order is not important.
            Here is the explaination of each element:
            **'Lemmatize'**: lemmatize the word, for example, wanted -> want
            **'POStags'**: add POS tags to the word **IN SUFFIX**, for example, want -> want[{self.pos_tag_label}:VB]
                see https://erwinkomen.ruhosting.nl/eng/2014_Longdale-Labels.htm for detail.
                Besides, other 12 kinds of punctuations are not list in penn treebank pos. Like `,:#
            **'NERtags'**: add NER tags to the word *IN PREFIX*,
                for example, Stanford University -> [{self.ner_tag_label}:ORGANIZATION]Stanford_University
                For English, by default, this annotator recognizes named (PERSON, LOCATION, ORGANIZATION, MISC),
                numerical (MONEY, NUMBER, ORDINAL, PERCENT),
                and temporal (DATE, TIME, DURATION, SET) entities (12 classes).
            **'DepParseMWECompounds'**: use dep parsing to concatenate MWEs and compounds, for example, go to -> go_to.
                If you want use this function, you have to set mwe_dep_types.
                like mwe_dep_types: set = set(["mwe", "compound", "compound:prt"])
            **SentenceSentiment**: add sentence sentiment to the sentence, if use, then 'sentiment' have to be added to
                stanford corenlp client pipeline choices.
        :param pos_tag_label: the POS tag used in the output
        :param ner_tag_label: the NER tag used in the output
        :param sentiment_tag_label: the sentiment tag used in the output
        :param compounding_sep_string: the separator string used to concatenate MWEs and compounds
        :param token_sep_string: the separator string used to separate tokens
        """
        self.annotation_choices_all = {'Lemmatize', 'POStags', 'NERtags', 'DepParseMWECompounds', 'SentenceSentiment'}

        if set([i for i in annotation_choices]).issubset(self.annotation_choices_all):
            self.annotation_choices = annotation_choices
        else:
            raise ValueError(f'parsing choices must be subset of {self.annotation_choices_all}')

        self.mwe_dep_types = mwe_dep_types
        if not isinstance(mwe_dep_types, set):
            raise ValueError('mwe dep types must be set!')

        self.pos_tag_label = pos_tag_label

        self.ner_tag_label = ner_tag_label

        self.sentiment_tag_label = sentiment_tag_label

        self.compounding_sep_string = compounding_sep_string

        self.token_sep_string = token_sep_string

    def sentence_mwe_finder(self, sentence_ann):
        """Find the edges between words that are MWEs

        Arguments:
            sentence_ann {CoreNLP_pb2.Sentence} -- An annotated sentence

        Keyword Arguments:
            dep_types {[str]} -- a list of MWEs in Universal Dependencies v1
            (default: s{set(["mwe", "compound", "compound:prt"])})
            MOREINFO, http://universaldependencies.org/docsv1/u/dep/compound.html
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
    def edge_simplifier(edges) -> set:
        """Simplify list of edges to a set of edge sources. Edges always points to the next word.
        Self-pointing edges are removed

        Arguments:
            edges {[[a,b], [c,d]...]} -- a list of edges using tokenBeginIndex; a <= b.

        Returns:
            {a, c, ...} -- a list of edge sources, edges always go from word_i to word_i+1
        """
        edge_sources = set([])  # edge that connects next token
        for e in edges:
            if e[0] + 1 == e[1]:
                edge_sources.add(e[0])
            else:
                for i in range(e[0], e[1]):
                    edge_sources.add(i)
        return edge_sources

    # ----------------------------------------------------------------------------------------------
    # Main workers
    # ----------------------------------------------------------------------------------------------

    def process_sentence(self, sentence_ann):
        """Process a raw sentence

        Arguments:
            sentence_ann {CoreNLP_pb2.Sentence} -- An annotated sentence

        Returns:
            str -- sentence with NER tagging and MWEs concatenated
        """
        # ['Lemmatize','POStags','NERtags', 'DepParseMWECompounds']

        # ----------------------------------------------------------------------------------------------
        # OPTION: DepParseMWECompounds
        # ----------------------------------------------------------------------------------------------
        if 'DepParseMWECompounds' in self.annotation_choices:
            # if mwe/compound is in choices then give a list of edges, else []
            mwe_edge_sources = self.edge_simplifier(self.sentence_mwe_finder(sentence_ann))
        else:
            mwe_edge_sources = set([])

        # ----------------------------------------------------------------------------------------------
        # OPTION: NERtags
        # ----------------------------------------------------------------------------------------------

        if 'NERtags' in self.annotation_choices:
            # NE_edges can span more than two words or self-pointing
            NE_edges, NE_types = self.sentence_NE_finder(sentence_ann)
            # For tagging NEs
            NE_BeginIndices = [e[0] for e in NE_edges]
            # Unpack NE_edges to two-word edges set([i,j],..)
            NE_edge_sources = self.edge_simplifier(NE_edges)
            # For concat MWEs, multi-words NEs are MWEs too
            mwe_edge_sources |= NE_edge_sources
        else:
            NE_BeginIndices = []
            NE_types = []

        sentence_parsed = []

        NE_j = 0
        for i, t in enumerate(sentence_ann.token):

            # ----------------------------------------------------------------------------------------------
            # OPTION: Lemmatize
            # ----------------------------------------------------------------------------------------------

            token_lemma = t.lemma if 'Lemmatize' in self.annotation_choices else t.originalText

            # ----------------------------------------------------------------------------------------------
            # OPTION: POStags
            # ----------------------------------------------------------------------------------------------

            token_lemma_postags = f"{token_lemma}[{self.pos_tag_label}:{t.pos}]" \
                if 'POStags' in self.annotation_choices else token_lemma

            # ----------------------------------------------------------------------------------------------
            # OPTION: DepParseMWECompounds, NERtags(NER is a kind of MWE too)
            # ----------------------------------------------------------------------------------------------

            # concate MWEs
            if t.tokenBeginIndex not in mwe_edge_sources:
                token_lemma_postags = token_lemma_postags + self.token_sep_string
            else:
                # token_lemma_postags = token_lemma_postags + "_"
                token_lemma_postags = token_lemma_postags + self.compounding_sep_string
            # Add NE tags
            if t.tokenBeginIndex in NE_BeginIndices:
                if t.ner != "O":
                    # Only add tag if the word itself is an entity.
                    # (If a Pronoun refers to an entity, mention will also tag it.)
                    token_lemma_postags = f"[{self.ner_tag_label}:{NE_types[NE_j]}]" + token_lemma_postags
                    NE_j += 1
            sentence_parsed.append(token_lemma_postags)

        processed_sentence = "".join(sentence_parsed)

        # ----------------------------------------------------------------------------------------------
        # OPTION: SentenceSentiment
        # ----------------------------------------------------------------------------------------------
        if 'SentenceSentiment' in self.annotation_choices:
            sentence_sentiment = sentence_ann.sentiment
            processed_sentence = f"[{self.sentiment_tag_label}:{sentence_sentiment}]" + \
                                 self.token_sep_string + \
                                 processed_sentence

        return processed_sentence


class LineAnnotator(_AnnotatorBasic):
    def __init__(self,
                 client,
                 annotation_choices: typing.Iterable[str],
                 mwe_dep_types: set,
                 pos_tag_label: str,
                 ner_tag_label: str,
                 sentiment_tag_label: str,
                 compounding_sep_string: str,
                 token_sep_string: str
                 ):
        """
        parse the sentence with NER tagging and MWEs concatenated, etc
        For example: Rickey Hall ->
        [{self.ner_tag_label}:PERSON]Rickey[{self.pos_tag_label}:NNP]_Hall[{self.pos_tag_label}:NNP]
        Moreover, if sentiment is choosed, then the output will be:
        [{self.sentiment_tag_label}:2] [{self.ner_tag_label}:PERSON]Rickey[{self.pos_tag_label}:NNP]_Hall[{self.pos_tag_label}:NNP]
        :param mwe_dep_types: a list of MWEs in Universal Dependencies v1: set(["mwe", "compound", "compound:prt"])
        :param annotation_choices: a list of parsing choices. The order is not important.
            Here is the explaination of each element:
            **'Lemmatize'**: lemmatize the word, for example, wanted -> want
            **'POStags'**: add POS tags to the word **IN SUFFIX**, for example, want -> want[{self.pos_tag_label}:VB]
                see https://erwinkomen.ruhosting.nl/eng/2014_Longdale-Labels.htm for detail.
                Besides, other 12 kinds of punctuations are not list in penn treebank pos. Like `,:#
            **'NERtags'**: add NER tags to the word *IN PREFIX*,
                for example, Stanford University -> [{self.ner_tag_label}:ORGANIZATION]Stanford_University
                For English, by default, this annotator recognizes named (PERSON, LOCATION, ORGANIZATION, MISC),
                numerical (MONEY, NUMBER, ORDINAL, PERCENT),
                and temporal (DATE, TIME, DURATION, SET) entities (12 classes).
            **'DepParseMWECompounds'**: use dep parsing to concatenate MWEs and compounds, for example, go to -> go_to.
                If you want use this function, you have to set mwe_dep_types.
                like mwe_dep_types: set = set(["mwe", "compound", "compound:prt"])
            **SentenceSentiment**: add sentence sentiment to the sentence, if use, then 'sentiment' have to be added to
                stanford corenlp client pipeline choices.
        :param pos_tag_label: the POS tag used in the output
        :param ner_tag_label: the NER tag used in the output
        :param sentiment_tag_label: the sentiment tag used in the output
        :param compounding_sep_string: the separator string used to concatenate MWEs and compounds, default "[SEP]"
        :param token_sep_string: the separator string used to separate tokens, default " "
        """
        super().__init__(annotation_choices=annotation_choices,
                         mwe_dep_types=mwe_dep_types,
                         pos_tag_label=pos_tag_label,
                         ner_tag_label=ner_tag_label,
                         sentiment_tag_label=sentiment_tag_label,
                         compounding_sep_string=compounding_sep_string,
                         token_sep_string=token_sep_string
                         )
        self.client = client

    def parse_line_to_sentences(self, doc, doc_id):
        """
        Main method: Annotate a document using CoreNLP client
        :param doc: raw string of a document
        :param doc_id: raw string of a document ID
        :return: **a tuple of (sentences_processed {[str]}, doc_ids {[str]})**
            sentences_processed {[str]} -- a list of processed_data sentences with NER tagged
                and MWEs concatenated
            doc_ids {[str]} -- a list of processed_data sentence IDs [docID1_1, docID1_2...]
        """
        doc_ann = self.client.annotate(doc)
        sentences_processed = []
        doc_ids = []
        for i, sentence in enumerate(doc_ann.sentence):
            sentences_processed.append(self.process_sentence(sentence))
            doc_ids.append(str(doc_id) + "_" + str(i))
        return sentences_processed, doc_ids


class LineAnnotatorParallel(_AnnotatorBasic):
    def __init__(self,
                 annotation_choices: typing.Iterable[str],
                 mwe_dep_types: set,
                 pos_tag_label: str,
                 ner_tag_label: str,
                 sentiment_tag_label: str,
                 compounding_sep_string: str,
                 token_sep_string: str,
                 timeout: int,
                 annotation_error: typing.Literal['raise', 'ignore', 'warn']
                 ):
        """
        parse the sentence with NER tagging and MWEs concatenated, etc
        For example: Rickey Hall ->
        [{self.ner_tag_label}:PERSON]Rickey[{self.pos_tag_label}:NNP]_Hall[{self.pos_tag_label}:NNP]
        Moreover, if sentiment is choosed, then the output will be:
        [{self.sentiment_tag_label}:2] [{self.ner_tag_label}:PERSON]Rickey[{self.pos_tag_label}:NNP]_Hall[{self.pos_tag_label}:NNP]
        :param mwe_dep_types: a list of MWEs in Universal Dependencies v1: set(["mwe", "compound", "compound:prt"])
        :param annotation_choices: a list of parsing choices. The order is not important.
            Here is the explaination of each element:
            **'Lemmatize'**: lemmatize the word, for example, wanted -> want
            **'POStags'**: add POS tags to the word **IN SUFFIX**, for example, want -> want[{self.pos_tag_label}:VB]
                see https://erwinkomen.ruhosting.nl/eng/2014_Longdale-Labels.htm for detail.
                Besides, other 12 kinds of punctuations are not list in penn treebank pos. Like `,:#
            **'NERtags'**: add NER tags to the word *IN PREFIX*,
                for example, Stanford University -> [{self.ner_tag_label}:ORGANIZATION]Stanford_University
                For English, by default, this annotator recognizes named (PERSON, LOCATION, ORGANIZATION, MISC),
                numerical (MONEY, NUMBER, ORDINAL, PERCENT),
                and temporal (DATE, TIME, DURATION, SET) entities (12 classes).
            **'DepParseMWECompounds'**: use dep parsing to concatenate MWEs and compounds, for example, go to -> go_to.
                If you want use this function, you have to set mwe_dep_types.
                like mwe_dep_types: set = set(["mwe", "compound", "compound:prt"])
            **SentenceSentiment**: add sentence sentiment to the sentence, if use, then 'sentiment' have to be added to
                stanford corenlp client pipeline choices.
        :param pos_tag_label: the POS tag used in the output
        :param ner_tag_label: the NER tag used in the output
        :param sentiment_tag_label: the sentiment tag used in the output
        :param compounding_sep_string: the separator string used to concatenate MWEs and compounds, default "[SEP]"
        :param token_sep_string: the separator string used to separate tokens, default " "
        :param timeout: the timeout for each document task, milliseconds
        :param annotation_error: the error handling method for annotation error, 'raise', 'ignore', 'warn'
        """
        super().__init__(annotation_choices=annotation_choices,
                         mwe_dep_types=mwe_dep_types,
                         pos_tag_label=pos_tag_label,
                         ner_tag_label=ner_tag_label,
                         sentiment_tag_label=sentiment_tag_label,
                         compounding_sep_string=compounding_sep_string,
                         token_sep_string=token_sep_string
                         )

        self.timeout = timeout

        self.annotation_error = annotation_error
        assert self.annotation_error in ['raise', 'ignore', 'warn'], "annotation_error should be 'raise', 'ignore', 'warn'"

    def parse_line_to_sentences(self, doc, doc_id, corenlp_endpoint: str):
        """
        Main method: Annotate a document using CoreNLP client
        :param doc: raw string of a document
        :param doc_id: raw string of a document ID
        :param corenlp_endpoint: core nlp port to deal with data, like "http://localhost:9002"
        :return: **a tuple of (sentences_processed {[str]}, doc_ids {[str]})**
            sentences_processed {[str]} -- a list of processed_data sentences with NER tagged
                and MWEs concatenated
            doc_ids {[str]} -- a list of processed_data sentence IDs [docID1_1, docID1_2...]
        """
        """
        TODO:old-version: even not, but may cause error to shut down the server.
        We have to know if the server is or not been stopped by stanza.server.CoreNLPClient.
        However, it is not sure for the code could running will. May caused by some special characters.
        """
        with CoreNLPClient(
                endpoint=corenlp_endpoint,
                start_server=stanza.server.StartServer.DONT_START,
                timeout=self.timeout
        ) as client:
            _stime = time.time()
            try:
                doc_ann = client.annotate(doc)
            except stanza.server.client.AnnotationException as ae:
                if self.annotation_error == 'raise':
                    raise ae
                elif self.annotation_error == 'warn':
                    print(f"{doc_id}: Annotation Exception, fail to annotate, therefore return None: {ae}")
                    print(f'Annotation time: {time.time() - _stime} seconds')
                    return "", str(doc_id) + "_" + str(0)

                elif self.annotation_error == 'ignore':
                    return "", str(doc_id) + "_" + str(0)
                else:
                    raise ValueError("annotation_error should be 'raise', 'ignore', 'warn'")
            except Exception as e:
                raise e

        sentences_processed = []
        doc_sent_ids = []
        for i, sentence in enumerate(doc_ann.sentence):
            sentences_processed.append(self.process_sentence(sentence))
            doc_sent_ids.append(str(doc_id) + "_" + str(i))
        return "\n".join(sentences_processed), "\n".join(doc_sent_ids)
