import time
import stanza.server
from stanza.server import CoreNLPClient
import typing


class _ParserBasic:

    def __init__(self,
                 parsing_choices: typing.Iterable[str],
                 mwe_dep_types: set
                 ):
        """
        parse the sentence with NER tagging and MWEs concatenated, etc
        For example: Rickey Hall -> [NER:PERSON]Rickey[pos:NNP]_Hall[pos:NNP]
        :param mwe_dep_types: a list of MWEs in Universal Dependencies v1: set(["mwe", "compound", "compound:prt"])
        :param parsing_choices: a list of parsing choices. The order is not important.
            Here is the explaination of each element:
            **'Lemmatize'**: lemmatize the word, for example, wanted -> want
            **'POStags'**: add POS tags to the word **IN SUFFIX**, for example, want -> want[pos:VB]
                see https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html for detail.
                Besides, other 12 kinds of punctuations are not list in penn treebank pos. Like `,:#
            **'NERtags'**: add NER tags to the word *IN PREFIX*,
                for example, Stanford University -> [NER:ORGANIZATION]Stanford_University
            **'DepParseMWECompounds'**: use dep parsing to concatenate MWEs and compounds, for example, go to -> go_to.
                                    If you want use this function, you have to set mwe_dep_types.
                                    like mwe_dep_types: set = set(["mwe", "compound", "compound:prt"])
        """
        self.mwe_dep_types = mwe_dep_types
        if not isinstance(mwe_dep_types, set):
            raise ValueError('mwe dep types must be set!')

        self.parsing_choices_all = {'Lemmatize', 'POStags', 'NERtags', 'DepParseMWECompounds'}

        if set([i for i in parsing_choices]).issubset(self.parsing_choices_all):
            self.parsing_choices = parsing_choices
        else:
            raise ValueError(f'parsing choices must be subset of {self.parsing_choices_all}')

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
        if 'DepParseMWECompounds' in self.parsing_choices:
            # if mwe/compound is in choices then give a list of edges, else []
            mwe_edge_sources = self.edge_simplifier(self.sentence_mwe_finder(sentence_ann))
        else:
            mwe_edge_sources = {}

        # ----------------------------------------------------------------------------------------------
        # OPTION: NERtags
        # ----------------------------------------------------------------------------------------------

        if 'NERtags' in self.parsing_choices:
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

            token_lemma = t.lemma if 'Lemmatize' in self.parsing_choices else t.originalText

            # ----------------------------------------------------------------------------------------------
            # OPTION: POStags
            # ----------------------------------------------------------------------------------------------

            token_lemma_postags = token_lemma \
                if 'POStags' not in self.parsing_choices else "{}[pos:{}]".format(token_lemma, t.pos)

            # ----------------------------------------------------------------------------------------------
            # OPTION: DepParseMWECompounds
            # ----------------------------------------------------------------------------------------------

            # concate MWEs
            if t.tokenBeginIndex not in mwe_edge_sources:
                token_lemma_postags = token_lemma_postags + " "
            else:
                token_lemma_postags = token_lemma_postags + "_"
            # Add NE tags
            if t.tokenBeginIndex in NE_BeginIndices:
                if t.ner != "O":
                    # Only add tag if the word itself is an entity.
                    # (If a Pronoun refers to an entity, mention will also tag it.)
                    token_lemma_postags = "[NER:{}]".format(NE_types[NE_j]) + token_lemma_postags
                    NE_j += 1
            sentence_parsed.append(token_lemma_postags)
        return "".join(sentence_parsed)


class DocParser(_ParserBasic):
    def __init__(self, client, parsing_choices: typing.Iterable[str], mwe_dep_types: set):
        """
        parser of document using stanza/stanford core nlp client, No-parallel version
        :param mwe_dep_types: a list of MWEs in Universal Dependencies v1: set(["mwe", "compound", "compound:prt"])
        :param parsing_choices: a list of parsing choices. The order is not important.
            Here is the explaination of each element:
            **'Lemmatize'**: lemmatize the word, for example, wanted -> want
            **'POStags'**: add POS tags to the word **IN SUFFIX**, for example, want -> want[pos:VB]
                see https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html for detail.
                Besides, other 12 kinds of punctuations are not list in penn treebank pos. Like `,:#
            **'NERtags'**: add NER tags to the word *IN PREFIX*,
                for example, Stanford University -> [NER:ORGANIZATION]Stanford_University
            **'DepParseMWECompounds'**: use dep parsing to concatenate MWEs and compounds, for example, go to -> go_to.
                                    If you want use this function, you have to set mwe_dep_types.
                                    like mwe_dep_types: set = set(["mwe", "compound", "compound:prt"])
        """
        super().__init__(parsing_choices=parsing_choices,
                         mwe_dep_types=mwe_dep_types)
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


class DocParserParallel(_ParserBasic):
    def __init__(self, parsing_choices: typing.Iterable[str], mwe_dep_types: set):
        """
        parser of document using stanza/stanford core nlp client, parallel version
        :param mwe_dep_types: a list of MWEs in Universal Dependencies v1: set(["mwe", "compound", "compound:prt"])
        :param parsing_choices: a list of parsing choices. The order is not important.
            Here is the explaination of each element:
            **'Lemmatize'**: lemmatize the word, for example, wanted -> want
            **'POStags'**: add POS tags to the word **IN SUFFIX**, for example, want -> want[pos:VB]
                see https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html for detail.
                Besides, other 12 kinds of punctuations are not list in penn treebank pos. Like `,:#
            **'NERtags'**: add NER tags to the word *IN PREFIX*,
                for example, Stanford University -> [NER:ORGANIZATION]Stanford_University
            **'DepParseMWECompounds'**: use dep parsing to concatenate MWEs and compounds, for example, go to -> go_to.
                                    If you want use this function, you have to set mwe_dep_types.
                                    like mwe_dep_types: set = set(["mwe", "compound", "compound:prt"])
        """
        super().__init__(parsing_choices=parsing_choices, mwe_dep_types=mwe_dep_types)

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
        # if not nlptoolkits._BasicKits.check_server(corenlp_endpoint, timeout=2100000):
        #     raise ConnectionError(f'{corenlp_endpoint} is not running, reset the port and try again.')
        wait_seconds = 10
        while True:
            try:
                with CoreNLPClient(
                        endpoint=corenlp_endpoint,
                        start_server=stanza.server.StartServer.DONT_START,
                        timeout=120000000
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
