import time
from stanfordnlp.server import CoreNLPClient


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


class DocParser(_ParserBasic):
    def __init__(self, client, mwe_dep_types: set):
        super().__init__(mwe_dep_types=mwe_dep_types)
        self.client = client

    def parse_line_to_sentences(self, doc, doc_id=None):
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

    def parse_line_to_sentences(self, doc, doc_id=None, corenlp_endpoint: str = "http://localhost:9002"):
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
        # if not nlptoolkits._BasicKits.check_server(corenlp_endpoint, timeout=2100000):
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
