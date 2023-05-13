
from pathlib import Path

from stanfordnlp.server import CoreNLPClient

import global_options
from seminlptools import file_util, preprocess

if __name__ == "__main__":
    with CoreNLPClient(
            endpoint='http://localhost:' + str(9001),
            properties={
                "ner.applyFineGrained": "false",
                "annotators": "tokenize, ssplit, pos, lemma, ner, depparse",
            },
            memory=global_options.RAM_CORENLP,
            threads=global_options.N_CORES,
            timeout=12000000,
            max_char_length=1000000,
    ) as client:
        corpus_preprocessor = preprocess.preprocessor(client)


        def process_line(line, lineID):
            """Process each line and return a tuple of sentences, sentence_IDs,

            Arguments:
                line {str} -- a document
                lineID {str} -- the document ID

            Returns:
                str, str -- processed document with each sentence in a line,
                            sentence IDs with each in its own line: lineID_0 lineID_1 ...
            """
            try:
                sentences_processed, doc_sent_ids = corpus_preprocessor.process_document(
                    line, lineID
                )
                return "\n".join(sentences_processed), "\n".join(doc_sent_ids)
            except Exception as e:
                print(e)
                print("Exception in line: {}".format(lineID))


        seminlpclassify.parse_largefile(
            path_input_file_txt=Path(global_options.DATA_FOLDER, "input", "documents.txt"),
            path_output=Path(
                global_options.DATA_FOLDER, "processed", "parsed", "documents.txt"
            ),
            path_input_file_ids=file_util.file_to_list(
                Path(global_options.DATA_FOLDER, "input", "document_ids.txt")
            ),
            path_output_index=Path(
                global_options.DATA_FOLDER, "processed", "parsed", "document_sent_ids.txt"
            ),
            parse_line_func=process_line,
            chunk_size=global_options.PARSE_CHUNK_SIZE,
        )
