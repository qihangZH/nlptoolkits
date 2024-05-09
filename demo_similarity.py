from nlptoolkits import GensimKits
import __glob_opts
from pathlib import Path

if __name__ == '__main__':
    democls = GensimKits.Similarity.SimilarityTfidf(
        path_sentences_dataset_txt=Path(
            __glob_opts.PROCESSED_DATA_FOLDER, "trigram", "documents.txt"
        ),
        path_sentences_dataset_index_txt=Path(
            __glob_opts.PROCESSED_DATA_FOLDER, "parsed", "document_sent_ids.txt",
        ),
        charset_error_encoding=__glob_opts.DEFAULT_ENCODING
    )

    print(democls.similarity_matrix(democls.doc_text_corpus_list[:3111],
                                    democls.doc_text_corpus_list[3111:],
                                    y_chunksize=1000
                                    ).shape
          )

    print(len(democls.doc_text_corpus_list[3222:]))
