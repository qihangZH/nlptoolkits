from nlptoolkits import GensimKits
import global_options
from pathlib import Path

if __name__ == '__main__':
    democls = GensimKits.Similarity.SimilarityTfidf(
        path_sentences_dataset_txt=Path(
            global_options.PROCESSED_DATA_FOLDER, "trigram", "documents.txt"
        ),
        path_sentences_dataset_index_txt=Path(
            global_options.PROCESSED_DATA_FOLDER, "parsed", "document_sent_ids.txt",
        )
    )

    print(democls.similarity_matrix(democls.doc_corpus_list[:2000],
                                    democls.doc_corpus_list[2000:]
                                    ).shape
          )