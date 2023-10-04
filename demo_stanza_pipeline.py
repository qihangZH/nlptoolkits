# %%
from nlptoolkits.StanzaKits import PipeLineT
import logging
import stanza
import nlptoolkits as nk

if __name__ == "__main__":
    
    logging.getLogger('stanza').setLevel(logging.ERROR)
    
    documents = [
        "Hello! This is the first document. It has several sentences. Isn't that interesting?",
        "Here's another document for you. It's a beautiful day, isn't it? I love the blue sky.",
        "The third document is here. How are you doing today? I hope everything is going well.",
        "This is the fourth document. Sometimes, it rains in the afternoon. I prefer sunny days though.",
        "Lastly, the fifth document. Did you know that pandas are native to China? They love bamboo!"
    ]
    # documents = nk.SmallKits.IOHandlerT.file_to_list('./input_data/tweets_origin.txt')

    # pcls = PipeLineT.PipeLineParser(doc_list=documents, processors='tokenize,sentiment', gpu_process_chunksize=500)
    pcls = PipeLineT.PipeLineParser(doc_list=documents, processors='tokenize,sentiment', use_gpu=False)
    
    print(pcls.raw_sentences)
    