"""Global options for analysis
"""
import os
from pathlib import Path
from typing import Dict, List, Set

# Hardware options
N_CORES: int = os.cpu_count()  # max number of CPU cores to use
RAM_CORENLP: str = "60G"  # max RAM allocated for parsing using CoreNLP; increase to speed up parsing
PARSE_CHUNK_SIZE: int = 20
# number of lines in the input_data file to process uing CoreNLP at once.
# Increase on workstations with larger RAM (e.g. to 1000 if RAM is 64G)
ADDRESS_CORENLP: str = "http://localhost:9010"
# Directory locations
# os.environ[
#     "CORENLP_HOME"
# ] = "F:/stanford-corenlp-4.5.4"  # location of the CoreNLP models; use / to seperate folders


PROJ_FOLDER = os.path.abspath(os.path.dirname(__file__)) + '/'

# result/processed data folder, should be cleaned before run
INPUT_DATA_FOLDER: str = PROJ_FOLDER + "input_data"
PROCESSED_DATA_FOLDER: str = PROJ_FOLDER + "processed_data/"
MODEL_FOLDER: str = PROJ_FOLDER + "models/"
OUTPUT_FOLDER: str = PROJ_FOLDER + "outputs/"

# Parsing and analysis options
STOPWORDS: Set[str] = set(
    Path("resources", "StopWords_Generic.txt").read_text().lower().split()
)  # Set of stopwords from https://sraf.nd.edu/textual-analysis/resources/#StopWords
PHRASE_THRESHOLD: int = 10  # threshold of the phraser module (smaller -> more phrases)
PHRASE_MIN_COUNT: int = 10  # min number of times a bigram needs to appear in the corpus to be considered as a phrase
W2V_DIM: int = 300  # dimension of word2vec vectors
W2V_WINDOW: int = 5  # window size in word2vec
W2V_ITER: int = 20  # number of iterations in word2vec
N_WORDS_DIM: int = 500  # max number of words in each dimension of the dictionary
DICT_RESTRICT_VOCAB = None
# change to a fraction number (e.g. 0.2) to restrict the dictionary vocab in the top 20% of most frequent vocab

# Inputs for constructing the expanded dictionary
DIMS: List[str] = ["integrity", "teamwork", "innovation", "respect", "quality"]

SEED_WORDS: Dict[str, List[str]] = {
    "risk": [
         'run', 'withdraw', 'deposit', 'insured', 'uninsured','risk'
    ],
    "contagion":[
        'contagion','systemic', 'spillover', 'fed', 'contagion', 'whole_system', 'spread', 'meltfown'
    ]
}

# SEED_WORDS: Dict[str, List[str]] = {
#     "integrity": [
#         "integrity",
#         "ethic",
#         "ethical",
#         "accountable",
#         "accountability",
#         "trust",
#         "honesty",
#         "honest",
#         "honestly",
#         "fairness",
#         "responsibility",
#         "responsible",
#         "transparency",
#         "transparent",
#     ],
#     "teamwork": [
#         "teamwork",
#         "collaboration",
#         "collaborate",
#         "collaborative",
#         "cooperation",
#         "cooperate",
#         "cooperative",
#     ],
#     "innovation": [
#         "innovation",
#         "innovate",
#         "innovative",
#         "creativity",
#         "creative",
#         "create",
#         "passion",
#         "passionate",
#         "efficiency",
#         "efficient",
#         "excellence",
#         "pride",
#     ],
#     "respect": [
#         "respectful",
#         "talent",
#         "talented",
#         "employee",
#         "dignity",
#         "empowerment",
#         "empower",
#     ],
#     "quality": [
#         "quality",
#         "customer",
#         "customer_commitment",
#         "dedication",
#         "dedicated",
#         "dedicate",
#         "customer_expectation",
#     ],
# }



