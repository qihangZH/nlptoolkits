from typing import Set
from pathlib import Path
from . import _helpers
import os

_DIR_RESOURCES = os.path.abspath(os.path.dirname(__file__)) + '/'
# The package include the resources
# The stopwords which used in Stanza...
SET_STOPWORDS: Set[str] = set(
    Path(_DIR_RESOURCES, "StopWords_Generic.txt").read_text().lower().split() +
    Path(_DIR_RESOURCES, "StopWords_Generic.txt").read_text().split()
)

SET_LOUGHRAN_MCDONALD_POSITIVE_WORDS_LOWER = _helpers.import_sentimentwords(
    Path(_DIR_RESOURCES, "LoughranMcDonald_MasterDictionary_2018.csv")
)['positive']

SET_LOUGHRAN_MCDONALD_NEGATIVE_WORDS_LOWER = _helpers.import_sentimentwords(
    Path(_DIR_RESOURCES, "LoughranMcDonald_MasterDictionary_2018.csv")
)['negative']

SET_OXFORD_SYNONYMS_RISK_WORDS_LOWER = _helpers.import_riskwords(
    Path(_DIR_RESOURCES, "riskwords_synonyms.txt")
)