from typing import Set
from pathlib import Path
import os

_DIR_RESOURCES = os.path.abspath(os.path.dirname(__file__)) + '/'
# The package include the resources
# The stopwords which used in Stanza...
SET_STOPWORDS: Set[str] = set(
    Path(_DIR_RESOURCES, "StopWords_Generic.txt").read_text().lower().split() +
    Path(_DIR_RESOURCES, "StopWords_Generic.txt").read_text().split()
)
