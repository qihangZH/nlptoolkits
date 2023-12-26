# The arguments.

# --------------------------------------------------------------------------
# Annotations
# --------------------------------------------------------------------------
DEFAULT_NER_TAG_LABEL: str = "NER"

DEFAULT_POS_TAG_LABEL: str = "POS"

DEFAULT_SENTIMENT_TAG_LABEL: str = "SENTIMENT"

DEFAULT_COMPOUNDING_SEP_STRING: str = "[SEP]"

DEFAULT_TOKEN_SEP_STRING: str = " "

# --------------------------------------------------------------------------
# DEFAULT ARGUMENT: AnnotatedLineCleaner
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# Flags: AnnotatedLineCleaner
# --------------------------------------------------------------------------

ANNOTATED_LINE_CLEANER_FLAGS = {
    "cleaned_line_only": 0,
    "sentiment_only": 1,
}

# CLEANEDLINE = 0, return cleaned line only, sentiment or other annotations are removed
FLAG_ANNOTATED_LINE_CLEANER_CLEANEDLINE = ANNOTATED_LINE_CLEANER_FLAGS["cleaned_line_only"]
# SENTIMENT = 1, return sentiment only, cleaned line and other annotations are removed
FLAG_ANNOTATED_LINE_CLEANER_SENTIMENT = ANNOTATED_LINE_CLEANER_FLAGS["sentiment_only"]

# --------------------------------------------------------------------------
# POS penn tree bank tags, all tags
# https://erwinkomen.ruhosting.nl/eng/2014_Longdale-Labels.htm
# --------------------------------------------------------------------------

POS_PENN_TREE_BANK_TAGS_UPPER_SET = {
    'ADJP', 'ADVP', 'CC', 'CD', 'CONJP', 'DT', 'EX', 'FRAG', 'FW', 'IN', 'INTJ',
    'JJ', 'JJR', 'JJS', 'LS', 'LST', 'MD', 'NAC', 'NN', 'NNS', 'NNP', 'NNPS',
    'NP', 'NX', 'PDT', 'POS', 'PP', 'PRN', 'PRP', 'PRPS', 'PRT', 'QP', 'RB',
    'RBR', 'RBS', 'RP', 'RRC', 'S', 'SBAR', 'SINV', 'SQ', 'SYM', 'TO', 'UCP',
    'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'VP', 'WDT', 'WHADJP', 'WHADVP',
    'WHNP', 'WHPP', 'WP', 'WPS', 'WRB', 'X', '.', ',', ':', ';', '?', '!', 'LRB', 'RRB', 'SQT', 'EQT',
    # Additional punctuation and special characters
    "-LRB-", "-RRB-", "-LSB-", "-RSB-", "-LCB-", "-RCB-", "``", "''", "`", "'", '"', "(", ")", "[", "]", "{", "}",
    "HYPH", "'S"
}

POS_PENN_TREE_BANK_TAGS_PUNCT_UPPER_SET = {
    '.', ',', ':', ';', '?', '!', 'LRB', 'RRB', 'SQT', 'EQT',
    # Additional punctuation and special characters
    "-LRB-", "-RRB-", "-LSB-", "-RSB-", "-LCB-", "-RCB-", "``", "''", "`", "'", '"', "(", ")", "[", "]", "{", "}",
    "HYPH", "'S"
}

# --------------------------------------------------------------------------
# STANFORD CORENLP NER tags in English, 12 tags
# https://stanfordnlp.github.io/CoreNLP/ner.html
# --------------------------------------------------------------------------

ALL_TAGS_FLAG = 'all'

STANFORD_CORENLP_NER_TAGS_UPPER_SET = {
    'PERSON', 'LOCATION', 'ORGANIZATION', 'MISC',    # Named entities
    'MONEY', 'NUMBER', 'ORDINAL', 'PERCENT',         # Numerical entities
    'DATE', 'TIME', 'DURATION', 'SET'                # Temporal entities
}
