import re
import wordninja
import string


def replace_sequence_letters_to_words_str(input_text):
    """
    make the seperated letters to words, by word ninja. REPLACE ALL OCCURRENCE.
    :param input_text: the whole text should replace the s e p l e t t e r s to words.
        Example: "l o w i n g the salary w i l l m a k e t h e worker a r g u i n g t h e c o m p a n y."

    """

    def _lambda_replace(match):
        # remove spaces within each match
        without_spaces = match.group().replace(' ', '')
        # split the resulting string back into words with WordNinja
        # add an extra ' ' to given back the ' ' which occupied
        corrected = ' '.join(wordninja.split(without_spaces)) + ' '
        # return the corrected string
        return corrected

    newtext = re.sub(
        # greedy method will make it match as much as possible(letters)
        # ?!\d to avoid mismatch \d.\d etc.
        rf'(\b\w((\b\s)|(\b$)|([{re.escape(string.punctuation)}](?!\d)\s*)))+',
        # r'((?:\b\w\b\s)+)',
        _lambda_replace,
        input_text,
        flags=re.IGNORECASE
    )

    return newtext
