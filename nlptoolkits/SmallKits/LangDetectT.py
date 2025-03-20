import fast_langdetect
import langdetect
import pathos
import tqdm
import fast_langdetect
import re
from .. import _BasicKits


def detect_seq_language_list(sentence_list, result_type: str = 'max', processes=1):
    """
    This function can help you find what language of a texture is.
    :param sentence_list: the sentence list to be define the language
    :param result_type: max/all/max_prob/all_prob,
        max:only max one, all: all observes, prob: return probs,
        <default max>
    :param processes: the number of processes to be used
    :return: the result, contains the list of language
    """

    def _lambda_detect_language(sentence):

        detected_languages = langdetect.detect_langs(sentence)

        if result_type == 'max':
            return detected_languages[0].lang
        elif result_type == 'max_prob':
            return {detected_languages[0].lang: detected_languages[0].prob}
        elif result_type == 'all':
            return [detected_languages[i].lang for i in range(len(detected_languages))]
        elif result_type == 'all_prob':
            return {detected_languages[i].lang: detected_languages[i].prob for i in range(len(detected_languages))}
        else:
            raise ValueError('Error occurs for result type only be one of: max/all/max_prob/all_prob ')

    rst_list = []

    if processes <= 1:
        for sen in tqdm.tqdm(sentence_list):
            rst_list.append(_lambda_detect_language(sen))

    else:
        with pathos.multiprocessing.Pool(
                processes=processes,
                initializer=_BasicKits._BasicFuncT.processes_interrupt_initiator
        ) as pool:
            for rst in tqdm.tqdm(pool.imap(_lambda_detect_language, iterable=sentence_list), total=len(sentence_list)):
                rst_list.append(rst)
    # else:
    #     raise ValueError('the choice method not provided, only raw/mp could be used')

    return rst_list


def fast_detect_seq_language_list(sentence_list, processes=1):
    """
    This function can help you find what language of a texture is.
    :param sentence_list: the sentence list to be define the language
    :param result_type: max/all/max_prob/all_prob,
        max:only max one, all: all observes, prob: return probs,
        <default max>
    :param processes: the number of processes to be used
    :return: the result, contains the list of language
    """
    if hasattr(fast_langdetect, 'detect_lang'):
        runfunc = fast_langdetect.detect_lang
    elif hasattr(fast_langdetect, 'detect_language'):
        runfunc = fast_langdetect.detect_language
    else:
        raise ValueError('The version of fast-langdetect do not fit and can not run')

    def _lambda_detect_language(sentence):
        try:

            detected_languages = runfunc(
                re.sub(r'\s+', ' ', sentence, flags=re.IGNORECASE | re.DOTALL)
            )

            return detected_languages
        except:
            raise ValueError(sentence)

    rst_list = []

    if processes <= 1:
        for sen in tqdm.tqdm(sentence_list):
            rst_list.append(_lambda_detect_language(sen))

    else:
        with pathos.multiprocessing.Pool(
                processes=processes,
                initializer=_BasicKits._BasicFuncT.processes_interrupt_initiator
        ) as pool:
            for rst in tqdm.tqdm(pool.imap(_lambda_detect_language, iterable=sentence_list), total=len(sentence_list)):
                rst_list.append(rst)
    # else:
    #     raise ValueError('the choice method not provided, only raw/mp could be used')

    return rst_list
