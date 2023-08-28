import langdetect
import pathos
import tqdm
from .. import _BasicKits


def detect_seq_language_list(sentence_list, loop_method: str = 'raw', result_type: str = 'max'):
    """
    This function can help you find what language of a texture is.
    :param sentence_list: the sentence list to be define the language
    :param loop_method: raw/mp the method to loop,default <forloop>
    :param result_type: max/all/max_prob/all_prob,
        max:only max one, all: all observes, prob: return probs,
        <default max>
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

    if loop_method == 'raw':
        for sen in tqdm.tqdm(sentence_list):
            rst_list.append(_lambda_detect_language(sen))

    elif loop_method == 'mp':
        with pathos.multiprocessing.Pool(
                processes=20,
                initializer=_BasicKits._BasicFuncT.processes_interrupt_initiator
        ) as pool:
            for rst in tqdm.tqdm(pool.imap(_lambda_detect_language, iterable=sentence_list), total=len(sentence_list)):
                rst_list.append(rst)
    else:
        raise ValueError('the choice method not provided, only raw/mp could be used')

    return rst_list
