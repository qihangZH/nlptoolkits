import json
import warnings
import openai
import pandas as pd
import tqdm
import typing
import ratelimit
import re
import backoff
from .. import _BasicKits

if int(openai.__version__[0]) < 1:
    raise ImportError('OpenAI version should be 1.0.0 or higher')


def _chatcompletion_requester(
        prompt: str,
        prompt_task_num: int,
        stop_text: str,
        result_type: typing.Literal['raw', 'json'],
        message_generate_func: typing.Callable,
        openai_client: openai.OpenAI,
        **kwargs
) -> (list, bool, bool):
    # first step, get data from chat.openai, However sometimes will meet errors.

    try:
        """v1.0+ -> update for new completion format"""
        completion = openai_client.chat.completions.create(
            messages=message_generate_func(prompt),
            stop=[stop_text],
            **kwargs
        )
    except openai.OpenAIError:
        raise
    except Exception:
        return [], True, True

    if result_type == 'raw':
        prediction_raw_list = [completion]

        return prediction_raw_list, False, False

    elif result_type == 'json':

        try:
            """v1.0+ -> update for new completion format"""
            result = dict(completion.choices[0].message)
            # result = dict(completion["choices"][0]["message"])

            if result["content"].endswith('...'):
                raise ValueError('Max token return exceed, try another model or change your strategy(chunksize)')

            # NOTE: sometimes if you require JSON mode, it may return in "JSON code mode", so must check and pick texture inside the json code window
            match = re.search(r'```json(.*?)```', result["content"], re.DOTALL)

            if match:
                json_content = match.group(1).strip()
            else:
                json_content = result["content"]

            prediction_json_list = json.loads(json_content, strict=False)

            # Check is Json Format suitable for return dataframe
            is_dataframe_format_error = not _BasicKits._BasicFuncT.check_is_list_of_dicts(prediction_json_list)

            is_dataframe_deficiency_error = len(prediction_json_list) != prompt_task_num

            return prediction_json_list, is_dataframe_format_error, is_dataframe_deficiency_error
        except:
            return [], True, True
    else:
        raise ValueError('Wrong result type input')


def _chatcompletion_dfjson_errorholder(
        prediction_json_list: list,
        is_dataframe_format_error: bool,
        is_dataframe_deficiency_error: bool,
        dataframe_format_error: typing.Literal['skip', 'raise'],
        dataframe_deficiency_error: typing.Literal['warn', 'ignore', 'raise']
) -> list:
    """
    The function to find the errors and hold them, finally return list of dict
    Args:
        prediction_json_list: ...
        is_dataframe_format_error: ...
        is_dataframe_deficiency_error: ...
        dataframe_format_error: skip/raise
        dataframe_deficiency_error: sometimes you could detect no sufficient reply from GPT,
            you could choose these method: warn/ignore/raise. 'warn' will only warn, 'ignore' do nothing

    Returns: list of dict, if no errors cast
    """
    if is_dataframe_format_error:
        if dataframe_format_error == 'skip':
            return []
        elif dataframe_format_error == 'raise':
            raise ValueError(f'the result {prediction_json_list} is not list which would cause error')
        else:
            raise ValueError('Wrong argument dataframe_format_error input')

    if is_dataframe_deficiency_error:
        if dataframe_deficiency_error == 'warn':
            warnings.warn(f'lost observations from gpt')
        elif dataframe_deficiency_error == 'ignore':
            pass
        elif dataframe_deficiency_error == 'raise':
            raise ValueError(f'lost observations from gpt')

        else:
            raise ValueError('Wrong argument dataframe_deficiency_error input')

    return prediction_json_list


def chatcompletion_worker(
        prompt_generator_func,
        statement_df,
        target_col: str,
        identifier_col: str,
        openai_apikey: str,
        chunksize: typing.Optional[int],
        stop_text: str,
        message_generate_func: typing.Callable,
        ratelimit_call: int,
        ratelimit_period: int,
        backoff_max_tries: int,
        result_type: typing.Literal['raw', 'df'],
        dataframe_format_error: typing.Literal['skip', 'raise'],
        dataframe_deficiency_error: typing.Literal['warn', 'ignore', 'raise'],
        is_dataframe_errors_onebyone_retry: bool,
        base_url: typing.Optional[str],
        **kwargs):
    """
    The main worker to get result from openai ChatGpt, Wrapped on several components to get result.
    the prompt send to openai is generated by **message_generate_func(prompt_generator_func(sub_prompt_dict_list))**
    sub_prompt_dict_list is a list of dict, each dict contains the target_col and identifier_col
    :param prompt_generator_func: the function to preprocess the target col converted dict_list, must return a list of json.
        SAMPLE: see ClassifierT.classify_single_task or ..multi_task, you can also write your own prompt generator.
    :param statement_df: a DataFrame/Series(Pandas) to input the question
    :param target_col: for each dict, the statement key to point out to use it to classify
    :param identifier_col: the col which takes the id func to know the each observations
    :param openai_apikey: the api key of openai
    :param chunksize: how many input to put in per session, default None
    :param stop_text: stop text which indicate the endpoint to save out, default '#S_#T#O#P_#'
    :param message_generate_func: the message generate function, input prompt and return a message,
            the argument showed in https://platform.openai.com/docs/api-reference/chat/create
            The function input ONLY the prompt like lambda prompt: [...message]
    :param ratelimit_call: the ratelimit's call times per period
    :param ratelimit_period: the period seconds
    :param backoff_max_tries: If meet errors, How much retry times? use backoff.expo to control the retry.
    :param result_type, raw/df, how would GPT return the result, raw is for debugger or specialize using.
    :param dataframe_format_error: 'warn', 'ignore', 'raise'
            how to deal with dataframe error when you choose 'df'/'chunksize'
    :param dataframe_deficiency_error: 'warn', 'ignore', 'raise'
            sometimes you could detect no sufficient reply from
    :param is_dataframe_errors_onebyone_retry: Do retry one by one when meets error? It will start when
            the code find format/deficiency errors in loop, and retry again, whatever your error-dealing method
            is.
    :param base_url: If you choose different service other than openai, like gemini or deekseek, then change here.
        Use None or other Nonetype input if you do not have other service provider.
    :param kwargs: any other parameters of openai.ChatCompletion.create
    :return:which contains the id/classify result
    """

    @backoff.on_exception(backoff.expo,
                          (ratelimit.RateLimitException, openai.OpenAIError),
                          max_tries=backoff_max_tries)
    @ratelimit.limits(calls=ratelimit_call, period=ratelimit_period)
    def _lambda_backoff_chatcompletion_requester(*largs, **lkwargs):
        return _chatcompletion_requester(*largs, **lkwargs)

    # --------------------------------------------------------------------------
    # Argument Prepare
    # --------------------------------------------------------------------------
    """v1.0+ -> use client"""
    # openai.api_key = openai_apikey

    if base_url:
        openai_client = openai.OpenAI(api_key=openai_apikey, base_url=base_url)
    else:
        openai_client = openai.OpenAI(api_key=openai_apikey)

    # How does finalrst need completionlist's type? a one to one dict
    finalrst_completionlist_type_dict = {'df': 'json', 'raw': 'raw'}

    kwargs['model'] = kwargs['model'] if 'model' in kwargs else "gpt-3.5-turbo"
    kwargs['temperature'] = kwargs['temperature'] if 'temperature' in kwargs else 0

    # Check the format is surely pd.DataFrame
    if isinstance(statement_df, pd.DataFrame):
        statement_df = statement_df[[target_col, identifier_col]]
        prompt_dict_list = statement_df.to_dict(orient='records')
    else:
        raise ValueError('Input must be pd.DataFrame')

    # we have to modify chunksize to suitable for one-time-running:
    chunksize = len(prompt_dict_list) if not chunksize else chunksize

    # --------------------------------------------------------------------------
    # Loop start, however if no chunksize the loop will be only one.
    # --------------------------------------------------------------------------
    prediction_list = []
    for chunkslice in tqdm.tqdm(range(0, len(prompt_dict_list), chunksize)):
        sub_prompt_dict_list = prompt_dict_list[chunkslice: chunkslice + chunksize]

        completion_list, is_dataframe_format_error, is_dataframe_deficiency_error = \
            _lambda_backoff_chatcompletion_requester(
                prompt=prompt_generator_func(sub_prompt_dict_list),
                prompt_task_num=len(sub_prompt_dict_list),
                stop_text=stop_text,
                # This place are different for this function actually got json
                result_type=finalrst_completionlist_type_dict[result_type],
                message_generate_func=message_generate_func,
                openai_client=openai_client,
                **kwargs
            )

        if result_type == 'raw':
            prediction_list += completion_list

        elif result_type == 'df':
            # If you set retry, and any problems are find
            if is_dataframe_errors_onebyone_retry and (is_dataframe_format_error or is_dataframe_deficiency_error):
                sub_pred_list = []
                for prpdict in sub_prompt_dict_list:
                    compl, formaterr, deficerr = _lambda_backoff_chatcompletion_requester(
                        prompt=prompt_generator_func([prpdict]),
                        prompt_task_num=1,
                        stop_text=stop_text,
                        # This place are different for this function actually got json
                        result_type=finalrst_completionlist_type_dict[result_type],
                        message_generate_func=message_generate_func,
                        openai_client=openai_client,
                        **kwargs
                    )
                    sub_pred_list += _chatcompletion_dfjson_errorholder(
                        prediction_json_list=compl,
                        is_dataframe_format_error=formaterr,
                        is_dataframe_deficiency_error=deficerr,
                        dataframe_format_error=dataframe_format_error,
                        dataframe_deficiency_error=dataframe_deficiency_error
                    )

                prediction_list += sub_pred_list
            else:
                prediction_list += _chatcompletion_dfjson_errorholder(
                    prediction_json_list=completion_list,
                    is_dataframe_format_error=is_dataframe_format_error,
                    is_dataframe_deficiency_error=is_dataframe_deficiency_error,
                    dataframe_format_error=dataframe_format_error,
                    dataframe_deficiency_error=dataframe_deficiency_error
                )

        else:
            raise ValueError('Wrong result type input for')

    if result_type == 'raw':
        return prediction_list

    elif result_type == 'df':
        return pd.DataFrame(prediction_list)


def default_message_generator(prp):
    """prototype for you to refer and make message_generate_func"""
    return [
        {"role": "system", "content": "You are a serious research assistant who"
                                      " follows exact instructions and returns only valid JSON."},
        {"role": "user", "content": prp}
    ]
