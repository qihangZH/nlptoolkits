import json
import warnings
import openai
import pandas as pd
import tqdm
import typing
from .. import _BasicKits


def chatcompletion_worker(
        prompt_generator_func,
        statement_serdf,
        target_col: str,
        identifier_col: str,
        openai_apikey: str,
        chunksize: typing.Optional[int],
        stop_text: str,
        system_message: str,
        result_type: typing.Literal['raw', 'df'],
        dataframe_format_error: typing.Literal['raise', 'skip'],
        dataframe_deficiency_error: typing.Literal['warn', 'ignore', 'onebyone', 'raise'],
        **kwargs):
    """
    :param prompt_generator_func: the function to make prompt from state dict, must return a list of json
           SAMPLE: see ClassifierT.classify_single_task or ..multi_task, you can also write your own prompt generator.
    :param statement_serdf: a DataFrame/Series(Pandas) to input the question
    :param target_col: for each dict, the statement key to point out to use it to classify
    :param openai_apikey: the api key of openai
    :param chunksize: how many input to put in per session, default None
    :param stop_text: stop text which indicate the endpoint to save out, default '#S_#T#O#P_#'
    :param system_message: the message to indicate the GPT act with what person
    :param result_type, raw/df, how would GPT return the result, raw is for debugger or specialize using.
    :param dataframe_format_error: raise/skip ,
            how to deal with dataframe error when you choose 'df'/'chunksize'
    :param dataframe_deficiency_error: if you use 'df'/'chunksize', sometimes you could detect no sufficient reply from
            GPT, you could choose these method:  warn/ignore/onebyone/raise
            'warn' will only warn, ignore do nothing,
             onebyone will retry one by one and return, but cost more money. If fail, return origin result.
    :param kwargs: any other parameters of openai.ChatCompletion.create
    :return:which contains the id/classify result
    """

    openai.api_key = openai_apikey

    kwargs['model'] = kwargs['model'] if 'model' in kwargs else "gpt-3.5-turbo"
    kwargs['temperature'] = kwargs['temperature'] if 'temperature' in kwargs else 0

    if isinstance(statement_serdf, pd.Series):
        prompt_dict_list = statement_serdf.to_dict()
    elif isinstance(statement_serdf, pd.DataFrame):
        prompt_dict_list = statement_serdf.to_dict(orient='records')
    else:
        raise ValueError('Input must be pd.Series/pd.DataFrame')

    col_list = statement_serdf.columns if isinstance(statement_serdf, pd.DataFrame) else statement_serdf.index

    if not (target_col in col_list) or not (identifier_col in col_list):
        raise ValueError('target_col/identifier_col must in statement_serdf columns!')

    if (not (chunksize is None)) and isinstance(statement_serdf, pd.DataFrame):
        prediction_list = []
        for chunkslice in tqdm.tqdm(range(0, len(prompt_dict_list), chunksize)):
            sub_prompt_dict_list = prompt_dict_list[chunkslice: chunkslice + chunksize]

            prompt = prompt_generator_func(sub_prompt_dict_list)

            completion = openai.ChatCompletion.create(
                stop=[stop_text],
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt}
                ],
                **kwargs
            )

            if result_type == 'raw':
                prediction_list += [completion]

            elif result_type == 'df':

                result = dict(completion["choices"][0]["message"])

                if result["content"].endswith('...'):
                    raise ValueError('Max token return exceed, try another model or change your strategy(chunksize)')

                sub_prediction_list = json.loads(result["content"])

                if not _BasicKits._BasicFuncT.check_is_list_of_dicts(sub_prediction_list):
                    if dataframe_format_error == 'raise':
                        raise ValueError(f'the result {sub_prediction_list} is not list which would cause error')
                    elif dataframe_format_error == 'skip':
                        print(f'{sub_prediction_list}, is not list, has been skipped')
                        continue
                    else:
                        raise ValueError('Wrong argument dataframe_format_error input')

                if len(sub_prediction_list) != len(sub_prompt_dict_list):
                    if dataframe_deficiency_error == 'warn':
                        warnings.warn(f'{len(sub_prediction_list)}!={len(sub_prompt_dict_list)}, '
                                      f'lost observations from gpt')
                    elif dataframe_deficiency_error == 'ignore':
                        pass
                    elif dataframe_deficiency_error == 'onebyone':

                        try:
                            retry_onebyone_dict_list = []

                            for rpdict in sub_prompt_dict_list:
                                p_list = [rpdict]

                                retry_p = prompt_generator_func(p_list)

                                temp_c = openai.ChatCompletion.create(
                                    stop=[stop_text],
                                    messages=[
                                        {"role": "system", "content": system_message},
                                        {"role": "user", "content": retry_p}
                                    ],
                                    **kwargs
                                )

                                temp_rst = dict(temp_c["choices"][0]["message"])

                                if temp_rst["content"].endswith('...'):
                                    raise ValueError('retry fail for tokens exceed')

                                temp_pred_list = json.loads(temp_rst["content"])

                                if not _BasicKits._BasicFuncT.check_is_list_of_dicts(temp_pred_list):
                                    raise ValueError('retry fail for not return dict list')

                                retry_onebyone_dict_list += temp_pred_list

                            if len(retry_onebyone_dict_list) != len(sub_prompt_dict_list):
                                raise ValueError('retry fail for not return full len list')

                            sub_prediction_list = retry_onebyone_dict_list

                            print('successfully retry the full result from OpenAI(one by one)')
                        except Exception as e:
                            print(e, 'use old return.')
                    elif dataframe_deficiency_error == 'raise':
                        raise ValueError(f'{len(sub_prediction_list)}!=len({sub_prompt_dict_list}), '
                                         f'lost observations from gpt')

                    else:
                        raise ValueError('Wrong argument dataframe_deficiency_error input')

                prediction_list += sub_prediction_list

            else:
                raise ValueError('Wrong result type input for')
    else:
        warnings.warn('We recommend to use chunksize, for non-chunksize could not be protected by'
                      'dataframe_format_error,dataframe_deficiency_error!')
        prompt = prompt_generator_func(dict_list=prompt_dict_list)

        completion = openai.ChatCompletion.create(
            stop=[stop_text],
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ],
            **kwargs
        )

        if result_type == 'raw':
            prediction_list = [completion]

        elif result_type == 'df':

            result = dict(completion["choices"][0]["message"])

            if result["content"].endswith('...'):
                raise ValueError('Max token return exceed, try another model or change your strategy(chunksize)')

            prediction_list = json.loads(result["content"])
        else:
            raise ValueError('Wrong result type input')

    if result_type == 'raw':
        return prediction_list

    elif result_type == 'df':
        return pd.DataFrame(prediction_list)
