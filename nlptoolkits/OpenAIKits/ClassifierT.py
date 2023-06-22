import openai
import json
import tqdm
import pandas as pd
import typing


def _chatcompletion_classifier_df(
        prompt_generator_func,
        statement_serdf,
        target_col,
        identifier_col,
        openai_apikey,
        chunksize,
        stop_text,
        system_message,
        **kwargs) -> pd.DataFrame:
    """
    private function for code reuse, not recommend for outside using
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

    if (not (target_col in statement_serdf.columns)) or (not (identifier_col in statement_serdf.columns)):
        raise ValueError('target_col/identifier_col must in statement_serdf columns!')

    if not (chunksize is None):
        prediction_list = []
        for chunkslice in tqdm.tqdm(range(0, len(prompt_dict_list), chunksize)):
            sub_prompt_dict_list = prompt_dict_list[chunkslice: chunkslice + chunksize]

            prompt = prompt_generator_func(sub_prompt_dict_list)

            print(prompt)

            completion = openai.ChatCompletion.create(
                stop=[stop_text],
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt}
                ],
                **kwargs
            )

            result = dict(completion["choices"][0]["message"])

            if result["content"].endswith('...'):
                raise ValueError('Max token return exceed, try another model or change your strategy(chunksize)')

            sub_prediction_list = json.loads(result["content"])

            prediction_list += sub_prediction_list
    else:
        prompt = prompt_generator_func(dict_list=prompt_dict_list)

        completion = openai.ChatCompletion.create(
            stop=[stop_text],
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ],
            **kwargs
        )

        result = dict(completion["choices"][0]["message"])

        if result["content"].endswith('...'):
            raise ValueError('Max token return exceed, try another model or change your strategy(chunksize)')

        prediction_list = json.loads(result["content"])

    return pd.DataFrame(prediction_list)


def classify_single_task_df(task_text,
                            task_choices_seq,
                            statement_serdf,
                            target_col,
                            identifier_col,
                            openai_apikey,
                            chunksize: typing.Optional[int] = None,
                            stop_text='#S_#T#O#P_#',
                            system_message="You are a serious research assistant who"
                                           " follows exact instructions and returns only valid JSON.",
                            **kwargs) -> pd.DataFrame:
    """
    :param task_text: the task which GPT should pre-know
    :param task_choices_seq: the choices from which GPT choose the answer, should be an array-like/sequence
    :param statement_serdf: a DataFrame/Series(Pandas) to input the question
    :param target_col: for each dict, the statement key to point out to use it to classify
    :param openai_apikey: the api key of openai
    :param chunksize: how many input to put in per session, default None
    :param stop_text: stop text which indicate the endpoint to save out, default '#S_#T#O#P_#'
    :param system_message: the message to indicate the GPT act with what person
    :param kwargs: any other parameters of openai.ChatCompletion.create
    :return: pd.DataFrame contains the id/classify result
    """

    # supplement the default values

    def _lambda_prompt_generator(dict_list):
        return f"""
            Task: {task_text}, use list of input dict->#Statements#
            Task-Question-Choices: {task_choices_seq}
            The <STATEMENT-VALUE> used for classify: {{{target_col}:<STATEMENT-VALUE>}}
            Rules:
            - Make a new key-value pair to old input dict:{{"class" : CHOICE}}.
            - Final Answer using JSON in the following format:
                [{{"class" : YOUR_CHOICE, "{identifier_col}": <origin {identifier_col} value >}}, ..]{stop_text}
            #Statements#:
            {dict_list}
            JSON =
            """

    return _chatcompletion_classifier_df(
        prompt_generator_func=_lambda_prompt_generator,
        statement_serdf=statement_serdf,
        target_col=target_col,
        identifier_col=identifier_col,
        openai_apikey=openai_apikey,
        chunksize=chunksize,
        stop_text=stop_text,
        system_message=system_message,
        **kwargs
    )


def classify_multi_task_dictdf(notion_text: str,
                               tasks_choices_dict: dict,
                               statement_serdf,
                               target_col,
                               identifier_col,
                               openai_apikey,
                               chunksize: typing.Optional[int] = None,
                               stop_text='#S_#T#O#P_#',
                               system_message="You are a serious research assistant who"
                                              " follows exact instructions and returns only valid JSON.",
                               **kwargs) -> tuple:
    """
    :param notion_text: the context which GPT should pre-know
    :param tasks_choices_dict: dict which contains the mapping from your question/task -> possible choices
                              by which GPT can choose from.
    :param statement_serdf: a DataFrame/Series(Pandas) to input the question
    :param target_col: for each dict, the statement key to point out to use it to classify
    :param openai_apikey: the api key of openai
    :param chunksize: how many input to put in per session, default None
    :param stop_text: stop text which indicate the endpoint to save out, default '#S_#T#O#P_#'
    :param system_message: the message to indicate the GPT act with what person
    :param kwargs: any other parameters of openai.ChatCompletion.create
    :return: a tuple contains dictionary of (Question i mapping to exact info,
                                            pd.DataFrame contains the id/classify result)
    """

    # supplement the default values

    qid_qkey_qvalue_tuple_list = [(i, key, values)
                                  for i, (key, values) in enumerate(tasks_choices_dict.items())
                                  ]
    task_choices_prompt_text = '\n\t'.join([f"Question{i}:{key}, Choices: {values}"
                                            for i, key, values in qid_qkey_qvalue_tuple_list
                                            ]
                                           ) + '\n'
    task_choice_return_format = ', '.join([f'"Question{i}": <YOUR-CHOICE>'
                                           for i, key, values in qid_qkey_qvalue_tuple_list
                                           ]
                                          )

    def _lambda_prompt_generator(dict_list):
        return f"""
        Context: {notion_text}, use list of input dict->#Statements#
        The <STATEMENT-VALUE> used for classify: {{{target_col}:<STATEMENT-VALUE>}}
        {task_choices_prompt_text}
        Rules:
        - Answer all questions, use choices if given else give your own idea in free, return format is below.
        - Final Answer using JSON in the following format:
            [{{{task_choice_return_format}, "{identifier_col}": <origin {identifier_col} value>}}, ..]{stop_text}
        #Statements#:
        {dict_list}
        JSON =
        """

    return {f'Question{i}': key
            for i, key, values in qid_qkey_qvalue_tuple_list
            }, _chatcompletion_classifier_df(
        prompt_generator_func=_lambda_prompt_generator,
        statement_serdf=statement_serdf,
        target_col=target_col,
        identifier_col=identifier_col,
        openai_apikey=openai_apikey,
        chunksize=chunksize,
        stop_text=stop_text,
        system_message=system_message,
        **kwargs
    )
