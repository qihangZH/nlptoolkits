import typing

from . import _ChatCompletionT

from ._ChatCompletionT import chatcompletion_worker


def classify_single_task_res(
        notion_text: str,
        task_text,
        choices_seq,
        statement_serdf,
        target_col,
        identifier_col,
        openai_apikey,
        chunksize: typing.Optional[int] = None,
        stop_text='#S_#T#O#P_#',
        message_generate_func: typing.Callable = _ChatCompletionT.default_message_generator,
        ratelimit_call: int = 1000,
        ratelimit_period: int = 60,
        backoff_max_tries: int = 8,
        result_type: typing.Literal['raw', 'df'] = 'df',
        dataframe_format_error: typing.Literal['skip', 'raise'] = 'raise',
        dataframe_deficiency_error: typing.Literal['warn', 'ignore', 'raise'] = 'raise',
        is_dataframe_errors_onebyone_retry: bool = False,
        base_url: typing.Optional[str] = None,
        **kwargs):
    """
    Remember-> If you want to make a response which have no any choices, make choice Null
    :param notion_text: the background information or the question context the AI/GPT should know
    :param task_text: the question you would like GPT to anwser
    :param choices_seq: the choices from which GPT choose the answer, should be an array-like/sequence
    :param statement_serdf: a DataFrame/Series(Pandas) to input the question
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

    # supplement the default values

    def _lambda_prompt_generator(dict_list):
        return f"""
            Context: {notion_text}, use list of input dict and statements in different dict seperately->
                from #*%Statements#*% I give to you.
            The <STATEMENT-VALUE> used for your analysis :  {{{target_col}:<STATEMENT-VALUE>}}
            If choices are given, you have to prefer pick from given choices, ELSE PLEASE determine by your self,
            Here are Question(task) and their corresponding Possible choices which used for response(ignore choices if nothing follow): 
                Question:{task_text}, Possible choices: {choices_seq}
            Rules:
            - Answer the question, use choices if given else give your own idea in free, return format is below.
            - Final Answer using JSON in the following format:
                [{{"response" : YOUR_RESPONSE_TO_THE_QUESTION(first dict statement), 
                    "{identifier_col}": <origin first dict {identifier_col} value >}}, 
                 {{"response" : YOUR_RESPONSE_TO_THE_QUESTION(second dict statement, if it exist), 
                    "{identifier_col}": <origin second dict {identifier_col} value if it exist>}}, 
                 ..and so on]{stop_text}
                 
            #*%Statements#*%: read statement one by one(in different dict) and answer them->
            {dict_list}
            JSON =
            """

    return _ChatCompletionT.chatcompletion_worker(
        prompt_generator_func=_lambda_prompt_generator,
        statement_df=statement_serdf,
        target_col=target_col,
        identifier_col=identifier_col,
        openai_apikey=openai_apikey,
        chunksize=chunksize,
        stop_text=stop_text,
        message_generate_func=message_generate_func,
        ratelimit_call=ratelimit_call,
        ratelimit_period=ratelimit_period,
        backoff_max_tries=backoff_max_tries,
        result_type=result_type,
        dataframe_format_error=dataframe_format_error,
        dataframe_deficiency_error=dataframe_deficiency_error,
        is_dataframe_errors_onebyone_retry=is_dataframe_errors_onebyone_retry,
        base_url=base_url,
        **kwargs
    )


def classify_multi_task_dictres(
        notion_text: str,
        tasks_choices_dict: dict,
        statement_serdf,
        target_col,
        identifier_col,
        openai_apikey,
        chunksize: typing.Optional[int] = None,
        stop_text='#S_#T#O#P_#',
        message_generate_func: typing.Callable = _ChatCompletionT.default_message_generator,
        ratelimit_call: int = 1000,
        ratelimit_period: int = 60,
        backoff_max_tries: int = 8,
        result_type: typing.Literal['raw', 'df'] = 'df',
        dataframe_format_error: typing.Literal['skip', 'raise'] = 'raise',
        dataframe_deficiency_error: typing.Literal['warn', 'ignore', 'raise'] = 'raise',
        is_dataframe_errors_onebyone_retry: bool = False,
        base_url: typing.Optional[str] = None,
        **kwargs) -> tuple:
    """
    Remember-> If you want to make a response which have no any choices, make choice Null

    :param notion_text: the context which GPT should pre-know
    :param tasks_choices_dict: dict which contains the mapping from your question/task -> possible choices
                              by which GPT can choose from.
    :param statement_serdf: a DataFrame/Series(Pandas) to input the question
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
    :return: a tuple contains dictionary of (Question i mapping to exact info,
                                            result contains the id/classify result)
    """

    # supplement the default values

    qid_qkey_qvalue_tuple_list = [(i, key, values)
                                  for i, (key, values) in enumerate(tasks_choices_dict.items())
                                  ]
    task_choices_prompt_text = '\n\t'.join([f"Question{i}:{key}, Possible choices: {values}"
                                            for i, key, values in qid_qkey_qvalue_tuple_list
                                            ]
                                           ) + '\n'
    task_first_choice_return_format = ', '.join([f'"Question{i}": <YOUR-RESPONSE-OF-FIRST-STATEMENT-DICT>'
                                                 for i, key, values in qid_qkey_qvalue_tuple_list
                                                 ]
                                                )
    task_second_choice_return_format = ', '.join([f'"Question{i}": <YOUR-RESPONSE-OF-SECOND-STATEMENT-DICT-IF-EXIST>'
                                                  for i, key, values in qid_qkey_qvalue_tuple_list
                                                  ]
                                                 )

    def _lambda_prompt_generator(dict_list):
        return f"""
        Context: {notion_text}, use list of input dict and statements in different dict seperately->
            from #*%Statements#*% I give to you.
        The <STATEMENT-VALUE> used for your analysis : {{{target_col}:<STATEMENT-VALUE>}}
        If choices are given, you have to prefer pick from given choices, ELSE PLEASE determine by your self,
        Here are Question(task) and their corresponding Possible choices which used for response(ignore choices if nothing follow): 
            {task_choices_prompt_text}
        Rules:
        - Answer all questions, use choices if given else give your own idea in free, return format is below. 
        - Final Answer using JSON in the following format:
            [{{{task_first_choice_return_format}, 
                "{identifier_col}": <origin first dict {identifier_col} value >}}, 
             {{{task_second_choice_return_format}, 
                "{identifier_col}": <origin second dict {identifier_col} value if-exist>}}, 
             ..(and so on)]{stop_text}
             
        #*%Statements#*%: read statement one by one(in different dict)
        {dict_list}
        JSON =
        """

    return {f'Question{i}': key
            for i, key, values in qid_qkey_qvalue_tuple_list
            }, _ChatCompletionT.chatcompletion_worker(
        prompt_generator_func=_lambda_prompt_generator,
        statement_df=statement_serdf,
        target_col=target_col,
        identifier_col=identifier_col,
        openai_apikey=openai_apikey,
        chunksize=chunksize,
        stop_text=stop_text,
        message_generate_func=message_generate_func,
        ratelimit_call=ratelimit_call,
        ratelimit_period=ratelimit_period,
        backoff_max_tries=backoff_max_tries,
        result_type=result_type,
        dataframe_format_error=dataframe_format_error,
        dataframe_deficiency_error=dataframe_deficiency_error,
        is_dataframe_errors_onebyone_retry=is_dataframe_errors_onebyone_retry,
        base_url=base_url,
        **kwargs
    )
