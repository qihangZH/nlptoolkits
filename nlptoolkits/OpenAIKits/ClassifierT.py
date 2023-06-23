import typing

from . import _Models


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
        system_message="You are a serious research assistant who"
                       " follows exact instructions and returns only valid JSON.",
        result_type: typing.Literal['raw', 'df'] = 'df',
        dataframe_format_error: typing.Literal['raise', 'skip'] = 'raise',
        dataframe_deficiency_error: typing.Literal['warn', 'ignore', 'onebyone', 'raise'] = 'warn',
        **kwargs):
    """
    Remember-> If you want to make a response which have no any choices, make choice Null
    :param notion_text: the background information or the question context the AI/GPT should know
    :param task_text: the question you would like GPT to anwser
    :param choices_seq: the choices from which GPT choose the answer, should be an array-like/sequence
    :param statement_serdf: a DataFrame/Series(Pandas) to input the question
    :param target_col: for each dict, the statement key to point out to use it to classify
    :param openai_apikey: the api key of openai
    :param chunksize: how many input to put in per session, default None
    :param stop_text: stop text which indicate the endpoint to save out, default '#S_#T#O#P_#'
    :param system_message: the message to indicate the GPT act with what person
    :param result_type, raw/df, default df how would GPT return the result, raw is for debugger or specialize using.
    :param dataframe_format_error: raise/skip default raise, how to deal with dataframe error when you choose 'df'
    :param dataframe_deficiency_error: if you use 'df'/'chunksize', sometimes you could detect no sufficient reply from
            GPT, you could choose these method: default 'warn', warn/ignore/onebyone/raise
            'warn' will only warn, ignore do nothing,
             onebyone will retry one by one and return, but cost more money. If fail, return origin result.
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

    return _Models.chatcompletion_worker(
        prompt_generator_func=_lambda_prompt_generator,
        statement_serdf=statement_serdf,
        target_col=target_col,
        identifier_col=identifier_col,
        openai_apikey=openai_apikey,
        chunksize=chunksize,
        stop_text=stop_text,
        system_message=system_message,
        result_type=result_type,
        dataframe_format_error=dataframe_format_error,
        dataframe_deficiency_error=dataframe_deficiency_error,
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
        system_message="You are a serious research assistant who"
                       " follows exact instructions and returns only valid JSON.",
        result_type: typing.Literal['raw', 'df'] = 'df',
        dataframe_format_error: typing.Literal['raise', 'skip'] = 'raise',
        dataframe_deficiency_error: typing.Literal['warn', 'ignore', 'onebyone', 'raise'] = 'warn',
        **kwargs) -> tuple:
    """
    Remember-> If you want to make a response which have no any choices, make choice Null

    :param notion_text: the context which GPT should pre-know
    :param tasks_choices_dict: dict which contains the mapping from your question/task -> possible choices
                              by which GPT can choose from.
    :param statement_serdf: a DataFrame/Series(Pandas) to input the question
    :param target_col: for each dict, the statement key to point out to use it to classify
    :param openai_apikey: the api key of openai
    :param chunksize: how many input to put in per session, default None
    :param stop_text: stop text which indicate the endpoint to save out, default '#S_#T#O#P_#'
    :param system_message: the message to indicate the GPT act with what person
    :param result_type, raw/df, default df how would GPT return the result, raw is for debugger or specialize using.
    :param dataframe_format_error: raise/skip default raise, how to deal with dataframe error when you choose 'df'
    :param dataframe_deficiency_error: if you use 'df'/'chunksize', sometimes you could detect no sufficient reply from
            GPT, you could choose these method: default 'warn', warn/ignore/onebyone/raise
            'warn' will only warn, ignore do nothing,
             onebyone will retry one by one and return, but cost more money. If fail, return origin result.
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
            }, _Models.chatcompletion_worker(
        prompt_generator_func=_lambda_prompt_generator,
        statement_serdf=statement_serdf,
        target_col=target_col,
        identifier_col=identifier_col,
        openai_apikey=openai_apikey,
        chunksize=chunksize,
        stop_text=stop_text,
        system_message=system_message,
        result_type=result_type,
        dataframe_format_error=dataframe_format_error,
        dataframe_deficiency_error=dataframe_deficiency_error,
        **kwargs
    )
