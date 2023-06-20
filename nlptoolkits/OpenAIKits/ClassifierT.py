import openai
import json


class CompletionClassifier:

    def __init__(self,
                 openai_apikey,
                 task_text,
                 classify_rules_class_dict,
                 stop_text='#S_#T#O#P_#',
                 system_message="You are a serious research assistant who"
                                " follows exact instructions and returns only valid JSON."
                 ):
        self.openai_apikey = openai_apikey

        self.task_text = task_text

        self.classify_rules_class_dict = classify_rules_class_dict

        self.stop_text = stop_text

        self.system_message = system_message

        openai.api_key = openai_apikey

    def _generate_prompt(self, statements, classify_key, identifier_key):
        return f"""
            Task: {self.task_text}, use list of input dict->#Statements#
            How to Classify: {self.classify_rules_class_dict}, pick choice from these values
            The <STATEMENT-VALUE> used for classify: {{{classify_key}:<STATEMENT-VALUE>}}
            Rules:
            - Make a new key-value pair to old input dict:{{"class" : CHOICE}}.
            - Final Answer using JSON in the following format:
                [{{"class" : CHOICE, "{identifier_key}": <origin {identifier_key} value >}}, ..]{self.stop_text}
            #Statements#:
            {statements}
            JSON =
            """

    def classify_list(self, text_dict_list, classify_key, identifier_key, **kwargs) -> list:
        """
        :param text_dict_list: a list which contains the dict of statements
        :param classify_key: for each dict, the statement key to point out to use it to classify
        :param kwargs: any other parameters of openai.ChatCompletion.create
        """
        # supplement the default values
        kwargs['model'] = kwargs['model'] if 'model' in kwargs else "gpt-3.5-turbo"
        kwargs['temperature'] = kwargs['temperature'] if 'temperature' in kwargs else 0

        prompt = self._generate_prompt(text_dict_list, classify_key, identifier_key)

        completion = openai.ChatCompletion.create(
            stop=[self.stop_text],
            messages=[
                {"role": "system", "content": self.system_message},
                {"role": "user", "content": prompt}
            ],
            **kwargs
        )

        result = dict(completion["choices"][0]["message"])
        prediction = json.loads(result["content"])

        return prediction
