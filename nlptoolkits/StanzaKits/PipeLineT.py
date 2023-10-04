import os
import warnings
import warnings
import stanza
import tqdm
import pathos
import math
import torch
import typing
from .. import _BasicKits


class PipeLineParser:

    def __init__(self,
                 doc_list: list,
                 doc_ids_list: typing.Optional[list]=None,
                 thread_num: int = os.cpu_count(),
                 show_progressbar: bool = True,
                 gpu_process_chunksize: typing.Optional[int] = None,
                 **kwargs):
        """
        More infos see: https://stanfordnlp.github.io/stanza/neural_pipeline.html
        Args:
            doc_list:
            thread_num:
            show_progressbar:
            gpu_process_chunksize:
            **kwargs:
        """
        kwargs['lang'] = kwargs['lang'] if 'lang' in kwargs else 'en'
        kwargs['processors'] = kwargs['processors'] if 'processors' in kwargs else 'tokenize'
        kwargs['download_method'] = kwargs['download_method'] if 'download_method' in kwargs \
            else stanza.DownloadMethod.REUSE_RESOURCES
        kwargs['use_gpu'] = kwargs['use_gpu'] if 'use_gpu' in kwargs else True

        if not doc_ids_list:
            warnings.warn('No ids are given, however, some attributes are based on ids, and clearly ids '
                          'could help you to have index')

        if not ('tokenize' in kwargs['processors']):
            raise NotImplementedError('The PipeLineParser only support tokenize-series workflow, '
                                      'if not contains, then do not use this.'
                                      'Multiling also not support')

        if not _BasicKits._BasicFuncT.check_is_list_of_string(doc_list):
            raise NotImplementedError('The PipeLineParser only support doc_list like ["...","...",...]')

        self.gpu_process_chunksize = gpu_process_chunksize if gpu_process_chunksize else 1

        self.thread_num = thread_num

        self.doc_list = doc_list

        self.show_progressbar = show_progressbar

        if kwargs['use_gpu'] and torch.cuda.is_available():
            print('Detect GPU(cuda), use gpu-version instead')
            self.parsed_docs = self._gpu_process(**kwargs)
        else:
            self.parsed_docs = self._multithread_process(**kwargs)

    def _gpu_process(self, **kwargs):

        nlp = stanza.Pipeline(**kwargs)

        task_len = len(self.doc_list)

        task_chunked_list = [
            self.doc_list[s: s + self.gpu_process_chunksize]
            for s in range(0, task_len, self.gpu_process_chunksize)
        ]

        parsed_doc_list = []
        _iterator = tqdm.tqdm(task_chunked_list) if self.show_progressbar else task_chunked_list
        for taskl in _iterator:
            parsed_doc_list.extend(
                nlp.bulk_process(taskl)
            )

        return parsed_doc_list

    def _multithread_process(self, **kwargs):

        def _lambda_parser_worker(taskl):
            nlp = stanza.Pipeline(**kwargs)

            out_list = []
            _iterator = tqdm.tqdm(taskl) if self.show_progressbar else taskl
            for doc in _iterator:
                out_list.append(nlp(doc))

            return out_list

        task_len = len(self.doc_list)

        chunksize = math.ceil(task_len / self.thread_num)

        task_chunked_list = [
            self.doc_list[s: s + chunksize]
            for s in range(0, task_len, chunksize)
        ]

        parsed_doc_list = []
        with pathos.multiprocessing.Pool(processes=self.thread_num,
                                         initializer=_BasicKits._BasicFuncT.processes_interrupt_initiator
                                         ) as pool:

            for r in pool.imap(func=_lambda_parser_worker, iterable=task_chunked_list):
                parsed_doc_list.extend(r)

        return parsed_doc_list

    # --------------------------------------------------------------------------
    # Document info
    # --------------------------------------------------------------------------
    @property
    def raw_docs(self):
        return [
            d.text
            for d in self.parsed_docs
        ]

    def attr_docs(self, attr: str):
        """
        Args:
            attr: attribute of docs in: https://stanfordnlp.github.io/stanza/data_objects.html#document

        Returns: list of docs and their attrs

        """
        return [
            getattr(d, attr)
            for d in self.parsed_docs
        ]

    # --------------------------------------------------------------------------
    # Sentences Info
    # --------------------------------------------------------------------------
    @property
    def raw_sentences(self):
        """
        Returns: A list which contains list of sentences of each document.
        """
        return [
            [
                s.text
                for s in self.parsed_docs[idx].sentences
            ]
            for idx in range(len(self.parsed_docs))
        ]

    def attr_sentences(self, attr: str):
        """
        Args:
            attr: attribute of sentences in: https://stanfordnlp.github.io/stanza/data_objects.html#sentence

        Returns: list of sentences and their attrs

        """
        return [
            [
                getattr(s, attr)
                for s in self.parsed_docs[idx].sentences
            ]
            for idx in range(len(self.parsed_docs))
        ]

    def dumptokens_sentences(self):
        """
        dump the list of dict which contains their tokens infos
        """
        return [
            [
                s.to_dict()
                for s in self.parsed_docs[idx].sentences
            ]
            for idx in range(len(self.parsed_docs))
        ]
