#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 23 21:18:19 2020

@author: markusschwedeler
"""

from bs4 import BeautifulSoup, FeatureNotFound
import os
import re
import pandas as pd


# Note: need to install html5lib parser


def import_sentimentwords(file):
    df = pd.read_csv(file, sep=',')
    tokeep = ['Word', 'Positive']
    positive = set([x['Word'].lower() for idx, x in df[tokeep].iterrows()
                    if x['Positive'] > 0])
    tokeep = ['Word', 'Negative']
    negative = set([x['Word'].lower() for idx, x in df[tokeep].iterrows()
                    if x['Negative'] > 0])
    return {'positive': positive, 'negative': negative}


def import_riskwords(file):
    synonyms = set()
    with open(file, 'r') as inp:
        for line in inp:
            split = line.split(' ')
            for syn in split:
                synonyms.add(re.sub('\n', '', syn))
    return synonyms


def import_politicalbigrams(file):
    df = pd.read_csv(file, sep=',', encoding='utf-8')
    df = df.assign(bigram=df['bigram'].str.replace('_', ' '))
    df.rename(columns={'politicaltbb': 'tfidf'}, inplace=True)
    df.set_index('bigram', inplace=True)
    return df.to_dict(orient='index')