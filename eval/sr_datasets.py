import os

import numpy as np
import pandas as pd
from sklearn.utils import Bunch
from collections import defaultdict as dd


def _get_as_df(name_or_path, **read_csv_kwargs):
    """
    Return the dataset as a pandas dataframe
    """
    path = os.getcwd() + '/eval/data/' + name_or_path
    return pd.read_csv(path, **read_csv_kwargs)


def get_WS353():
    """
    Fetch WS353 similarity dataset. Other versions available in the data
    folder but not currently used.
    """

    data = _get_as_df('WS353/wordsim_similarity_goldstandard.txt',
                      header=0,
                      sep="\t",
                      ).values

    return Bunch(X=data[:, 0:2].astype("object"),
                 y=data[:, 2].astype(np.float))


def get_RG65():
    """
    Rubenstein and Goodenough similarity dataset.
    """
    data = _get_as_df('EN-RG-65.txt', header=None, sep="\t").values

    return Bunch(X=data[:, 0:2].astype("object"),
                 y=data[:, 2].astype(np.float))


def get_RW():
    """
    Rare Words similarity dataset.
    """

    data = _get_as_df('EN-RW.txt', header=None, sep="\t").values

    return Bunch(X=data[:, 0:2].astype("object"),
                 y=data[:, 2].astype(np.float),
                 sd=np.std(data[:, 3:].astype(np.float)))


def get_GRU65():
    """
    Rubenstein and Goodenough similarity dataset, German version.
    65 pairs.
    """

    data = _get_as_df('GRU65.txt', header=None, sep="\t").values

    return Bunch(X=data[:, 0:2].astype("object"),
                 y=data[:, 2].astype(np.float))


def get_GRU350():
    """
    Rubenstein and Goodenough similarity dataset, German version.
    350 pairs. 
    """

    data = _get_as_df('GRU350.txt', header=None, sep="\t").values

    return Bunch(X=data[:, 0:2].astype("object"),
                 y=data[:, 2].astype(np.float))


def get_ZG222():
    """
    Zesch and Gurevych similarity dataset, German.
    222 pairs.
    """

    data = _get_as_df('ZG222.txt', header=None, sep="\t").values

    return Bunch(X=data[:, 0:2].astype("object"),
                 y=data[:, 2].astype(np.float))


def get_google_analogy(filename) -> dict:
    """
    Google word analogy dataset
    :return: dictionary of different categories of data with list of data instances
    """
    path = 'drive/MyDrive/data/' + filename +'.txt'
    line_data = open(path).read()
    L = line_data.split('\n')
    questions = []
    answers = []
    category = []
    cat = None
    for l in L:
        if l.startswith(":"):
            cat =l.lower().split()[1]
        else:
            words =  l.split()
            if words:
              questions.append(words[0:3])
              answers.append(words[3])
              category.append(cat)

    assert set(category) == set(['gram3-comparative', 'gram8-plural', 'capital-common-countries',
                                         'city-in-state', 'family', 'gram9-plural-verbs', 'gram2-opposite',
                                         'currency', 'gram4-superlative', 'gram6-nationality-adjective',
                                         'gram7-past-tense',
                                         'gram5-present-participle', 'capital-world', 'gram1-adjective-to-adverb'])


    syntactic = set([c for c in set(category) if c.startswith("gram")])
    category_high_level = []
    for cat in category:
         category_high_level.append("syntactic" if cat in syntactic else "semantic")

    # dtype=object for memory efficiency
    return Bunch(X=np.vstack(questions).astype("object"),
                 y=np.hstack(answers).astype("object"),
                 category=np.hstack(category).astype("object"),
                 category_high_level=np.hstack(category_high_level).astype("object"))

