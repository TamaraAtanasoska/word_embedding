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
    path = os.getcwd() + '/eval/data/' + filename +'.txt'
    line_data = open(path).read()
    datalist = line_data.split('\n')
    data = dd(list)
    for line in datalist:
        if line.startswith(':'):
            key = line[2:]
            data[key] = []
        else:
            data[key].append(line.split())
    return data
