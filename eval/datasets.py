import numpy as np
import pandas as pd
from sklearn.utils import Bunch

def _get_as_df(name_or_path, **read_csv_kwargs):
    """
    Return the dataset as a pandas dataframe
    """
    path = 'data/' + name_or_path
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
    Scores were scaled by factor 10/4.
    """
    data = _get_as_df('EN-RG-65.txt', header=None, sep="\t").values

    return Bunch(X=data[:, 0:2].astype("object"),
                 y=data[:, 2].astype(np.float) * 10.0 / 4.0)

def get_RW():
    """
    Rare Words similarity dataset.
    """

    data = _get_as_df('EN-RW.txt', header=None, sep="\t").values

    return Bunch(X=data[:, 0:2].astype("object"),
                 y=data[:, 2].astype(np.float),
                 sd=np.std(data[:, 3:].astype(np.float)))
