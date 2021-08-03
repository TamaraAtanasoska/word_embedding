import numpy as np
from scipy.stats import spearmanr
from scipy.spatial.distance import cosine as cosine_similarity

from datasets import get_WS353, get_RG65, get_RW


#THIS DOESN'T WORK YET!
#MERGING ONLY A FUNCTION SHELL TO PLACE CODE IN SAME FILE
def semantic_similarity_datasets(vocab):

    tasks = {
        "WS353": get_WS353(),
         "RG65": get_RG65(),
         "RW": get_RW(),
        }

     pass
