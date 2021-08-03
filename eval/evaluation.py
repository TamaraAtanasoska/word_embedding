import numpy as np
from scipy.stats import spearmanr
from scipy.spatial.distance import cosine as cosine_similarity

import datasets


def semantic_similarity_datasets(vocab):

    #these are the tasks, we have only three
    #there will be 6 once the german sets are there
    tasks = {
        "WS353": datasets.get_WS353(),
         "RG65": datasets.get_RG65(),
         "RW": datasets.get_RW(),
        }

    spearman_errors = []
    cosine_errors = []

    for name, data in tasks.items():
        #now we take one dataset at a time
        print("Sampling data from ", name)
        spearman_err = 0
        cosine_err = 0
        analogies = 0

        for i in range(len(data.X)):
            #we go pair by pair
            word1, word2 = data.X[i][0], data.X[i][1]
            #look if they are in our vocab. if not, we move on
            if word1 not in vocab or word2 not in vocab:
                continue

            #lookup_table needs to be substituted with our function 
            #like we have in the Vocabulary class 
            spearman_corr, _ = spearmanr(lookup_table(word1), lookup_table(word2))
            #spearman correlation found, there is a way to write this shorter
            spearman_corr = abs(spearman_corr)
            spearman_err += abs(spearman_corr - data.y[i] / 10)

            #then we check for cosine similarity of the words too
            cosine_sim = 1 - cosine_similarity(lookup_table(word1), lookup_table(word2))
            cosine_err += abs(cosine_sim - data.y[i] / 10)
            #print here just to check accuracy
            print(word1, word2, data.y[i], cosine_sim)

            #confusing name, here just to check how many pairs we guessed on 
            analogies += 1

        spearman_err = 1 - spearman_err / analogies
        cosine_err = 1 - cosine_err / analogies
        spearman_errors.append(spearman_err)
        cosine_errors.append(cosine_err)

        print("Spearman correlation error on {} dataset: {}".format(name, spearman_err))
        print("Cosine similarity error on {} dataset: {}".format(name, cosine_err))
