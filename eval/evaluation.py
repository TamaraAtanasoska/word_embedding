import torch
import numpy as np

from scipy.stats import spearmanr
from scipy.spatial.distance import cosine as cosine_similarity

import eval.sr_datasets as sr_datasets

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def semantic_similarity_datasets(model, vocab_inst):

    datasets = {
        "WS353": sr_datasets.get_WS353(),
         "RG65": sr_datasets.get_RG65(),
         "RW": sr_datasets.get_RW(),
         "GRU65": sr_datasets.get_GRU65(),
         "GRU350": sr_datasets.get_GRU350(),
         "ZG222": sr_datasets.get_ZG222(),
        }

    vocab = vocab_inst.get_vocab()
    
    #Lists to store all the similarity measurments 
    spearman_corr_all = []
    cosine_sim_all = []

    for name, data in datasets.items():

        print("Sampling data from ", name)

        spearman_err = 0
        cosine_err = 0
        word_pairs = 0

        for i in range(len(data.X)):
            word1, word2 = data.X[i][0], data.X[i][1]
            if word1 not in vocab or word2 not in vocab:
                #Only proceed if the words are found in vocab
                continue

            #Lookup the indices representation of found word
            token1 = torch.LongTensor([vocab_inst.lookup_token(word1)]).to(device)
            token2 = torch.LongTensor([vocab_inst.lookup_token(word2)]).to(device)
            #Look for the indices representation in the embeddings
            vec1 = model.out_embeddings(token1)
            vec2 = model.out_embeddings(token2)

            #Calculate the spearman correlation 
            spearman_corr, _ = spearmanr(vec1, vec2)
            spearman_corr = abs(spearman_corr)
            spearman_err += abs(spearman_corr - data.y[i] / 10)

            #Calculate cosine similarity
            cosine_sim = 1 - cosine_similarity(vec1, vec2)
            cosine_err += abs(cosine_sim - data.y[i] / 10)

            word_pairs += 1

        if word_pairs:
            spearman_err = 1 - spearman_err / word_pairs
            cosine_err = 1 - cosine_err / word_pairs
            spearman_corr_all.append(spearman_err)
            cosine_sim_all.append(cosine_err)

            print("Word pairs found: {}".format(word_pairs))
            print("Spearman correlation error: {}".format(spearman_err))
            print("Cosine similarity errort: {}".format(cosine_err))
        else:
            print("No word pairs for {} dataset found in vocab. \
                   Similarity cannot be reported".format(name))

