import torch
import numpy as np

from scipy.stats.mstats import spearmanr
from scipy.spatial.distance import cosine as cosine_similarity
import utils
import eval.sr_datasets as sr_datasets

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def semantic_similarity_datasets(embeddings, vocab_inst):
    datasets = {
        "WS353": sr_datasets.get_WS353(),
        "RG65": sr_datasets.get_RG65(),
        "RW": sr_datasets.get_RW(),
        "GRU65": sr_datasets.get_GRU65(),
        "GRU350": sr_datasets.get_GRU350(),
        "ZG222": sr_datasets.get_ZG222(),
        "EN-GOOGLE": sr_datasets.get_google_analogy()
    }

    vocab = vocab_inst.get_vocab()

    # Lists to store all the similarity measurments
    spearman_corr_all = []
    cosine_sim_all = []

    for name, data in datasets.items():

        print("Sampling data from ", name)

        if name == 'EN-GOOGLE':
            for category, instances in data.items():
                correct = 0
                total = 0
                for instance in instances:
                    if all(elem.lower()+'</w>' in vocab for elem in instance) and instance:
                        total += 1
                        true = instance[3].lower()+'/<w>'
                        predicted = utils.word_analogy(instance[:3], embeddings, vocab_inst)
                        if true == predicted:
                            correct += 1
                accuracy = correct / float(total) if total != 0 else -1
                print(f'Accuracy for type:{category} is {accuracy}')
        else:

            spearman_err = 0
            cosine_err = 0
            word_pairs = 0

            for i in range(len(data.X)):
                word1, word2 = data.X[i][0] + '</w>', data.X[i][1] + '</w>'
                #print(word1,word2)
                if word1 not in vocab or word2 not in vocab:
                    # Only proceed if the words are found in vocab
                    continue

                # Lookup the indices representation of found word
                token1 = torch.LongTensor([vocab_inst.lookup_token(word1)]).to(device)
                token2 = torch.LongTensor([vocab_inst.lookup_token(word2)]).to(device)
                # Look for the indices representation in the embeddings
                vec1 = embeddings(token1).detach().cpu().numpy()
                vec2 = embeddings(token2).detach().cpu().numpy()

                # Calculate the spearman correlation
                spearman_corr, _ = spearmanr(vec1, vec2)

                spearman_corr = abs(spearman_corr)

                spearman_err += abs(spearman_corr - data.y[i] / 10)


                # Calculate cosine similarity
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
                print("Cosine similarity error: {}".format(cosine_err))
            else:
                print("No word pairs for {} dataset found in vocab. \
                       Similarity cannot be reported".format(name))
