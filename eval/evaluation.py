import torch
import numpy as np

from scipy.stats.mstats import spearmanr
from sklearn.metrics.pairwise import cosine_similarity
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
        "EN-GOOGLE": sr_datasets.get_google_analogy('EN-GOOGLE'),
        "DE-GOOGLE": sr_datasets.get_google_analogy('DE-GOOGLE')
    }

    vocab = vocab_inst.get_vocab()

    # Lists to store all the similarity measurments
    spearman_corr_all = []
    cosine_sim_all = []

    for name, data in datasets.items():

        print("Sampling data from ", name)

        if name in ['EN-GOOGLE', 'DE-GOOGLE']:
            for category, instances in data.items():
                correct = 0
                total = 0
                for instance in instances:
                    if all(elem.lower()+'</w>' in vocab for elem in instance) and instance:
                        total += 1
                        true = instance[3].lower()+'/<w>'
                        predicted = utils.word_analogy(instance[:3], embeddings, vocab_inst)
                        #print(instance, predicted, true)
                        if true == predicted:
                            correct += 1
                accuracy = correct / float(total) if total != 0 else -1
                print(f'Accuracy for type:{category} is {accuracy}')
        else:

            human_judgmnt = []
            cosine_sim = []
            word_pairs = 0

            for i in range(len(data.X)):
                word1, word2 = data.X[i][0].lower() + '</w>', data.X[i][1].lower() + '</w>'
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

                cosine_sim.append(np.round(cosine_similarity(vec1, vec2).squeeze()*10, 3))
                human_judgmnt.append(data.y[i])

                word_pairs += 1

            if word_pairs:
                print(human_judgmnt)
                print(cosine_sim)
                spearman_corr, _ = spearmanr(human_judgmnt, cosine_sim)
                print('Word pairs found: ',word_pairs)
                print('Spearman rank correlation coefficient: ', spearman_corr)
            else:
                print("No word pairs for {} dataset found in vocab. \
                       Similarity cannot be reported".format(name))
