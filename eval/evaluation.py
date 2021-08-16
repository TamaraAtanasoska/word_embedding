import string
from typing import Any, List

import torch
import numpy as np

from scipy.stats.mstats import spearmanr
from sklearn.metrics.pairwise import cosine_similarity
import random

from torch.nn import Embedding

import eval.sr_datasets as sr_datasets
from utils import Vocabulary

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def show_learning(embeddings:  Any, vocab: Any, device: Any) -> None:
    '''
    This function uses embeddings from the provided model and randomly select some words from vocabulary. Then for each
    random word, it find similar words using cosine similarity. Finally it prints out top 6 similar words to randonly
    chosen words.
    :param model: Trained model
    :param vocab: vocabulary instance
    :param device:
    :return: None
    '''

    embed_vectors = embeddings.weight

    word_vocab = list(vocab.get_word_vocab())
    total = len(word_vocab)
    idxs = random.sample(range(total), 5)
    words = [ word_vocab[idx] for idx in idxs]
    valid_examples = torch.LongTensor([vocab.lookup_token(word) for word in words]).to(device)
    valid_vectors = embeddings(valid_examples)

    magnitudes = embed_vectors.pow(2).sum(dim=1).sqrt().unsqueeze(0)
    valid_similarities = torch.mm(valid_vectors, embed_vectors.t()) / magnitudes
    _, closest_idxs = valid_similarities.topk(6)

    valid_examples, closest_idxs = valid_examples.to('cpu'), closest_idxs.to('cpu')
    for ii, valid_idx in enumerate(valid_examples):
        closest_words = [vocab.lookup_index(idx.item()) for idx in closest_idxs[ii] if idx.item() < len(vocab)][1:]
        print('Chosen word: '+vocab.lookup_index(valid_idx.item()) + " ----SIMILAR WORDS---- " + ', '.join(closest_words))
    print("...\n")


def word_analogy(words: list, embeddings, vocab) -> List:
    """
    This function perform word analogy task to test how well word embeddings are trained.
    NOTE:  This function expects given words to be part of vocabulary if gram_model = False i.e if we are not using
            ngram model.
    :param gram_model: Boolean to check if we'll use ngram model or not
    :param vocab: Vocabulary of dataset
    :param words: list of three words such that ### words[0] is to words[1] as words[2] is to ? ###
    :param embeddings: trained embeddings from model
    :return: target word which fits the analogy best
    """
    embed_vectors = embeddings.weight
    tokens = torch.LongTensor([vocab.lookup_token(ex) for ex in words]).to(device)
    vectors = embeddings(tokens)
    inp1 = (vectors[1] - vectors[0]) + vectors[2]
    inp2 = embed_vectors
    magnitudes = inp2.pow(2).sum(dim=1).sqrt().unsqueeze(0) * inp1.pow(2).sum(dim=0).sqrt().unsqueeze(0)
    similarities = torch.mm(inp1.unsqueeze(0), inp2.t()) / magnitudes
    val, idxs = similarities.topk(2)
    '''targets = []
    for id, v in zip(idxs.squeeze(), val.squeeze()):
        targets.append(vocab.lookup_index(id.item()))'''

    target = vocab.lookup_index(idxs[0][1].item())
    return target

def semantic_similarity_datasets(embeddings: Embedding, vocab_inst: Vocabulary) -> None:
    """
    This method test trained embeddings against different datasets discussed in Section 5.1 of
    [https://arxiv.org/abs/1607.04606] for evaluation. There are 6 dataset (3 English, 3 German)
    which contain human similarity judgement values between word pair different words category.
    This value is correlated against cosine similarity of those pairs (obtained using embeddings)
    using Spearman's rank correlation.
    :param embeddings: Trained embedding matrix from model
    :param vocab_inst: Instance of Vocabulary class
    :return: None
    """

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

    for name, data in datasets.items():

        print("Sampling data from ", name)

        if name in ['EN-GOOGLE', 'DE-GOOGLE']:
            correct = {'semantic': 0, 'syntactic': 0}
            total = {'semantic': 0, 'syntactic': 0}
            for i in range(len(data.X)):
                words = data.X[i]
                target = data.y[i]
                words = [word.lower() + '</w>' for word in words]
                target = target.lower() + '</w>'
                if all(elem in vocab_inst.get_word_vocab() for elem in words + [target]) and words:
                    total[data.category_high_level[i]] += 1
                    predicted = word_analogy(words, embeddings, vocab_inst)
                    if target in predicted:
                        correct[data.category_high_level[i]] += 1

            accuracy = correct['semantic'] / float(total['semantic']) if total['semantic'] != 0 else -1
            print(f'Accuracy for semantic data is {accuracy}')
            accuracy = correct['syntactic'] / float(total['syntactic']) if total['syntactic'] != 0 else -1
            print(f'Accuracy for syntactic data is {accuracy}')
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

            if word_pairs > 3:
                spearman_corr, _ = spearmanr(human_judgmnt, cosine_sim)
                print('Word pairs found: ',word_pairs)
                print('Spearman rank correlation coefficient: ', spearman_corr)
            else:
                print("No word pairs for {} dataset found in vocab. \
                       Similarity cannot be reported".format(name))
