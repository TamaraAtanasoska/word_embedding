import nltk
import numpy as np
import os, utils
from collections import Counter, defaultdict as dd
import re
import random
from torch.utils.data import Dataset


def preprocess(text):
    words = text.split()
    word_counts = Counter(words)
    trimmed_words = [word for word in words if word_counts[word] > 5]
    return trimmed_words


def sub_sampling(tokens, threshold=1e-5):
    words_count = Counter(tokens)
    total_words = len(tokens)
    word_freq = {word: count / total_words for word, count in words_count.items()}
    word_prob = {word: 1 - np.sqrt(threshold / word_freq[word]) for word in words_count}
    sampled_vocab = [word for word in tokens if random.random() < word_prob[word]]
    return sampled_vocab


class Vocabulary(object):
    def __init__(self, token_to_idx=None):
        if token_to_idx is None:
            token_to_idx = {}
        self._token_to_idx = token_to_idx
        self._idx_to_token = {idx: token for token, idx in self._token_to_idx.items()}

    def to_serializable(self):
        return {'token_to_idx': self._token_to_idx}

    @classmethod
    def from_serializable(cls, contents):
        return cls(**contents)

    def add_token(self, token):
        if token in self._token_to_idx:
            index = self._token_to_idx[token]
        else:
            index = len(self._token_to_idx)
            self._token_to_idx[token] = index
            self._idx_to_token[index] = token
        return index

    def lookup_token(self, token):
        return self._token_to_idx[token]

    def create_vocab(self, words):
        word_counts = Counter(words)
        # sorting the words from most to least frequent in text occurrence
        sorted_vocab = sorted(word_counts, key=word_counts.get, reverse=True)
        # create int_to_vocab dictionaries
        self._idx_to_token = {ii: word for ii, word in enumerate(sorted_vocab)}
        self._token_to_idx = {word: ii for ii, word in self._idx_to_token.items()}

        return self._token_to_idx, self._idx_to_token

    def lookup_index(self, index):
        if index not in self._idx_to_token:
            raise KeyError("the index (%d) is not in the Vocabulary" % index)
        return self._idx_to_token[index]

    def __str__(self):
        return "<Vocabulary(size=%d)>" % len(self)

    def __len__(self):
        return len(self._token_to_idx)


class Loader(object):
    def __init__(self, args):
        self.args = args

    def load(self, path):
        file = open(path).read()

        words = utils.preprocess(file)
        vocab = utils.Vocabulary()
        vocab.create_vocab(words)

        return words, vocab


class Dataset(Dataset):
    def __init__(self, args):
        self.args = args
        self.loader = Loader(args)
        self.split = {'train': 'train', 'val': 'eval'}
        self.data_dict = dd(dd)

        # Based on dataset statistics, not many examples length > 50

        print('Loading datasets...')

        for item in self.split:
            path = 'data/' + self.split[item]
            self.data_dict[item]['data'], self.data_dict[item]['vocab'] = self.loader.load(path)

    def __getitem__(self, idx):
        return self.data_dict[self.args.RUN_MODE]['data'][idx], self.data_dict[self.args.RUN_MODE]['vocab']

    def __len__(self):
        return self.data_dict[self.args.RUN_MODE].__len__()


class Dataloader(object):
    def __init__(self, dataset, batch_size=5, shuffle=True, window_size=5):
        self.words = [word for word, _ in dataset]
        self.vocab = dataset[0][1]
        self.tokens = [self.vocab.lookup_token(word) for word in self.words]
        self.batch_size = batch_size
        self.window_size = window_size
        self.shuffle = shuffle  # TO DO

    def get_target(self, idx):
        R = np.random.randint(1, self.window_size + 1)
        start = idx - R if (idx - R) > 0 else 0
        stop = idx + R
        target_words = self.tokens[start:idx] + self.tokens[idx + 1:stop + 1]
        return list(target_words)

    def get_batches(self):
        words = sub_sampling(self.tokens)
        n_batches = len(words) // self.batch_size
        # only full batches
        words = words[:n_batches * self.batch_size]

        for idx in range(0, len(words), self.batch_size):
            x, y = [], []
            batch = words[idx:idx + self.batch_size]
            for ii in range(len(batch)):
                batch_x = batch[ii]
                # print(batch_x)
                batch_y = self.get_target(ii)
                y.extend(batch_y)
                x.extend([batch_x] * len(batch_y))
            yield x, y
