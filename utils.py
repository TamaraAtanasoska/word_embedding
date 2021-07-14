import nltk
import numpy as np
import os, utils
from collections import Counter, defaultdict as dd
import re, torch
import random
from torch.utils.data import Dataset


def preprocess(text):
    """
    This function converts raw text data into words and remove words with frequency less than 5
    :param text: [string] sequence of string
    :return: [list] list of words in raw data
    """
    words = text.split()
    word_counts = Counter(words)
    trimmed_words = [word for word in words if word_counts[word] > 5]
    return trimmed_words


def sub_sampling(tokens, threshold=1e-5):
    """
    This function samples words from a defined probability distribution in order to counter imbalance of
    the rare and frequent words. Proposed probability is chances that a word will be discarded from training set.
    :param tokens: [list] dataset in integer form
    :param threshold: [float]
    :return: [list] subsampled training data
    """
    print("Running subsampling...")
    words_count = Counter(tokens)
    total_words = len(tokens)
    word_freq = {word: count / total_words for word, count in words_count.items()}
    word_prob = {word: 1 - np.sqrt(threshold / word_freq[word]) for word in words_count}  # Proposed Probability
    sampled_vocab = [word for word in tokens if random.random() < word_prob[word]]
    return sampled_vocab


def get_noise_dist(words):
    counter = Counter(words)
    total = len(words)
    freqs = {word: count / total for word, count in counter.items()}
    word_freqs = np.array(sorted(freqs.values(), reverse=True))
    unigram_dist = word_freqs / word_freqs.sum()
    noise_dist = torch.from_numpy(unigram_dist ** (0.75) / np.sum(unigram_dist ** (0.75)))
    return noise_dist


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
        print('Vocabulary created')
        return words, vocab


class Dataset(Dataset):
    def __init__(self, args):
        self.args = args
        # self.config = config
        self.loader = Loader(args)
        self.split = {'train': 'train', 'val': 'eval'}
        self.data_dict = dd(dd)

        # Based on dataset statistics, not many examples length > 50

        print('Loading datasets...')

        for item in self.split:
            path = 'data/' + self.split[item]
            self.data_dict[item]['data'], self.data_dict[item]['vocab'] = self.loader.load(path)

    def get_data(self, split):
        return self.data_dict[split]['data']

    def get_vocab(self, split):
        return self.data_dict[split]['vocab']

    def __getitem__(self, idx):
        """
        :param idx: [int] index for dataset object
        :return: [tuple] value at given index and a vocabulary object
        """

        # TO DO : This looks inefficient for as we are sending vocabulary with each word. Should be taken care in future
        return self.data_dict[self.args.RUN_MODE]['data'][idx], self.data_dict[self.args.RUN_MODE]['vocab']

    def __len__(self):
        return self.data_dict[self.args.RUN_MODE].__len__()


class Dataloader(object):
    def __init__(self, 
                 dataset, 
                 split, 
                 batch_size = 5, 
                 shuffle = True,
                 window_size = 5, 
                 subsampling = False
                ):
        self.words = dataset.get_data(split)
        self.vocab = dataset.get_vocab(split)
        self.tokens = [self.vocab.lookup_token(word) for word in self.words]
        self.batch_size = batch_size
        self.window_size = window_size
        self.shuffle = shuffle  # TO DO
        self.subsampling = subsampling

    def get_context(self, batch, idx):
        """
        This function returns list of context words for a given target word from batch
        :param batch: [list] sequence of training data in tokenized form
        :param idx: [int] index of target word in the batch
        :return: [list] list of c context words for given target word
        """
        c = np.random.randint(1, self.window_size + 1)
        start = idx - c if (idx - c) > 0 else 0
        stop = idx + c
        target_words = batch[start:idx] + batch[idx + 1:stop + 1]
        return list(target_words)

    def get_batches(self):
        """
        It generate a batch of training data as pair of target and context word
        :return: [list] [list] list of target words and their corresponding context words
        """
        words = sub_sampling(self.tokens) if self.subsampling else self.tokens
        n_batches = len(words) // self.batch_size
        # only full batches
        words = words[:n_batches * self.batch_size]

        for idx in range(0, len(words), self.batch_size):
            target_words, context_words = [], []
            batch = words[idx:idx + self.batch_size]
            for target_idx in range(len(batch)):
                batch_x = batch[target_idx]
                batch_y = self.get_context(batch, target_idx)
                context_words.extend(batch_y)
                target_words.extend([batch_x] * len(batch_y))
            yield target_words, context_words
