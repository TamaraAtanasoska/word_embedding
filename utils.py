import nltk
import numpy as np
import os, utils
from collections import Counter, defaultdict as dd
import re, torch
import random
from torch.utils.data import Dataset
import yaml

with open('./config.yml', 'r') as f:
    config = yaml.safe_load(f)

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

def build_vocab(corpus: str) -> dict:
    tokens = [" ".join(word) + " </w>" for word in corpus]
    vocab = Counter(tokens)  
    return vocab


def get_stats(vocab: dict) -> dict:
    pairs = dd(int)
    for word, frequency in vocab.items():
        symbols = word.split()

        for i in range(len(symbols) - 1):
            pairs[symbols[i], symbols[i + 1]] += frequency

    return pairs


def merge_vocab(pair: tuple, v_in: dict) -> dict:
    v_out = {}
    bigram = re.escape(' '.join(pair))
    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
    
    for word in v_in:
        w_out = p.sub(''.join(pair), word)
        v_out[w_out] = v_in[word]

    return v_out


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
        print(token)
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
    def __init__(self, args, config):
        self.args = args
        self.config = config
        self.loader = Loader(args)
        #self.split = {'train': 'train', 'val': 'eval'}
        self.split = {'train': 'eval'}
        self.data_dict = dd(dd)

        # Based on dataset statistics, not many examples length > 50

        print('Loading datasets...')

        for item in self.split:
            path = 'data/' + self.split[item]
            self.data_dict[item]['data'], self.data_dict[item]['vocab'] = self.loader.load(path)
            print(str(item) + ' data loaded !')
            self.data_dict[item]['tokens'] = self.get_tokens(item)
            tokens = self.data_dict[item]['tokens']
            print('Preparing ' + str(item) + ' data...')
            target_words, context_words = [], []
            for idx in range(0, len(tokens)):
                target = tokens[idx]
                context = self.get_context(item, idx)
                context_words.extend(context)
                target_words.extend([target] * len(context))
            self.data_dict[item]['target'] = target_words
            self.data_dict[item]['context'] = context_words
            print('Data preparation completed')

    def get_context(self, split, idx):
        """
        This function returns list of context words for a given target word from batch
        :param split: [int] type of data {train, val}
        :param idx: [int] index of target word in the batch
        :return: [list] list of c context words for given target word
        """
        tokens = self.data_dict[split]['tokens']
        c = np.random.randint(1, self.config['WINDOW_SIZE'] + 1)
        start = idx - c if (idx - c) > 0 else 0
        stop = idx + c
        target_words = tokens[start:idx] + tokens[idx + 1:stop + 1]
        return list(target_words)

    def get_data(self, split):
        return self.data_dict[split]['data']

    def get_vocab(self, split):
        return self.data_dict[split]['vocab']

    def get_tokens(self, split):
        new_tokens = []
        if self.args.NGRAMS:
            print("Ngrams processing...") 

            ngram_vocab = build_vocab(self.get_data(split)) 
            for i in range(config['NUM_MERGES']):
                pairs = get_stats(ngram_vocab)
                if not pairs:
                    break

                best = max(pairs, key=pairs.get)
                ngram_vocab = merge_vocab(best, ngram_vocab)

            ngram_vocab = [*ngram_vocab]
            print(ngram_vocab)
            #new_tokens = [self.vocab.lookup_char_token(word) for word in ngram_vocab]
        else:
            words = self.get_data(split)
            vocab = self.get_vocab(split)
            tokens = [vocab.lookup_token(word) for word in words]
            new_tokens = sub_sampling(tokens) if self.args.SUBSAMPLING else tokens

        return new_tokens

    def __getitem__(self, idx):
        """
        :param idx: [int] index for dataset object
        :return: [tuple] value at given index and a vocabulary object
        """
        return self.data_dict[self.args.RUN_MODE]['target'][idx], self.data_dict[self.args.RUN_MODE]['context'][idx]

    def __len__(self):
        return self.data_dict[self.args.RUN_MODE].__len__()
