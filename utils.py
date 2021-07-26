import nltk
import numpy as np
import os, utils, string
from collections import Counter, defaultdict as dd
import re, torch
import random

from torch import Tensor
from torch.utils.data import Dataset
import yaml

with open('./config.yml', 'r') as f:
    config = yaml.safe_load(f)


def preprocess(text: string) -> list:
    """
    This function converts raw text data into words and remove words with frequency less than 5
    :param text: [string] sequence of string
    :return: [list] list of words in raw data
    """
    words = text.split()
    word_counts = Counter(words)
    trimmed_words = [word + '</w>' for word in words if word_counts[word] > 5]
    return trimmed_words


def sub_sampling(tokens: list, threshold=1e-5) -> list:
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


def get_noise_dist(words: list) -> Tensor:
    '''
    This function returns noise distribution to find negative samples for a target word
    :param words: list of all words in dataset
    :return: probability distribution over all words
    '''
    counter = Counter(words)
    total = len(words)
    freqs = {word: count / total for word, count in counter.items()}
    word_freqs = np.array(sorted(freqs.values(), reverse=True))
    unigram_dist = word_freqs / word_freqs.sum()
    noise_dist = torch.from_numpy(unigram_dist ** (0.75) / np.sum(unigram_dist ** (0.75)))
    return noise_dist


def merge_vocab(pair: tuple, v_in: dict) -> dict:
    '''

    :param pair: a tuple of two strings e.g ('es', 't') or ('e', 'r')
    :param v_in: current vocabulary with space seperated words
    :return: new vocabulary with given pair as single string in all occurrence over the vocabulary
    e.g: v_in {'m o d e s t':12, 'f a s t e r': 34, ... }
         v_out { 'm o d est':12, 'f a s t er': 34, ....}
    '''
    v_out = {}
    ngram = re.escape(' '.join(pair))
    p = re.compile(r'(?<!\S)' + ngram + r'(?!\S)')

    for word in v_in:
        w_out = p.sub(''.join(pair), word)
        v_out[w_out] = v_in[word]

    return v_out


def get_pairs(vocab: dict) -> dict:
    '''

    :param vocab: dictionary containing words as key and their corresponding count as value
    :return: [dict] a dictionary containing pairs of string as key and their count as value.
    e.g: { ('a', 'n'): 1508,
             ('es', 't'): 527,
             ('e', 'r'): 1031, ... }
    '''
    pairs = dd(int)
    for word, frequency in vocab.items():
        symbols = word.split()

        for i in range(len(symbols) - 1):
            pairs[symbols[i], symbols[i + 1]] += frequency

    return pairs


class Vocabulary(object):
    def __init__(self, cnfig, token_to_idx=None):

        self.config = cnfig
        if token_to_idx is None:
            token_to_idx = {}
        self._token_to_idx = token_to_idx
        self._idx_to_token = {idx: token for token, idx in self._token_to_idx.items()}
        self._ngram_vocab = None
        self._word_vocab = None

    def to_serializable(self):
        return {'token_to_idx': self._token_to_idx}

    @classmethod
    def from_serializable(cls, contents):
        return cls(**contents)

    def add_token(self, token: string) -> int:
        if token in self._token_to_idx:
            index = self._token_to_idx[token]
        else:
            index = len(self._token_to_idx)
            self._token_to_idx[token] = index
            self._idx_to_token[index] = token
        return index

    def lookup_token(self, token: string) -> int:
        return self._token_to_idx[token]

    def create_vocab(self, words: list):
        '''
        This function uses Byte Pair Encoding to find ngrams in given words.
        :param words: list of words to create vocabulary from
        :return: None
        '''
        tokens = [" ".join(word) for word in words]  # space seperated words
        vocab = Counter(tokens)
        for i in range(self.config['NUM_MERGES']):
            pairs = get_pairs(vocab)
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            vocab = merge_vocab(best, vocab)
        tokens = list(vocab.keys())
        vocab = list(string.ascii_lowercase)
        for token in tokens:
            vocab += token.split()
        vocab = set(vocab)  # ngrams from BPE algorithm
        word_in_ngram = set(
            [word for word in words if word in vocab])  # full words in ngram vocab
        whole_words = set(words) ^ (word_in_ngram)  # words not in ngram vocab
        # aplbhabets =
        self._ngram_vocab = list(set(vocab) ^ set(word_in_ngram))
        self._word_vocab = set(words)
        self._idx_to_token = {ii: word for ii, word in enumerate(list(vocab) + list(whole_words))}
        self._token_to_idx = {word: ii for ii, word in self._idx_to_token.items()}

    def lookup_index(self, index: int) -> string:
        if index not in self._idx_to_token:
            raise KeyError("the index (%d) is not in the Vocabulary" % index)
        return self._idx_to_token[index]

    def lookup_ngram(self, words: list) -> list:
        '''

        :param words: [list] list of words
        :return: [list] list of ngrams in words
        '''
        outputs = []
        vocab = self.get_ngram_vocab()
        for word in words:
            start, end = 0, len(word)
            cur_output = []
            # Look for grams with the longest possible length in ngram vocab
            while start < len(word) and start < end:
                if word[start:end] in vocab:
                    cur_output.append(word[start:end])
                    start = end
                    end = len(word)
                else:
                    end -= 1
            outputs.append(' '.join(cur_output))
        return outputs

    def get_vocab(self) -> list:
        '''
        :return: Combined vocabulary containing words and ngrams
        '''
        return list(self._token_to_idx.keys())

    def get_word_vocab(self) -> list:
        '''

        :return: list of words vocabulary
        '''
        return self._word_vocab

    def get_ngram_vocab(self) -> list:
        '''
        :return: list of ngram vocabulary. NOTE: this vocabulary always contains alphabets.
        '''
        return self._ngram_vocab

    def __str__(self):
        return "<Vocabulary(size=%d)>" % len(self)

    def __len__(self):
        return len(self._token_to_idx)


class Loader(object):
    def __init__(self, args, cnfg):
        self.args = args
        self.cnfg = cnfg

    def load(self, path: string) -> tuple:
        '''

        :param path: [string] path of data file to read from
        :return: ( list, Vocabulary) list of words in dataset and a Vocabulary class created using these words is
        returned
        '''
        file = open(path).read()
        words = preprocess(file)
        vocab = Vocabulary(self.cnfg)
        vocab.create_vocab(words)
        print('Vocabulary created')
        return words, vocab


class Dataset(Dataset):
    def __init__(self, args):
        self.args = args

        self.config = config
        self.loader = Loader(args, config)
        # self.split = {'train': 'train', 'val': 'eval'}
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
            ngram_idx = self.get_ngram_token(item) if args.NGRAMS else None
            target_words, context_words, ngrams = [], [], []
            for idx in range(0, len(tokens)):
                target = tokens[idx]
                # gram = self.get_ngram_tok(item, target)
                gram = ngram_idx[idx] if args.NGRAMS else None
                context = self.get_context(item, idx)
                context_words.extend(context)
                target_words.extend([target] * len(context))
                if args.NGRAMS:
                    ngrams.extend([gram] * len(context))
            self.data_dict[item]['target'] = target_words
            self.data_dict[item]['context'] = context_words
            self.data_dict[item]['ngram'] = ngrams

            print('Data preparation completed')

    def get_context(self, split: string, idx: int) -> list:
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

    def get_data(self, split: string) -> list:
        '''
        :param split: [string] (train,val,test)
        :return: [list] list of words for given split
        '''
        return self.data_dict[split]['data']

    def get_vocab_cls(self, split: string) -> Vocabulary:
        '''

        :param split: [string] (train,val,test)
        :return: [Vocabulary] Vocabulary class instance for given split
        '''
        return self.data_dict[split]['vocab']


    def get_tokens(self, split: string) -> list:
        '''
        This function converts words from given split data into indices from vocabulary
        :param split: [string] (train,val,test)
        :return: [list] list of integers corresponding to indices of words in vocabulary for given split
        '''
        words = self.get_data(split)
        vocab = self.get_vocab_cls(split)
        tokens = [vocab.lookup_token(word) for word in words]
        new_tokens = sub_sampling(tokens) if self.args.SUBSAMPLING else tokens
        return new_tokens


    def get_ngram_token(self, split: string) -> list:
        '''
        This function convert each word into list of indices of words and ngrams present in the word.
        e.g: say word is 'hello' with index 134 in vocabulary. Next we look for all possible ngrams of hello present in
        our ngram vocab. say we have 'he'(index:23) and 'lo'(index:48) in our ngram vocab. So we create a list of
        indices as [134, 23, 11, 48] which corresponds to [hello, he, l, lo]. We do this for all the words in given
        dataset and return a list of list.
        :param split: [string] (train,val,test)
        :return: [list] list of integers corresponding to indices of words and ngram in vocabulary for given split
        '''
        print("Ngrams processing...")
        tokens = [word for word in self.get_data(split)]
        vocab_cls = self.get_vocab_cls(split)  # Using ngram vocabulary to look for ngrams in given word
        new_tokens = vocab_cls.lookup_ngram(tokens)
        ngram_tokens = [[vocab_cls.lookup_token(tok)] + [vocab_cls.lookup_token(gram) for gram in word.split()] for
                        word, tok in zip(new_tokens, tokens)]
        return ngram_tokens

    def __getitem__(self, idx: int) -> tuple:
        """
        :param idx: [int] index for dataset object
        :return: [tuple] value at given index and a vocabulary object
        """

        if self.args.NGRAMS:
            return self.data_dict[self.args.RUN_MODE]['ngram'][idx], self.data_dict[self.args.RUN_MODE]['context'][idx]
        else:
            return self.data_dict[self.args.RUN_MODE]['target'][idx], self.data_dict[self.args.RUN_MODE]['context'][idx]


    def __len__(self):
        return self.data_dict[self.args.RUN_MODE].__len__()


class DataLoader(object):
    def __init__(self,
                 dataset,
                 batch_size=5,
                 shuffle=True
                 ):
        self.data = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle  # TO DO

    def get_batches(self):
        """
        It generate a batch of training data as pair of target and context word
        :return: [list] [list] list of target words and their corresponding context words
        """
        for i in range(len(self.data)):
            yield self.data[i:i + self.batch_size]

