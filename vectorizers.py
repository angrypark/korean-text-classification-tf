# -*- coding: utf-8 -*-
from collections import Counter

class Vectorizer:
    def __init__(self, tokenizer, config):
        PAD_IDX = 0
        UNK_IDX = 1
        SOS_IDX = 2
        EOS_IDX = 3
        prefix = ['<PAD>', '<UNK>', '<SOS>', '<EOS>']

        self.tokenizer = tokenizer
        self.vocab_size = config.vocab_size
        self.embed_dim = config.embed_dim
        self.prefix = prefix
        self.vocab_words = list()
        self.idx2word = dict()
        self.word2idx = dict()

    def build_vectorizer(self, lines):
        self.vocab_words, self.word2idx, self.idx2word = self._build_vocab(lines)

    def indexer(self, word):
        try:
            return self.word2idx[word]
        except:
            return self.word2idx['<UNK>']

    def _build_vocab(self, lines):
        counter = Counter([token for line in lines for token in self.tokenizer.tokenize(line)])
        print("Total number of unique tokens : ", len(counter))
        vocab_words = self.prefix + sorted(list([word for (word, freq) in counter.most_common(self.vocab_size-4)]))

        word2idx = {word:idx for idx, word in enumerate(vocab_words)}
        idx2word = {idx:word for idx, word in enumerate(vocab_words)}

        return vocab_words, word2idx, idx2word

    def state_dict(self):
        state = {'idx2word' : self.idx2word,
                 'word2idx' : self.word2idx,
                 'vocab_words' : self.vocab_words}
        return state

    def load_state_dict(self, state_dict):
        self.idx2word = state_dict['idx2word']
        self.word2idx = state_dict['word2idx']
        self.vocab_words = state_dict['vocab_words']