from jamo import h2j, j2hcj
import re
from konlpy.tag import Twitter
from soynlp.tokenizer import MaxScoreTokenizer
from soynlp.word import WordExtractor
from collections import Counter
from abc import *


class BaseTokenizer(ABC):
    """
    Base class for tokenizers
    """
    def __init__(self, config):
        pass

    def fit(self, raw_text):
        pass

    def state_dict(self):
        pass

    def load_state_dict(self):
        pass

    @abstractmethod
    def tokenize(self, raw_text):
        tokenized_text = [sentence.split() for sentence in raw_text]
        return tokenized_text


class JamoTokenizer(BaseTokenizer):
    """
    Split text into jamos, delete all whitespace
    """
    def __init__(self, config):
        pass

    def tokenize(self, raw_text):
        tokenized_text = [j2hcj(h2j(sentence)) for sentence in raw_text]
        return tokenized_text


class TwitterTokenizer(BaseTokenizer):
    """
    Tokenize text using Twitter of KoNLPy
    """
    def __init__(self, config):
        self.twitter = Twitter()

    def tokenize(self, raw_text, stem=False, norm=False):
        tokenized_text = [self.twitter.pos(sentence, stem=stem, norm=norm) for sentence in raw_text]
        tokenized_text = [[token for token, pos in sentence] for sentence in tokenized_text]
        return tokenized_text


class SoyNLPTokenizer(BaseTokenizer):
    """
    Tokenize text using MaxScoreTokenizer of SoyNLP
    """
    def __init__(self):
        self.tokenizer = None
        self.scores = list()
        self.word_extractor = WordExtractor(min_coount=100,
                                            min_cohesion_forward=0.05,
                                            min_right_branching_entropy=0.0)

    def fit(self, raw_text):
        self.word_extractor.train(raw_text)
        scores = self.word_extractor.extract()
        scores = [(word, (score.cohesion_forward + score.cohesion_backward) * \
                   (score.left_branching_entropy + score.right_branching_entropy)) for word, score in scores.items()]
        self.scores = scores
        self.tokenizer = MaxScoreTokenizer(scores=self.scores)

    def state_dict(self):
        return {'scores': self.scores}

    def load_state_dict(self, state_dict):
        self.scores = state_dict['scores']
        self.tokenizer = MaxScoreTokenizer(scores=self.scores)

    def tokenize(self, raw_text):
        tokenized_text = [self.tokenizer.tokenize(sentence) for sentence in raw_text]
        return tokenized_text

