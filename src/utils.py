import string
import random
from collections import Counter, OrderedDict
from itertools import chain
from typing import Iterable, Optional, Dict

from torchtext.vocab import vocab


class SubSampler:
    def __init__(self, threshold: float, data: Optional[Iterable[str]] = None):
        self.threshold = threshold
        self.word_counts = Counter()
        self.subsample_probs: Dict[str, float] = {}

        if data is not None:
            self.update_counts_probs(data)

    def _calc_prob(self, freq: float) -> float:
        """
        Calculates the probability of dropping a word based on its frequency.
        """
        ratio = self.threshold / freq
        return 1 - ratio**0.5 if ratio < 1.0 else 0.0

    def update_counts(self, data: Iterable[str]) -> None:
        self.word_counts.update(data)

    def update_probs(self) -> None:
        """
        Updates the subsampling probabilities for each word in the word counts.
        """
        total_words = self.total_words
        self.subsample_probs = {
            word: self._calc_prob(count / total_words)
            for word, count in self.word_counts.items()
        }

    @property
    def total_words(self) -> int:
        return sum(self.word_counts.values())

    def update_counts_probs(self, data: Iterable[str]) -> None:
        """
        Updates words counts and subsampling probabilities
        """
        self.update_counts(data)
        self.update_probs()

    def sample(self, data: Iterable[str]) -> list[str]:
        """
        Applies subsampling to the provided data based on the computed probabilities.
        """
        return [
            word for word in data if random.random() > self.subsample_probs.get(word, 0)
        ]


def preprocess(example):
    """lowercases, splits, and removes punctuation from text"""
    trans_table = str.maketrans("", "", string.punctuation)
    no_punct = example["sentence"].translate(trans_table)
    example["tokens"] = no_punct.lower().split()
    return example


def create_vocab(
    sentences: list[list[str]], max_size=None, oov_token="<unk>", **kwargs
):
    """builds torchtext vocabular object with an option to limit the size to most frequent terms"""
    word_counter = Counter(chain.from_iterable(sentences))

    # Limit vocabulary size to the most frequent terms
    if max_size is not None:
        word_counter = dict(word_counter.most_common(max_size))

    vocab_obj = vocab(
        ordered_dict=OrderedDict(word_counter), specials=[oov_token], **kwargs
    )
    vocab_obj.set_default_index(vocab_obj[oov_token])
    return vocab_obj
