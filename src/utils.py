import string
from collections import Counter, OrderedDict
from itertools import chain

from torchtext.vocab import vocab


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
