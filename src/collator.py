from typing import List

import torch
import torchtext


class CBOWCollator:
    def __init__(self, context_length: int, vocab: torchtext.vocab):
        self.context_length = context_length
        self.vocab = vocab

    def collate(self, sentences: List[str]):
        """
        Creates predictor, target pairs for CBOW model. For CBOW, the context words are
        predictors, and the center word is the target.
        
        Context ---> Target
        [5431, 2, 25, 550] ---> [6]
        """
        predictors = []
        targets = []

        for sentence in sentences:

            # skipping sentences that aren't log enough
            if len(sentence) < 1 + 2 * self.context_length:
                continue

            encoded = self.vocab.lookup_indices(sentence)

            # target index refers to the target word, with context to left and right
            for center_idx in range(
                self.context_length, len(encoded) - self.context_length
            ):
                center = encoded[center_idx]
                context = (
                    encoded[center_idx - self.context_length : center_idx]
                    + encoded[center_idx + 1 : center_idx + self.context_length + 1]
                )

                predictors.append(context)
                targets.append(center)

        return torch.tensor(predictors), torch.tensor(targets)


class SkipGramCollator:
    def __init__(self, context_length: int, vocab: torchtext.vocab):
        self.context_length = context_length
        self.vocab = vocab

    def collate(self, sentences: List[str]):
        """
        Creates predictor, target pairs for skipgram model. For skipgram, the context words are
        the targets, and the center words are the predictors.

        context ---> target
        [4] ---> [67]
        """
        predictors = []
        targets = []

        for sentence in sentences:

            # skipping sentences that aren't log enough
            if len(sentence) < 1 + 2 * self.context_length:
                continue

            encoded = self.vocab.lookup_indices(sentence)

            # target index refers to target word, with context to the left and right
            for center_idx in range(
                self.context_length, len(encoded) - self.context_length
            ):
                center = encoded[center_idx]
                context = (
                    encoded[center_idx - self.context_length : center_idx]
                    + encoded[center_idx + 1 : center_idx + self.context_length + 1]
                )

                predictors.extend([center] * len(context))
                targets.extend(context)

        return torch.tensor(predictors), torch.tensor(targets)
