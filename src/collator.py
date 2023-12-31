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
        targets = []

        for sentence in sentences:

            # skipping sentences that aren't log enough
            if len(sentence) < 1 + 2 * self.context_length:
                continue

            encoded = self.vocab.lookup_indices(sentence)

            # target index refers to the target word, with context to left and right
            for target_idx in range(
                self.context_length, len(encoded) - self.context_length
            ):
                target = encoded[target_idx]
                context = (
                    encoded[target_idx - self.context_length : target_idx]
                    + encoded[target_idx + 1 : target_idx + self.context_length + 1]
                )

                contexts.append(context)
                targets.append(target)

        return torch.tensor(contexts), torch.tensor(targets)


class SkipGramCollator:
    def __init__(self, context_length: int, vocab: torchtext.vocab):
        self.context_length = context_length
        self.vocab = vocab

    def collate(self, sentences: List[str]):
        """
        Creates predictor, target pairs for skipgram model. For skipgram, the context words are
        the targets, and the context words are the targets.

        context ---> target
        [4] ---> [67]
        """
        targets = []

        for sentence in sentences:

            # skipping sentences that aren't log enough
            if len(sentence) < 1 + 2 * self.context_length:
                continue

            encoded = self.vocab.lookup_indices(sentence)

            for target_idx in range(
                self.context_length, len(encoded) - self.context_length
            ):
                target = encoded[target_idx]
                context = (
                    encoded[target_idx - self.context_length : target_idx]
                    + encoded[target_idx + 1 : target_idx + self.context_length + 1]
                )

                contexts.extend(context)
                targets.extend([target] * len(context))

        return torch.tensor(contexts), torch.tensor(targets)
