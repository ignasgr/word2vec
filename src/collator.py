import torch


class CBOWCollator:
    def __init__(self, context_length, vocab):
        self.context_length = context_length
        self.vocab = vocab

    def collate(self, sentences):
        """creates context, target pairs"""
        contexts = []
        targets = []

        for sentence in sentences:
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
    def __init__(self, context_length, vocab):
        self.context_length = context_length
        self.vocab = vocab

    def collate(self, sentences):
        """creates context, target pairs"""
        contexts = []
        targets = []

        for sentence in sentences:
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
