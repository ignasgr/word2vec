import torch.nn as nn


class CBOW(nn.Module):
    def __init__(self, vocab_size, dims):
        super().__init__()
        self.embeddings = nn.Embedding(num_embeddings=vocab_size, embedding_dim=dims)
        self.linear = nn.Linear(in_features=dims, out_features=vocab_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs).sum(dim=1)
        out = self.linear(embeds)
        return out


class SkipGram(nn.Module):
    def __init__(self, vocab_size, dims):
        super().__init__()
        self.embeddings = nn.Embedding(num_embeddings=vocab_size, embedding_dim=dims)
        self.linear = nn.Linear(in_features=dims, out_features=vocab_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs)
        out = self.linear(embeds)
        return out
