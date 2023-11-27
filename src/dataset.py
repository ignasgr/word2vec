from torch.utils.data import Dataset


class GenericPairDataset(Dataset):
    def __init__(self, contexts, targets):
        self.contexts = contexts
        self.targets = targets

    def __len__(self):
        return len(self.contexts)

    def __getitem__(self, idx):
        return self.contexts[idx], self.targets[idx]
