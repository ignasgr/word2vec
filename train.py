import logging
import os
import torch
import torch.nn as nn
from datasets import load_dataset
from torch.utils.data import DataLoader
import torch.optim as optim
import argparse

from src.model import CBOW
from src.collator import CBOWCollator
from src.utils import preprocess, create_vocab
from src.constants import EMBEDDING_DIMS, VOCAB_SIZE, MIN_WORD_FREQ, CONTEXT_LENGTH
from src.dataset import GenericPairDataset


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename="./logs/training_log.log",
    filemode="w",
)

# parse command-line arguments
parser = argparse.ArgumentParser(description="CBOW Model Training")
parser.add_argument("--epochs", type=int, default=5, help="number of training epochs")
parser.add_argument(
    "--output_dir",
    required=True,
    help="directory for saving checkpoints",
)
args = parser.parse_args()

# load dataset and preprocess
logging.info("Loading and preprocessing dataset...")
data = load_dataset(
    "deokhk/en_wiki_sentences_100000", split="train", cache_dir="./data"
)
data = data.map(preprocess, remove_columns="sentence", num_proc=7)

# create a vocabulary
logging.info("Creating vocabulary...")
vocabulary = create_vocab(
    sentences=data["tokens"], max_size=VOCAB_SIZE, min_freq=MIN_WORD_FREQ
)
torch.save(vocabulary, os.path.join(args.output_dir, f"vocab.pt"))

# load the model
logging.info("Loading the model...")
model = CBOW(vocab_size=len(vocabulary), dims=EMBEDDING_DIMS)

# create dataset
logging.info("Creating dataset...")
collator = CBOWCollator(context_length=CONTEXT_LENGTH, vocab=vocabulary)
contexts, targets = collator.collate(data["tokens"])
dataset = GenericPairDataset(contexts, targets)

# prepare data loader
logging.info("Preparing data loader...")
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
criterion = nn.CrossEntropyLoss(ignore_index=vocabulary.get_default_index())
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ensure the output directory exists
os.makedirs(args.output_dir, exist_ok=True)

# main training loop
logging.info("Starting training loop...")
for epoch in range(args.epochs):
    epoch_loss = 0

    for batch_contexts, batch_targets in dataloader:
        optimizer.zero_grad()

        # forward pass
        pred = model(batch_contexts)
        loss = criterion(pred, batch_targets)

        # backward pass
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    logging.info(f"Epoch {epoch+1}/{args.epochs}, Loss: {epoch_loss/len(dataset):.4f}")

    # Save the model checkpoint
    checkpoint_path = os.path.join(args.output_dir, f"model_epoch_{epoch+1}.pt")
    torch.save(model.state_dict(), checkpoint_path)
