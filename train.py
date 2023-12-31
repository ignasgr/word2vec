import logging
import os
import torch
import torch.nn as nn
from datasets import load_dataset
from torch.utils.data import DataLoader
import torch.optim as optim
import argparse

from src.model import CBOW, SkipGram
from src.collator import CBOWCollator, SkipGramCollator
from src.utils import preprocess, create_vocab, SubSampler
from src.constants import (
    EMBEDDING_DIMS,
    VOCAB_SIZE,
    MIN_WORD_FREQ,
    CONTEXT_LENGTH,
    SUBSAMPLE_THRESH,
)
from src.dataset import GenericPairDataset


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename="./logs/training_log.log",
    filemode="w",
)

# parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True, choices=["skipgram", "cbow"])
parser.add_argument("--epochs", type=int, default=5, help="number of training epochs")
parser.add_argument("--batch_size", type=int, default=64, help="training batch size")
parser.add_argument("--lr", type=float, default=0.001, help="training batch size")
parser.add_argument("--split", default="train", choices=["train", "dev"])
parser.add_argument("--output_dir", required=True, help="directory for saving checkpoints")
args = parser.parse_args()

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Use device {device}")

# load dataset and preprocess
logging.info("Loading and preprocessing dataset...")
data = load_dataset(
    "deokhk/en_wiki_sentences_100000", split=args.split, cache_dir="./data"
)
data = data.map(preprocess, remove_columns="sentence", num_proc=7)

# # profiling data for subsampling
# subsampler = SubSampler(threshold=SUBSAMPLE_THRESH)
# for sents in data["tokens"]:
#     subsampler.update_counts(sents)
# subsampler.update_probs()

# # sampling data
# data = data.map(lambda example: {"tokens": subsampler.sample(example["tokens"])})

# create a vocabulary
logging.info("Creating vocabulary...")
vocabulary = create_vocab(
    sentences=data["tokens"], max_size=VOCAB_SIZE, min_freq=MIN_WORD_FREQ
)
torch.save(vocabulary, os.path.join(args.output_dir, f"vocab.pt"))

# load models and collators
logging.info("Loading models and collators...")
if args.model == "skipgram":
    model = SkipGram(vocab_size=len(vocabulary), dims=EMBEDDING_DIMS)
    collator = SkipGramCollator(context_length=CONTEXT_LENGTH, vocab=vocabulary)
elif args.model == "cbow":
    model = CBOW(vocab_size=len(vocabulary), dims=EMBEDDING_DIMS)
    collator = CBOWCollator(context_length=CONTEXT_LENGTH, vocab=vocabulary)
else:
    raise ValueError(f"Invalid model type: {args.model}")

model.to(device)

# create dataset
logging.info("Creating dataset...")
contexts, targets = collator.collate(data["tokens"])
dataset = GenericPairDataset(contexts, targets)

# prepare data loader
logging.info("Preparing data loader...")
dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
criterion = nn.CrossEntropyLoss(ignore_index=vocabulary.get_default_index())
optimizer = optim.Adam(model.parameters(), lr=args.lr)

# ensure the output directory exists
os.makedirs(args.output_dir, exist_ok=True)

# main training loop
logging.info("Starting training loop...")
for epoch in range(args.epochs):
    epoch_loss = 0

    for batch_contexts, batch_targets in dataloader:

        # Send data to GPU if available
        batch_contexts, batch_targets = batch_contexts.to(device), batch_targets.to(device)

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
    checkpoint_path = os.path.join(args.output_dir, f"{args.model}_epoch_{epoch+1}.pt")
    torch.save(model.state_dict(), checkpoint_path)
