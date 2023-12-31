# Word2Vec

This repository is an implementation of methodologies presented in the paper *Efficient Estimation of Word Representations in Vector Space* (2013), authored by Mikolov et al. The original paper can be found here: https://arxiv.org/abs/1301.3781

# Results



# Methodology

The models were trained on a single T4 gpu, using the deokhk/en_wiki_sentences_1000000 Huggingface dataset. Below are training parameters:

| Parameter            | Value |
|----------------------|-------|
| Embedding Dimensions | 100   |
| Vocabulary Size      | 10,000 |
| Epochs               | 10    |
| Learning Rate        | 0.001 |
| Batch Size           | 64    |


# Useage

Below is an example for how to run the training script:

`python train.py --model skipgram --output_dir ./checkpoints`

See the `train.py` for a full list of acceptable parameters.