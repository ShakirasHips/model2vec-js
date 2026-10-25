# Bundled data

This file is not covered by the package's MIT licence for its code; it keeps its source's terms.

## potion-base-8M-128d.bin

[minishlab/potion-base-8M](https://huggingface.co/minishlab/potion-base-8M) by MinishLab, a
[Model2Vec](https://github.com/MinishLab/model2vec) model distilled from
[BAAI/bge-base-en-v1.5](https://huggingface.co/BAAI/bge-base-en-v1.5), released under the MIT licence. Its
vocabulary is bge-base-en-v1.5's (BERT's, uncased).

Modified: repacked by `mini-semantic-search pack` to its first 128 of 256 dimensions at 4 bits per value (each
token's 128 values sharing an f16 scale), 2.2 MB instead of 30.5 MB. All 29,528 tokens are kept.
