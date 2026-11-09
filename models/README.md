# Bundled data

This file is not covered by the package's MIT licence for its code; it keeps its source's terms.

## potion-base-8M.bin

[minishlab/potion-base-8M](https://huggingface.co/minishlab/potion-base-8M) by MinishLab, a
[Model2Vec](https://github.com/MinishLab/model2vec) model distilled from
[BAAI/bge-base-en-v1.5](https://huggingface.co/BAAI/bge-base-en-v1.5), released under the MIT licence. Its
vocabulary is bge-base-en-v1.5's (BERT's, uncased).

Modified: repacked by `mini-semantic-search pack` at 2 bits per value (each value one of four fitted levels, times
an f32 scale for its dimension and an f16 scale for its token), 2.2 MB instead of 30.5 MB. All 29,528 tokens and all
256 dimensions are kept.
