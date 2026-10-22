# Bundled data

These files are not covered by the package's MIT licence for its code; each keeps its source's terms.

## potion-base-8M.bin

[minishlab/potion-base-8M](https://huggingface.co/minishlab/potion-base-8M) by MinishLab, a
[Model2Vec](https://github.com/MinishLab/model2vec) model distilled from
[BAAI/bge-base-en-v1.5](https://huggingface.co/BAAI/bge-base-en-v1.5), released under the MIT licence. Its
vocabulary is bge-base-en-v1.5's (BERT's, uncased).

Modified: repacked by `model2vec-js pack` at 4 bits per value (groups of 64 sharing an f16 scale), 4.2 MB
instead of 30.5 MB. All 29,528 tokens and 256 dimensions are kept.

## words.txt

Real English words for the spelling corrector to leave alone, from
[SCOWL](http://wordlist.aspell.net/) (Spell Checker Oriented Word Lists) 2020.12.07 by Kevin Atkinson: size 60
in American, British, Canadian and Australian spelling, with proper names up to size 50.

Modified: only words of 4 or more letters that potion-base-8M has no token for, lower-cased with accents
removed, and front-coded. `scripts/build-words.sh` rebuilds it.

SCOWL's copyright and permission notices, which include those of the lists it is made from, are reproduced
verbatim in [SCOWL-COPYRIGHT.txt](SCOWL-COPYRIGHT.txt).
