"""Writes test/fixtures/reference.json: token ids from Hugging Face's tokenizer and embeddings from the model2vec
Python library (full precision) for the sentences in it, to check the JavaScript against.

    pip install model2vec
    python scripts/make-fixture.py <potion-base-8M folder>   # e.g. a snapshot of minishlab/potion-base-8M
"""
import json
import sys

from model2vec import StaticModel
from tokenizers import Tokenizer

SENTENCES = [
    "How do I cancel my order?",
    "where's my parcel??? it's been 2 weeks!!!",
    "Café naïve résumé, coöperate and Ångström",
    "“Smart quotes”, em—dashes – and ellipses…",
    "e-mail: help@example.com, $12.95 (3–5 days) #1 @home 50% off",
    "你好，世界 and 東京 mixed with English",
    "Tabs\tand\nnewlines\r\nand   many    spaces",
    "supercalifragilisticexpialidocious antidisestablishmentarianism",
    "x" * 120,
    "UPPER CASE QUESTION ABOUT REFUNDS",
    "emoji 🙂 and symbols ™ © ® ° ± ×",
    "",
    "   ",
    "a",
    "The quick brown fox jumps over the lazy dog.",
    "reprots exprot transations vouchr",
    # Over 512 tokens, about two subjects: the vector depends on where it's cut.
    "Bread needs flour, water, salt and yeast; knead the dough and let it rise. " * 40
    + "Engines need fuel, oil, spark plugs and coolant; service the car every year. " * 40,
]

model_dir = sys.argv[1]
model = StaticModel.from_pretrained(model_dir)
tokenizer = Tokenizer.from_file(f"{model_dir}/tokenizer.json")
ids = [e.ids for e in tokenizer.encode_batch(SENTENCES, add_special_tokens=False)]
emb = model.encode(SENTENCES)
with open("test/fixtures/reference.json", "w") as f:
    json.dump({
        "model": "minishlab/potion-base-8M",
        "sentences": SENTENCES,
        "ids": ids,
        "embeddings": [[round(float(x), 6) for x in row] for row in emb],
    }, f, ensure_ascii=False)
print(f"{len(SENTENCES)} sentences")
