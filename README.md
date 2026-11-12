# mini-semantic-search

Semantic search with [MiniSearch](https://github.com/lucaong/minisearch)'s API. Documents are matched by meaning
as well as by shared words: "stop sending me newsletters" finds "How do I unsubscribe from your emails?", and "the
mug came smashed" finds "My item arrived damaged or broken", while a name or a part number still finds the document
that has it. It runs in browsers and Node with no dependencies, no server and no API calls, and the package is
2.2 MB with its model (2.0 MB gzipped).

```ts
import MiniSemanticSearch from 'mini-semantic-search';

const search = await MiniSemanticSearch.create({
  fields: ['title', 'text'],       // fields to embed and search
  storeFields: ['title'],          // fields to return with results
});

search.addAll([
  { id: 1, title: 'Returns', text: 'Start a return under Orders in your account.' },
  { id: 2, title: 'Damaged items', text: 'Send us a photo of the damage and we will replace it.' },
  { id: 3, title: 'Shipping times', text: 'Standard shipping takes 3 to 5 business days.' },
]);

search.search('the mug came smashed');
// => [
//   { id: 2, score: 0.17, title: 'Damaged items' },
//   { id: 3, score: 0.08, title: 'Shipping times' },
//   { id: 1, score: 0.01, title: 'Returns' },
// ]
```

The embeddings come from [Model2Vec](https://github.com/MinishLab/model2vec), a static embedding model. It keeps one
vector per token, and a text's vector is the average of its tokens' vectors. There is no neural network to run, so
embedding a paragraph takes about 15 µs on a CPU.

## Install

```sh
npm install mini-semantic-search
```

- **Node 18+:** `import MiniSemanticSearch from 'mini-semantic-search'`. The bundled model loads from the package.
- **Bundlers (Vite, webpack 5, esbuild, Rollup):** the same import. The model is found with
  `new URL('../models/…', import.meta.url)`, which bundlers copy as an asset. Or pass `model:` your own URL.
- **Straight from a CDN, no build step:**

  ```html
  <script type="module">
    import MiniSemanticSearch from 'https://cdn.jsdelivr.net/npm/mini-semantic-search@0.3.0/dist/index.js';
    const search = await MiniSemanticSearch.create({ fields: ['title', 'text'] });
  </script>
  ```

  The model (2.2 MB) then loads from the CDN too.

## API

It follows MiniSearch, so code written for one mostly works with the other. The one difference in setting it up is
that the model has to load first, so you either `await MiniSemanticSearch.create(options)` or pass a loaded model
to the constructor:

```ts
import MiniSemanticSearch, { StaticModel } from 'mini-semantic-search';

const model = await StaticModel.load();  // load once, share between indexes
const search = new MiniSemanticSearch({ model, fields: ['title', 'text'] });
```

| Option | Default | |
|---|---|---|
| `fields` | required | The fields embedded and searched. A field holding an array is embedded item by item, and its closest item counts. |
| `storeFields` | `[]` | The fields returned with each result. |
| `idField` | `'id'` | The field holding each document's unique id. |
| `extractField` | `doc[field]` | `(document, fieldName) => value`, for nested or computed fields. |
| `searchOptions` | `{}` | Defaults for `search`. |
| `passageLength` | `120` | Texts longer than this many tokens (about ¾ of a word each) are split into passages of about this length, at sentence ends, and the closest passage counts. `0` leaves texts whole. |
| `model` | bundled potion-base-8M | A `StaticModel`, or (in `create`) a URL, path or bytes of a packed model. |

| Method | |
|---|---|
| `add(doc)`, `addAll(docs)` | Index documents. They can be searched straight away. A duplicate id throws. |
| `addAllAsync(docs, { chunkSize: 100 })` | The same, yielding between chunks so a page stays responsive. |
| `remove(doc)`, `discard(id)`, `discardAll(ids)` | Remove documents, by document or by id. |
| `removeAll(docs?)` | Remove the given documents, or all of them. |
| `replace(doc)` | Replace the document with the same id. |
| `has(id)`, `getStoredFields(id)`, `documentCount` | What's indexed. |
| `search(query, options?)` | Every document, best first, as `{ id, score, ...storeFields }`. |

Search options:

| Option | |
|---|---|
| `fields` | Search only these fields, e.g. `{ fields: ['title'] }`. |
| `boost` | Weights of fields against each other, e.g. `{ boost: { title: 3 } }`. |
| `filter` | `(result) => boolean`; the result includes its stored fields. |
| `minScore` | Leave out results scoring under this. |
| `keywordWeight` | How much of the score comes from shared words: `0` is meaning only, `1` is words only. Default `0.5`. |

**How documents are scored.** A score is the average of two, weighted by `keywordWeight`:

- **Meaning.** Each document is embedded once per field and once with all its fields together. This score is half
  the whole document's cosine similarity to the query, plus half its fields' similarities averaged (weighted by
  `boost`). A long text is embedded passage by passage, and its closest passage counts.
- **Words.** [BM25](https://en.wikipedia.org/wiki/Okapi_BM25) over the model's tokens, so rare words count for more
  than common ones, as a share of the most the query could score: 1 if the document is full of every word of the
  query, 0 if it has none. A word's count is added up over the fields searched, weighted by `boost`.

Scores run from about −0.1 to 1. A document always scores something, so every document comes back: `slice` the
results, or set `minScore` to drop the ones too far off to count as a match. Where to set it depends on your
documents' wording. Try your own off-topic questions and put it just above the scores they get.

**Which `keywordWeight`.** The default suits a mix of questions. Go lower (0.3) when people ask in their own words
for short entries they haven't seen, as with an FAQ. Go higher (0.7) when they search long documents for names and
terms. The tables below show both.

**How it differs from MiniSearch.** Results have no `terms`, `queryTerms` or `match`. There's no `prefix`, `fuzzy`,
`combineWith`, `autoSuggest` or query syntax, and no
`toJSON`/`loadJSON`: indexing costs about 0.05 ms per document, so you can just index again on load.

Indexing takes about 0.04 ms per document with three short fields, and about 0.15 ms for a 1,000-character text. A
search over 10,000 short documents takes about 4 ms, or about 13 ms when its `fields` or `boost` differ from the
last search's. (Options used twice running get each document's vectors added up into one, so a search is one dot
product per document, plus one per passage of the documents long enough to be split.)

## Embeddings

```ts
import { StaticModel, similarity } from 'mini-semantic-search';

const model = await StaticModel.load();            // or StaticModel.load(urlOrPathOrBytes)
const a = model.embed('how do I cancel my order');  // Float32Array(256), unit length
const b = model.embed('stop my purchase before it ships');
similarity(a, b);                                   // cosine similarity, 1 = same meaning
model.embedMany(['refund', 'exchange']);           // several at once
model.tokenize('Café');                             // token ids
```

This matches the Python library. Token ids are identical to Hugging Face's tokenizer, and with a full-precision
model the vectors match `model2vec` to within 1e-6. The tests check the token ids and the bundled model's vectors
against fixtures made with `scripts/make-fixture.py`.

Any Model2Vec model with a lower-casing BERT tokenizer and no token weights can be packed (most `minishlab/potion-*`
models). At 4 bits instead of 2, potion-base-8M is 4.1 MB and scores as the full-precision model does:

```sh
npx mini-semantic-search pack ./potion-base-8M ./potion-base-8M-4bit.bin --bits 4
```

```ts
const search = await MiniSemanticSearch.create({ fields: ['title', 'text'], model: '/potion-base-8M-4bit.bin' });
```

## Command line

```sh
npx mini-semantic-search search docs.json "how do I get a refund" --fields title,text [--k 5]
npx mini-semantic-search embed "some text"                  # the vector as JSON
npx mini-semantic-search pack <model folder> out.bin [--bits 2|4|8|32] [--dims N]
```

## How well it works

On the example FAQ in `examples/` (a made-up shop, 45 entries searched on `section`, `q` and `a`, and 135 casual
test questions), with the right entry ranked first or in the top 3:

| Model | Size | `boost` | First | Top 3 |
|---|---|---|---|---|
| **potion-base-8M at 2 bits (bundled)** | **2.2 MB** | none | **65.2%** | **85.2%** |
| **potion-base-8M at 2 bits (bundled)** | **2.2 MB** | `{ q: 3 }` | **65.2%** | **87.4%** |
| potion-base-8M at 4 bits | 4.1 MB | `{ q: 3 }` | 65.9% | 88.9% |
| potion-base-8M, full precision | 30.5 MB | `{ q: 3 }` | 65.9% | 88.9% |
| 0.1's model: 128 of the 256 dimensions at 4 bits | 2.2 MB | `{ q: 3 }` | 66.7% | 85.9% |

With the bundled model and `{ q: 3 }`, `keywordWeight: 0.3` gets 67.4% and 87.4%, `0` (meaning only) 65.2% and
85.9%, and `1` (words only) 46.7% and 65.2%: these questions mostly share no words with their entries.

The test questions were written with the entries in view, so expect real questions to do somewhat worse. The set
is small: one question is 0.7%.

[NanoBEIR](https://huggingface.co/collections/zeta-alpha-ai/nanobeir-66e1a0af21dfd93e620cd9f6) is bigger and more
varied: 13 public retrieval sets, each with 50 queries over 2,000 to 6,000 documents, from duplicate questions
(Quora) to fact checking against Wikipedia (FEVER). `scripts/nanobeir.mjs` runs it, scoring nDCG@10, against
MiniSearch with its defaults:

| | nDCG@10 | To index a set | Per search | Memory |
|---|---|---|---|---|
| **mini-semantic-search** | **56.4** | 0.6 s | 3.7 ms | 15.5 MB |
| the same with `keywordWeight: 0`, `passageLength: 0` (as 0.2 was) | 45.4 | 0.5 s | 1.6 ms | 11.4 MB |
| MiniSearch 7.2.0 | 51.6 | 0.3 s | 9.1 ms | 33.4 MB |

These are averages over the 13 sets, on one desktop core. Set by set, mini-semantic-search is ahead on nine,
level on two (FEVER, SCIDOCS) and behind on two: HotpotQA (71.6 to 80.7), where the answer turns on a name, and
Touche2020 (66.3 to 76.9), whose documents are long arguments. It is furthest ahead where questions are reworded
(Quora, 93.3 to 73.2). `keywordWeight: 0.7` averages 57.5. Fifty queries are few: a set's score can move several
points by chance, the average by a point or two.

The model on its own, without the index, is scored by two more public benchmarks. In [Banking77](https://huggingface.co/datasets/mteb/banking77),
3,080 customer questions each count as right when the closest of 10,003 other questions has the same intent (of 77).
In [SciFact](https://huggingface.co/datasets/mteb/scifact), 300 claims look for the abstracts about them among 5,183
(nDCG@10). `scripts/eval.mjs` runs both, comparing vectors only (no keywords or passages):

| Model | Size | Banking77 | SciFact |
|---|---|---|---|
| **potion-base-8M at 2 bits (bundled)** | **2.2 MB** | **88.3%** | **50.1** |
| potion-base-8M at 4 bits | 4.1 MB | 88.3% | 50.4 |
| potion-base-8M, full precision | 30.5 MB | 88.5% | 50.6 |
| 0.1's model: 128 of the 256 dimensions at 4 bits | 2.2 MB | 87.3% | 45.9 |

**How the model is packed.** Each value is one of four levels fitted to the values (Lloyd-Max), times a scale for
its dimension and one for its token. The scale per dimension matters because Model2Vec's dimensions are PCA
components whose spread falls about fivefold from the first to the last. At 2 bits a single vector is noisy (cosine
about 0.95 to the full-precision one), but the noise is spread across 256 dimensions and mostly cancels out when
vectors are compared, so it loses less than keeping half the dimensions at 4 bits. The model stays packed in memory,
so it takes about 3 MB of it rather than 30 MB, and loads in about 6 ms.

**What it can't do:**
- Averaging tokens loses word order, so "can I add a shopper" and "can a shopper add me" look alike.
- It only knows the meaning of words its training text had. Brand names and jargon ("amex") match documents that
  have the same word, but not ones that say it differently ("American Express"): for those, add an array field of
  the ways people ask.
- Misspelt words break into word pieces that mean something else and match no keywords, so typos hurt.
- English only.

## Development

```sh
npm install
npm test                       # builds, then runs the tests (Node 21+)
sh scripts/smoke.sh            # packs the package, installs it in an empty project and uses it from there
python scripts/make-fixture.py <potion-base-8M folder>   # rebuilds the reference fixture (needs model2vec)
node scripts/eval.mjs <data folder> <model.bin> ...       # Banking77 and SciFact (see the script for the data)
node --expose-gc scripts/nanobeir.mjs <data folder>       # NanoBEIR against MiniSearch (see the script)
```

The bundled model is rebuilt with
`node dist/cli.js pack <potion-base-8M folder> models/potion-base-8M.bin --bits 2`.

CI (`.github/workflows/ci.yml`) builds every push and pull request on Node 18, 20, 22 and 24, runs the tests (22
and 24) and the install check (all four).

To release, bump the version and push the tag: `npm version patch && git push --follow-tags`. The release workflow
checks that the tag matches `package.json`, tests again, publishes to npm with provenance and creates a GitHub
release. It publishes with npm's trusted publishing (the package's Settings → Trusted publishing: GitHub Actions,
repository `ShakirasHips/model2vec-js`, workflow `release.yml`, environment `npm`), so no token is stored anywhere.

## Credits and licence

The code is MIT-licensed (`LICENSE`). The bundled model is
[potion-base-8M](https://huggingface.co/minishlab/potion-base-8M) by MinishLab, MIT, repacked at 2 bits (see
`models/README.md`). The API follows [MiniSearch](https://github.com/lucaong/minisearch) by Luca Ongaro.

This package reimplements Model2Vec's inference and BERT's tokenizer in TypeScript; it contains no code from
Model2Vec, BERT or MiniSearch.
