# mini-semantic-search

Semantic search with [MiniSearch](https://github.com/lucaong/minisearch)'s API. Documents are matched by meaning
rather than shared words: "stop sending me newsletters" finds "How do I unsubscribe from your emails?", and "the mug
came smashed" finds "My item arrived damaged or broken". It runs in browsers and Node with no dependencies, no
server and no API calls, and the package is 2.2 MB with its model (1.8 MB gzipped).

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
//   { id: 2, score: 0.39, title: 'Damaged items' },
//   { id: 3, score: 0.15, title: 'Shipping times' },
//   { id: 1, score: 0.08, title: 'Returns' },
// ]
```

The embeddings come from [Model2Vec](https://github.com/MinishLab/model2vec), a static embedding model. It keeps one
vector per token, and a text's vector is the average of its tokens' vectors. There is no neural network to run, so
embedding a document takes about 0.1 ms on a CPU.

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
    import MiniSemanticSearch from 'https://cdn.jsdelivr.net/npm/mini-semantic-search@0.1.0/dist/index.js';
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

**How documents are scored.** Each document is embedded once per field and once with all its fields together. Its
score is half the whole document's cosine similarity to the query, plus half its fields' similarities averaged
(weighted by `boost`). Scores run from about −0.1 to 1. A document always scores something, so every document comes
back: `slice` the results, or set `minScore` to drop the ones too far off to count as a match. Where to set it
depends on your documents' wording. Try your own off-topic questions and put it just above the scores they get.

**How it differs from MiniSearch.** Results have no `terms`, `queryTerms` or `match`, because nothing is matched
word for word. There's no `prefix`, `fuzzy`, `combineWith`, `autoSuggest` or query syntax, and no
`toJSON`/`loadJSON`: indexing costs about 0.1 ms per document, so you can just index again on load.

Indexing takes 0.05–0.2 ms per document. A search over 10,000 documents takes about 8 ms.

## Embeddings

```ts
import { StaticModel, similarity } from 'mini-semantic-search';

const model = await StaticModel.load();            // or StaticModel.load(urlOrPathOrBytes)
const a = model.embed('how do I cancel my order');  // Float32Array(128), unit length
const b = model.embed('stop my purchase before it ships');
similarity(a, b);                                   // cosine similarity, 1 = same meaning
model.embedMany(['refund', 'exchange']);           // several at once
model.tokenize('Café');                             // token ids
```

This matches the Python library. Token ids are identical to Hugging Face's tokenizer, and with a full-precision
model the vectors match `model2vec` to within 1e-6. The tests check the token ids and the bundled model's vectors
against fixtures made with `scripts/make-fixture.py`.

Any Model2Vec model with a lower-casing BERT tokenizer and no token weights can be packed (most `minishlab/potion-*`
models). A bigger model is more accurate. The full 256 dimensions of potion-base-8M at 4 bits are 4.2 MB:

```sh
npx mini-semantic-search pack ./potion-base-8M ./potion-base-8M.bin --bits 4
```

```ts
const search = await MiniSemanticSearch.create({ fields: ['title', 'text'], model: '/potion-base-8M.bin' });
```

## Command line

```sh
npx mini-semantic-search search docs.json "how do I get a refund" --fields title,text [--k 5]
npx mini-semantic-search embed "some text"                  # the vector as JSON
npx mini-semantic-search pack <model folder> out.bin [--bits 4|8|32] [--group 64] [--dims N]
```

## How well it works

On the example FAQ in `examples/` (a made-up shop, 45 entries searched on `section`, `q` and `a`, and 135 casual
test questions), with the right entry ranked first or in the top 3:

| Model | Size | `boost` | First | Top 3 |
|---|---|---|---|---|
| potion-base-8M, 256 dims at 4 bits | 4.2 MB | `{ q: 3 }` | 63.0% | 87.4% |
| **potion-base-8M, 128 dims at 4 bits (bundled)** | **2.2 MB** | none | **59.3%** | **80.0%** |
| **potion-base-8M, 128 dims at 4 bits (bundled)** | **2.2 MB** | `{ q: 3 }` | **60.7%** | **83.7%** |

The test questions were written with the entries in view, so expect real questions to do somewhat worse. The set
is small: one question is 0.7%. The bundled model keeps potion-base-8M's first 128 dimensions (its PCA components
with the most variance), and its vectors are within cosine 0.98 of the full-precision ones cut to 128.

**What it can't do:**
- Averaging tokens loses word order, so "can I add a shopper" and "can a shopper add me" look alike.
- It only knows words its training text had. Brand names and jargon ("amex") need other phrasings, such as an
  array field of the ways people ask.
- Misspelt words break into word pieces that mean something else, so typos hurt.
- English only.

## Development

```sh
npm install
npm test                       # builds, then runs the tests (Node 21+)
sh scripts/smoke.sh            # packs the package, installs it in an empty project and uses it from there
python scripts/make-fixture.py <potion-base-8M folder>   # rebuilds the reference fixture (needs model2vec)
```

The bundled model is rebuilt with
`node dist/cli.js pack <potion-base-8M folder> models/potion-base-8M-128d.bin --bits 4 --dims 128 --group 128`.

CI (`.github/workflows/ci.yml`) builds every push and pull request on Node 18, 20, 22 and 24, runs the tests (22
and 24) and the install check (all four).

To release, bump the version and push the tag: `npm version patch && git push --follow-tags`. The release workflow
checks that the tag matches `package.json`, tests again, publishes to npm with provenance and creates a GitHub
release. It publishes with npm's trusted publishing (the package's Settings → Trusted publishing: GitHub Actions,
repository `ShakirasHips/model2vec-js`, workflow `release.yml`, environment `npm`), so no token is stored anywhere.

## Credits and licence

The code is MIT-licensed (`LICENSE`). The bundled model is
[potion-base-8M](https://huggingface.co/minishlab/potion-base-8M) by MinishLab, MIT, repacked at 128 dimensions and
4 bits (see `models/README.md`). The API follows [MiniSearch](https://github.com/lucaong/minisearch) by Luca Ongaro.

This package reimplements Model2Vec's inference and BERT's tokenizer in TypeScript; it contains no code from
Model2Vec, BERT or MiniSearch.
