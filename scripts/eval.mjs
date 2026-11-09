// Scores packed models on two public benchmarks, as in the README's table. Get the data first:
//
//   mkdir -p data/banking77 data/scifact/qrels
//   for f in train.jsonl test.jsonl; do curl -sL -o data/banking77/$f https://huggingface.co/datasets/mteb/banking77/resolve/main/$f; done
//   for f in corpus.jsonl queries.jsonl qrels/test.tsv; do curl -sL -o data/scifact/$f https://huggingface.co/datasets/mteb/scifact/resolve/main/$f; done
//
//   npm run build && node scripts/eval.mjs data models/potion-base-8M.bin [other.bin ...]
//
// Banking77: each test question counts as right when the closest training question has its intent.
// SciFact: nDCG@10 for the abstracts relevant to each test claim.
import { readFileSync } from 'node:fs';
import { StaticModel } from '../dist/node.js';

const [dataDir, ...models] = process.argv.slice(2);
if (!dataDir || !models.length) throw new Error('usage: node scripts/eval.mjs <data folder> <model.bin> [...]');
const read = (f) => readFileSync(`${dataDir}/${f}`, 'utf8').trim().split('\n');
const jsonl = (f) => read(f).map((l) => JSON.parse(l));
const train = jsonl('banking77/train.jsonl'), test = jsonl('banking77/test.jsonl');
const corpus = jsonl('scifact/corpus.jsonl'), claims = new Map(jsonl('scifact/queries.jsonl').map((q) => [q._id, q.text]));
const relevant = new Map();
for (const [q, doc, score] of read('scifact/qrels/test.tsv').slice(1).map((l) => l.split('\t'))) {
  if (+score > 0) relevant.set(q, (relevant.get(q) ?? new Set()).add(doc));
}

/** Row indexes of the k rows of `matrix` closest to `v`, best first. */
function nearest(v, matrix, k) {
  const dims = v.length, best = [];
  for (let r = 0; r < matrix.length / dims; r++) {
    let s = 0;
    for (let d = 0; d < dims; d++) s += v[d] * matrix[r * dims + d];
    if (best.length < k || s > best[best.length - 1][0]) {
      best.push([s, r]);
      best.sort((a, b) => b[0] - a[0]);
      if (best.length > k) best.pop();
    }
  }
  return best.map(([, r]) => r);
}

const embedAll = (model, texts) => {
  const m = new Float32Array(texts.length * model.dims);
  texts.forEach((t, i) => m.set(model.embed(t), i * model.dims));
  return m;
};

for (const file of models) {
  const model = StaticModel.fromBytes(readFileSync(file));
  const trainVectors = embedAll(model, train.map((x) => x.text));
  const right = test.filter((x) => train[nearest(model.embed(x.text), trainVectors, 1)[0]].label === x.label).length;
  const docVectors = embedAll(model, corpus.map((c) => `${c.title} ${c.text}`));
  let ndcg = 0;
  for (const [q, docs] of relevant) {
    const found = nearest(model.embed(claims.get(q)), docVectors, 10);
    let dcg = 0, ideal = 0;
    found.forEach((r, i) => (dcg += docs.has(corpus[r]._id) ? 1 / Math.log2(i + 2) : 0));
    for (let i = 0; i < Math.min(10, docs.size); i++) ideal += 1 / Math.log2(i + 2);
    ndcg += dcg / ideal;
  }
  const mb = (readFileSync(file).length / 1e6).toFixed(1);
  console.log(`${file} (${model.dims} dims, ${mb} MB): Banking77 ${((100 * right) / test.length).toFixed(1)}%, SciFact nDCG@10 ${((100 * ndcg) / relevant.size).toFixed(1)}`);
}
