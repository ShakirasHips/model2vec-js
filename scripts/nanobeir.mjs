// Scores the library on NanoBEIR (13 retrieval sets of 50 queries each) against MiniSearch, with the time and memory
// each takes, as in the README's table. It needs two packages that the library doesn't, and the data:
//
//   npm install --no-save hyparquet minisearch
//   for d in ArguAna ClimateFEVER DBPedia FEVER FiQA2018 HotpotQA MSMARCO NFCorpus NQ QuoraRetrieval SCIDOCS SciFact Touche2020; do
//     mkdir -p data/nanobeir/$d
//     for p in corpus queries qrels; do
//       curl -sfL -o data/nanobeir/$d/$p.parquet https://huggingface.co/datasets/zeta-alpha-ai/Nano$d/resolve/main/$p/train-00000-of-00001.parquet
//     done
//   done
//
//   npm run build && node --expose-gc scripts/nanobeir.mjs data/nanobeir
//
// nDCG@10 is for the documents marked relevant to each query. Times are the best of three runs; memory is what the
// index holds after searching (heap plus array buffers).
import { readFileSync, readdirSync } from 'node:fs';
import { parquetReadObjects } from 'hyparquet';
import MiniSearch from 'minisearch';
import MiniSemanticSearch, { StaticModel } from '../dist/node.js';

const [dataDir] = process.argv.slice(2);
if (!dataDir) throw new Error('usage: node --expose-gc scripts/nanobeir.mjs <data folder>');
if (!globalThis.gc) throw new Error('run node with --expose-gc, to measure memory');
const read = (file) => {
  const b = readFileSync(file);
  return parquetReadObjects({ file: b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength) });
};
const model = await StaticModel.load();
const semantic = (options, searchOptions) => (corpus) => {
  const s = new MiniSemanticSearch({ model, fields: ['text'], idField: '_id', ...options, searchOptions });
  s.addAll(corpus);
  return s;
};
const systems = {
  'meaning only': semantic({ passageLength: 0 }, { keywordWeight: 0 }),
  'defaults': semantic({}, {}),
  'MiniSearch': (corpus) => {
    const s = new MiniSearch({ fields: ['text'], idField: '_id' });
    s.addAll(corpus);
    return s;
  },
};
const memory = async () => {
  for (let i = 0; i < 3; i++) {
    globalThis.gc();
    await new Promise((resolve) => setTimeout(resolve, 30));
  }
  const m = process.memoryUsage();
  return m.heapUsed + m.arrayBuffers;
};

const names = Object.keys(systems), totals = Object.fromEntries(names.map((n) => [n, [0, 0, 0, 0]]));
const sets = readdirSync(dataDir).sort();
const line = (label, cells) => console.log(label.padEnd(16) + cells.map((c) => c.padStart(34)).join(''));
line('', names);
for (const set of sets) {
  const corpus = await read(`${dataDir}/${set}/corpus.parquet`);
  const relevant = new Map();
  for (const r of await read(`${dataDir}/${set}/qrels.parquet`)) relevant.set(r['query-id'], (relevant.get(r['query-id']) ?? new Set()).add(r['corpus-id']));
  const queries = (await read(`${dataDir}/${set}/queries.parquet`)).filter((q) => relevant.has(q._id));
  const cells = [];
  systems[names[0]](corpus); // so that what reading the files left behind is freed before memory is measured
  for (const name of names) {
    systems[name](corpus.slice(0, 200)); // warms the code up
    const before = await memory();
    let index, indexMs = Infinity;
    for (let i = 0; i < 3; i++) {
      index = null;
      const t = performance.now();
      index = systems[name](corpus);
      indexMs = Math.min(indexMs, performance.now() - t);
    }
    for (const q of queries.slice(0, 3)) index.search(q.text);
    let ndcg = 0, queryMs = 0;
    for (const q of queries) {
      let best = Infinity, results = null;
      for (let i = 0; i < 3; i++) {
        const t = performance.now();
        results = index.search(q.text);
        best = Math.min(best, performance.now() - t);
      }
      queryMs += best / queries.length;
      // A query that is itself a document (ArguAna's are) isn't its own answer.
      const found = results.slice(0, 11).map((r) => r.id).filter((id) => id !== q._id).slice(0, 10), docs = relevant.get(q._id);
      let dcg = 0, ideal = 0;
      found.forEach((id, i) => (dcg += docs.has(id) ? 1 / Math.log2(i + 2) : 0));
      for (let i = 0; i < Math.min(10, docs.size); i++) ideal += 1 / Math.log2(i + 2);
      ndcg += (100 * dcg) / ideal / queries.length;
    }
    const mb = ((await memory()) - before) / 1e6;
    index = null;
    [ndcg, indexMs, queryMs, mb].forEach((x, i) => (totals[name][i] += x / sets.length));
    cells.push(`${ndcg.toFixed(1)}  ${indexMs.toFixed(0)} ms  ${queryMs.toFixed(1)} ms  ${mb.toFixed(1)} MB`);
  }
  line(set, cells);
}
line('Average', names.map((n) => `${totals[n][0].toFixed(1)}  ${totals[n][1].toFixed(0)} ms  ${totals[n][2].toFixed(1)} ms  ${totals[n][3].toFixed(1)} MB`));
console.log('\nEach cell: nDCG@10, time to index the set, time per search, memory.');
