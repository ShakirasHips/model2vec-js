import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';
import { decodeModel, encodeModel, similarity, StaticModel } from '../dist/node.js';

const ref = JSON.parse(readFileSync(new URL('fixtures/reference.json', import.meta.url), 'utf8'));
const model = await StaticModel.load();

test('tokenizes exactly as Hugging Face does', () => {
  ref.sentences.forEach((s, i) => assert.deepEqual(model.tokenize(s), ref.ids[i], JSON.stringify(s)));
});

test('embeds like the Python library (4-bit, so cosine >= 0.985)', () => {
  ref.sentences.forEach((s, i) => {
    const r = Float32Array.from(ref.embeddings[i]);
    const v = model.embed(s);
    if (r.every((x) => x === 0)) assert.ok(v.every((x) => x === 0), `${JSON.stringify(s)} should embed to zeros`);
    else assert.ok(similarity(v, r) >= 0.985, `${JSON.stringify(s).slice(0, 60)}: ${similarity(v, r)}`);
  });
});

test('embeddings have unit length', () => {
  const v = model.embed('how do I return an item');
  assert.ok(Math.abs(similarity(v, v) - 1) < 1e-5);
});

test('related sentences score higher than unrelated ones', () => {
  const a = model.embed('how do I cancel my order');
  assert.ok(similarity(a, model.embed('stop my purchase before it ships')) > similarity(a, model.embed('the weather is nice today')));
});

test('model files round-trip at 32, 8 and 4 bits', () => {
  const tokens = ['[PAD]', '[UNK]', 'hello', 'world', '##s'];
  const dims = 64;
  const vectors = Float32Array.from({ length: tokens.length * dims }, (_, i) => Math.sin(i * 0.7) * (1 + (i % 5)));
  for (const [bits, tol] of [[32, 0], [8, 0.03], [4, 0.6]]) {
    const back = decodeModel(encodeModel({ tokens, dims, vectors }, { bits, group: 32 }));
    assert.deepEqual(back.tokens, tokens);
    assert.equal(back.dims, dims);
    const err = Math.max(...back.vectors.map((x, i) => Math.abs(x - vectors[i])));
    assert.ok(err <= tol, `${bits} bits: error ${err}`);
  }
  assert.throws(() => decodeModel(new Uint8Array(40)), /not a model2vec-js model file/);
});

test('loads from bytes, a path and a file URL', async () => {
  const path = new URL('../models/potion-base-8M.bin', import.meta.url);
  const fromBytes = StaticModel.fromBytes(readFileSync(path));
  const fromPath = await StaticModel.load(path.pathname);
  const fromUrl = await StaticModel.load(path);
  assert.deepEqual(fromPath.embed('refund'), fromBytes.embed('refund'));
  assert.deepEqual(fromUrl.embed('refund'), fromBytes.embed('refund'));
});

test('the browser build does not import Node modules', () => {
  for (const f of ['index', 'model', 'faq', 'speller', 'tokenizer', 'format', 'load', 'evaluate', 'off-topic']) {
    const src = readFileSync(new URL(`../dist/${f}.js`, import.meta.url), 'utf8');
    assert.doesNotMatch(src, /from ['"]node:|import\(['"]node:/, f);
    for (const [, dep] of src.matchAll(/from '\.\/([\w-]+)\.js'/g)) assert.notEqual(dep, 'node', `${f} imports node.js`);
  }
});
