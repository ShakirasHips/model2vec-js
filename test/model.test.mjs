import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';
import { decodeModel, encodeModel, similarity, StaticModel } from '../dist/node.js';

const ref = JSON.parse(readFileSync(new URL('fixtures/reference.json', import.meta.url), 'utf8'));
const model = await StaticModel.load();

test('tokenizes exactly as Hugging Face does', () => {
  ref.sentences.forEach((s, i) => assert.deepEqual(model.tokenize(s), ref.ids[i], JSON.stringify(s)));
});

// The fixture has all 256 dimensions at full precision. At 2 bits a single vector is noisy (cosine about 0.95 to the
// full-precision one), but the noise is spread over 256 dimensions, so rankings barely move.
const unit = (row) => {
  const v = Float32Array.from(row);
  const n = Math.hypot(...v) || 1;
  return v.map((x) => x / n);
};

test('embeds like the Python library (2 bits, so cosine >= 0.9)', () => {
  assert.equal(model.dims, ref.embeddings[0].length);
  ref.sentences.forEach((s, i) => {
    const r = unit(ref.embeddings[i]);
    const v = model.embed(s);
    if (r.every((x) => x === 0)) assert.ok(v.every((x) => x === 0), `${JSON.stringify(s)} should embed to zeros`);
    else assert.ok(similarity(v, r) >= 0.9, `${JSON.stringify(s).slice(0, 60)}: ${similarity(v, r)}`);
  });
});

test('a packed model embeds as its unpacked table does', () => {
  const unpacked = new StaticModel({ tokens: model.tokens, dims: model.dims, vectors: model.vectors });
  assert.equal(model.vectors.length, model.tokens.length * model.dims);
  for (const s of ref.sentences) {
    const a = model.embed(s), b = unpacked.embed(s);
    assert.ok(a.every((x, i) => Math.abs(x - b[i]) < 1e-5), JSON.stringify(s).slice(0, 60));
  }
});

test('embeddings have unit length', () => {
  const v = model.embed('how do I return an item');
  assert.ok(Math.abs(similarity(v, v) - 1) < 1e-5);
});

test('related sentences score higher than unrelated ones', () => {
  const a = model.embed('how do I cancel my order');
  assert.ok(similarity(a, model.embed('stop my purchase before it ships')) > similarity(a, model.embed('the weather is nice today')));
});

test('model files round-trip at 32, 8, 4 and 2 bits', () => {
  // Rows of normal values whose spread falls off across the dimensions, as Model2Vec's do.
  let seed = 1;
  const rand = () => (seed = (seed * 1103515245 + 12345) % 2 ** 31) / 2 ** 31;
  const normal = () => Math.sqrt(-2 * Math.log(rand() + 1e-12)) * Math.cos(2 * Math.PI * rand());
  const tokens = Array.from({ length: 200 }, (_, i) => (i ? `t${i}` : '[UNK]'));
  const dims = 64;
  const vectors = Float32Array.from({ length: tokens.length * dims }, (_, i) => normal() * (3 - (2 * (i % dims)) / dims));
  // Lloyd-Max levels for normal values err by about 0.096 (4 bits) and 0.34 (2 bits) of the values' size.
  for (const [bits, tol] of [[32, 0], [8, 0.01], [4, 0.11], [2, 0.36]]) {
    const bytes = encodeModel({ tokens, dims, vectors }, { bits, group: 32 });
    const back = decodeModel(bytes);
    assert.deepEqual(back.tokens, tokens);
    assert.equal(back.dims, dims);
    let err = 0, size = 0;
    back.vectors.forEach((x, i) => {
      err += (x - vectors[i]) ** 2;
      size += vectors[i] ** 2;
    });
    assert.ok(Math.sqrt(err / size) <= tol, `${bits} bits: error ${Math.sqrt(err / size)}`);
    // As a Node Buffer too, whose `slice` is a view rather than a copy.
    assert.deepEqual(decodeModel(Buffer.concat([Buffer.alloc(8), bytes]).subarray(8)).vectors, back.vectors);
    const packed = StaticModel.fromBytes(bytes), unpacked = new StaticModel(back);
    assert.deepEqual(Array.from(packed.embed('t1 t2 t3'), (x) => x.toFixed(5)), Array.from(unpacked.embed('t1 t2 t3'), (x) => x.toFixed(5)));
  }
  assert.throws(() => encodeModel({ tokens, dims: 63, vectors: vectors.subarray(0, 200 * 63) }, { bits: 2 }), /whole number of bytes/);
  assert.throws(() => decodeModel(new Uint8Array(40)), /not a model2vec-js model file/);
});

test('reads version 1 files (4 bits, a scale per group), as 0.1 packed them', () => {
  const tokens = ['[UNK]', 'hello', 'world'], dims = 4, group = 4, nibbles = [[-8, -1, 0, 7], [1, 2, 3, 4], [-3, 0, 3, 0]];
  const vocab = new TextEncoder().encode(tokens.join('\n')), pad = (n) => (n + 3) & ~3;
  const bytes = new Uint8Array(28 + pad(vocab.length) + 8 + 6), dv = new DataView(bytes.buffer);
  [0x4d514146, 1, 3, dims, 4, group, vocab.length].forEach((v, i) => dv.setUint32(i * 4, v, true));
  bytes.set(vocab, 28);
  let off = 28 + pad(vocab.length);
  [0x3c00, 0x4000, 0x3800].forEach((h, i) => dv.setUint16(off + i * 2, h, true)); // scales 1, 2 and 0.5
  off += 8;
  nibbles.flat().forEach((q, i) => (bytes[off + (i >> 1)] |= (q + 8) << ((i & 1) * 4)));
  assert.deepEqual(Array.from(decodeModel(bytes).vectors), [-8, -1, 0, 7, 2, 4, 6, 8, -1.5, 0, 1.5, 0]);
  const m = StaticModel.fromBytes(bytes);
  assert.deepEqual(Array.from(m.embed('hello'), (x) => +x.toFixed(4)), [0.1826, 0.3651, 0.5477, 0.7303]);
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
  for (const f of ['index', 'model', 'search', 'tokenizer', 'format', 'load']) {
    const src = readFileSync(new URL(`../dist/${f}.js`, import.meta.url), 'utf8');
    assert.doesNotMatch(src, /from ['"]node:|import\(['"]node:/, f);
    for (const [, dep] of src.matchAll(/from '\.\/([\w-]+)\.js'/g)) assert.notEqual(dep, 'node', `${f} imports node.js`);
  }
});
