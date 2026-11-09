// The packed model file. Little-endian. It starts with seven u32s: magic "FAQM", version, vocabulary size, dimensions,
// bits, group size (version 1 only, else 0) and the vocabulary's byte length; then the tokens joined by "\n", padded
// to 4 bytes; then the vectors.
//
// Version 2 (2 or 4 bits): each value is a code into a learned table of 2^bits levels, scaled by its column's scale
// and its row's scale. After the vocabulary come the levels (f32), the column scales (one f32 per dimension), the row
// scales (one f16 per token, padded to 4 bytes) and the codes, each row in dims × bits / 8 bytes, low bits first.
// Model2Vec's dimensions are PCA components whose spread falls about fivefold from the first to the last, so a scale
// per column lets every dimension use all the levels, and a table fitted to the values (Lloyd-Max) puts the levels
// where the values are.
//
// Version 1 (32, 8 or 4 bits), still read: at 32 bits plain f32; at 8 bits each value is an int8, and at 4 bits two
// values share a byte (low nibble first, each nibble − 8); either way every `group` values share one f16 scale, and
// all the scales come first, padded to 4 bytes.

const MAGIC = 0x4d514146;

export interface ModelData {
  tokens: string[];
  dims: number;
  /** tokens.length × dims values, row by row. */
  vectors: Float32Array;
}

/** A version 2 model as stored: value (row, d) = levels[code] × rowScales[row] × colScales[d]. */
export interface PackedModelData {
  tokens: string[];
  dims: number;
  bits: 2 | 4;
  levels: Float32Array;
  colScales: Float32Array;
  rowScales: Float32Array;
  /** dims × bits / 8 bytes per row. */
  codes: Uint8Array;
}

export interface EncodeOptions {
  /** 32, 8, 4 or 2 (default 4). 4 and 2 use a learned table of levels; 8 a scale per group. */
  bits?: 32 | 8 | 4 | 2;
  /** At 8 bits, the values sharing one scale (default 64); must divide the number of dimensions. */
  group?: number;
}

export function fromF16(h: number): number {
  const e = (h >> 10) & 31, m = h & 1023, s = h & 0x8000 ? -1 : 1;
  if (e === 0) return s * m * 2 ** -24;
  if (e === 31) return m ? NaN : s * Infinity;
  return s * (1 + m / 1024) * 2 ** (e - 15);
}

export function toF16(x: number): number {
  const u = new Uint32Array(new Float32Array([x]).buffer)[0];
  const s = (u >> 16) & 0x8000, e = ((u >> 23) & 255) - 127 + 15;
  let m = u & 0x7fffff;
  if (e <= 0) return s | Math.round((m | 0x800000) / 2 ** (14 - e));
  if (e >= 31) return s | 0x7c00;
  m = Math.round(m / 8192);
  return m === 1024 ? s | ((e + 1) << 10) : s | (e << 10) | m;
}

const pad4 = (n: number): number => (n + 3) & ~3;

/** Reads a model file, leaving a version 2 model packed (as `StaticModel` keeps it). */
export function decodePacked(bytes: Uint8Array): ModelData | PackedModelData {
  const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  if (bytes.byteLength < 28 || dv.getUint32(0, true) !== MAGIC) throw new Error('not a model2vec-js model file');
  const [, version, n, dims, bits, group, vocabLen] = Array.from({ length: 7 }, (_, i) => dv.getUint32(i * 4, true));
  if (version !== 1 && version !== 2) throw new Error(`unsupported model file version ${version}; a newer mini-semantic-search may read it`);
  let off = 28;
  const tokens = new TextDecoder().decode(bytes.subarray(off, off + vocabLen)).split('\n');
  if (tokens.length !== n) throw new Error('model file is corrupt: vocabulary size');
  off += pad4(vocabLen);
  const count = n * dims;
  if (version === 2) {
    if (bits !== 2 && bits !== 4) throw new Error(`unsupported bits: ${bits}`);
    const k = 1 << bits, rowBytes = (dims * bits) / 8;
    if (bytes.byteLength < off + 4 * (k + dims) + pad4(n * 2) + n * rowBytes) throw new Error('model file is corrupt: too short');
    const f32 = (len: number): Float32Array => {
      const a = new Float32Array(len);
      for (let i = 0; i < len; i++, off += 4) a[i] = dv.getFloat32(off, true);
      return a;
    };
    const levels = f32(k), colScales = f32(dims);
    const rowScales = new Float32Array(n);
    for (let i = 0; i < n; i++) rowScales[i] = fromF16(dv.getUint16(off + i * 2, true));
    off += pad4(n * 2);
    return { tokens, dims, bits, levels, colScales, rowScales, codes: bytes.subarray(off, off + n * rowBytes) };
  }
  const vectors = new Float32Array(count);
  if (bits === 32) {
    // Copied byte for byte: a Node Buffer's `slice` is a view, so its `.buffer` can hold much more than this.
    new Uint8Array(vectors.buffer).set(bytes.subarray(off, off + count * 4));
  } else if (bits === 8 || bits === 4) {
    const groups = count / group;
    const scales = new Float32Array(groups);
    for (let g = 0; g < groups; g++) scales[g] = fromF16(dv.getUint16(off + g * 2, true));
    off += pad4(groups * 2);
    for (let i = 0; i < count; i++) {
      const q = bits === 8 ? (bytes[off + i] << 24) >> 24 : ((bytes[off + (i >> 1)] >> ((i & 1) * 4)) & 15) - 8;
      vectors[i] = q * scales[(i / group) | 0];
    }
  } else {
    throw new Error(`unsupported bits: ${bits}`);
  }
  return { tokens, dims, vectors };
}

/** The full table of a packed model. */
export function unpack({ tokens, dims, bits, levels, colScales, rowScales, codes }: PackedModelData): ModelData {
  const vectors = new Float32Array(tokens.length * dims), mask = (1 << bits) - 1;
  for (let i = 0; i < vectors.length; i++) {
    const bit = i * bits, code = (codes[bit >> 3] >> (bit & 7)) & mask;
    vectors[i] = levels[code] * rowScales[(i / dims) | 0] * colScales[i % dims];
  }
  return { tokens, dims, vectors };
}

/** Reads a model file into its full table of vectors. */
export function decodeModel(bytes: Uint8Array): ModelData {
  const data = decodePacked(bytes);
  return 'vectors' in data ? data : unpack(data);
}

/** Lloyd-Max: the `k` levels that minimise the squared error of rounding `values` (sorted) to the nearest one. */
function fitLevels(values: Float32Array, k: number, iterations = 30): Float32Array {
  const levels = Float32Array.from({ length: k }, (_, i) => values[Math.floor(((i + 0.5) / k) * values.length)]);
  for (let it = 0; it < iterations; it++) {
    const sum = new Float64Array(k), count = new Float64Array(k);
    for (let i = 0, j = 0; i < values.length; i++) {
      const x = values[i];
      while (j < k - 1 && Math.abs(x - levels[j + 1]) <= Math.abs(x - levels[j])) j++;
      sum[j] += x;
      count[j]++;
    }
    for (let j = 0; j < k; j++) if (count[j]) levels[j] = sum[j] / count[j];
  }
  return levels;
}

function encodeBody(n: number, dims: number, vectors: Float32Array, { bits = 4, group = 64 }: EncodeOptions): Uint8Array {
  if (bits === 32) return new Uint8Array(new Float32Array(vectors).buffer);
  if (bits === 8) {
    if (dims % group) throw new Error(`group (${group}) must divide dims (${dims})`);
    const count = n * dims, groups = count / group, scaleBytes = pad4(groups * 2);
    const body = new Uint8Array(scaleBytes + count), dv = new DataView(body.buffer);
    for (let g = 0; g < groups; g++) {
      let max = 0;
      for (let i = g * group; i < (g + 1) * group; i++) max = Math.max(max, Math.abs(vectors[i]));
      const h = toF16(max / 127), scale = fromF16(h) || 1;
      dv.setUint16(g * 2, h, true);
      for (let i = g * group; i < (g + 1) * group; i++) dv.setInt8(scaleBytes + i, Math.max(-127, Math.min(127, Math.round(vectors[i] / scale))));
    }
    return body;
  }
  if (bits !== 4 && bits !== 2) throw new Error(`unsupported bits: ${bits}`);
  if ((dims * bits) % 8) throw new Error(`dims (${dims}) × bits (${bits}) must be a whole number of bytes`);
  const k = 1 << bits, rowBytes = (dims * bits) / 8;
  // Column scales, then row scales over the column-scaled values, as stored (f32, f16).
  const col = new Float64Array(dims);
  for (let i = 0; i < vectors.length; i++) col[i % dims] += vectors[i] ** 2;
  const colScales = Float32Array.from(col, (s) => Math.sqrt(s / n) || 1);
  const scaled = Float32Array.from(vectors, (x, i) => x / colScales[i % dims]);
  const rowHalves = new Uint16Array(n), rowScales = new Float32Array(n);
  for (let r = 0; r < n; r++) {
    let s = 0;
    for (let d = 0; d < dims; d++) s += scaled[r * dims + d] ** 2;
    rowHalves[r] = toF16(Math.sqrt(s / dims));
    rowScales[r] = fromF16(rowHalves[r]);
    const div = rowScales[r] || 1;
    for (let d = 0; d < dims; d++) scaled[r * dims + d] /= div;
  }
  const levels = fitLevels(scaled.slice().sort(), k);
  const bounds = levels.subarray(1).map((l, j) => (levels[j] + l) / 2);
  const body = new Uint8Array(4 * (k + dims) + pad4(n * 2) + n * rowBytes), dv = new DataView(body.buffer);
  let off = 0;
  for (const x of [...levels, ...colScales]) dv.setFloat32((off += 4) - 4, x, true);
  rowHalves.forEach((h, r) => dv.setUint16(off + r * 2, h, true));
  off += pad4(n * 2);
  for (let i = 0; i < scaled.length; i++) {
    let code = 0;
    while (code < k - 1 && scaled[i] > bounds[code]) code++;
    body[off + ((i * bits) >> 3)] |= code << ((i * bits) & 7);
  }
  return body;
}

export function encodeModel({ tokens, dims, vectors }: ModelData, options: EncodeOptions = {}): Uint8Array {
  const n = tokens.length, bits = options.bits ?? 4;
  if (vectors.length !== n * dims) throw new Error('vectors must hold tokens.length × dims values');
  if (tokens.some((t) => t.includes('\n'))) throw new Error('tokens must not contain newlines');
  const vocab = new TextEncoder().encode(tokens.join('\n'));
  const body = encodeBody(n, dims, vectors, { ...options, bits });
  const version = bits === 4 || bits === 2 ? 2 : 1, group = bits === 8 ? options.group ?? 64 : 0;
  const out = new Uint8Array(28 + pad4(vocab.length) + body.length);
  const dv = new DataView(out.buffer);
  [MAGIC, version, n, dims, bits, group, vocab.length].forEach((v, i) => dv.setUint32(i * 4, v, true));
  out.set(vocab, 28);
  out.set(body, 28 + pad4(vocab.length));
  return out;
}
