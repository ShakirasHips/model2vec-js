import { decodePacked, unpack, type ModelData, type PackedModelData } from './format.js';
import { DEFAULT_MODEL, loadBytes, type Source } from './load.js';
import { wordPiece } from './tokenizer.js';

/**
 * A Model2Vec static embedding model: one vector per token, and a sentence's vector is the mean of its tokens'
 * vectors, normalised to length 1. No network runs, so embedding takes microseconds. Long texts are cut as the
 * Python library cuts them: to `maxTokens` × the median token length in characters, then to `maxTokens` tokens,
 * before unknown tokens are left out.
 *
 * A packed model (2 or 4 bits) stays packed in memory: about 2 MB for the bundled one instead of 30 MB.
 */
export class StaticModel {
  readonly tokens: string[];
  readonly dims: number;
  readonly unk: number;
  /** Tokens read per text; Model2Vec's default is 512. */
  maxTokens = 512;
  private readonly vocab: Map<string, number>;
  private readonly medianTokenLength: number;
  private readonly longestPiece: number;
  private table: Float32Array | null;
  private readonly packed: (PackedModelData & { lookup: Float32Array; perByte: number; rowBytes: number }) | null;

  constructor(data: ModelData | PackedModelData) {
    const { tokens, dims } = data;
    this.tokens = tokens;
    this.dims = dims;
    this.vocab = new Map(tokens.map((t, i) => [t, i]));
    this.unk = this.vocab.get('[UNK]') ?? -1;
    // Token lengths in characters, counted (they are short), for the median and the longest word piece.
    const counts: number[] = [];
    let longest = 0;
    for (const t of tokens) {
      const len = t.length === 1 || !/[\uD800-\uDFFF]/.test(t) ? t.length : Array.from(t).length;
      counts[len] = (counts[len] ?? 0) + 1;
      longest = Math.max(longest, len);
    }
    const nth = (k: number): number => {
      for (let len = 0, seen = 0; ; len++) if ((seen += counts[len] ?? 0) > k) return len;
    };
    const mid = tokens.length >> 1;
    this.medianTokenLength = Math.trunc(tokens.length % 2 ? nth(mid) : (nth(mid - 1) + nth(mid)) / 2);
    this.longestPiece = longest;
    if ('vectors' in data) {
      this.table = data.vectors;
      this.packed = null;
    } else {
      // Each code byte holds `perByte` values; `lookup` gives their levels for every possible byte.
      const perByte = 8 / data.bits, mask = (1 << data.bits) - 1;
      const lookup = new Float32Array(256 * perByte);
      for (let b = 0; b < 256; b++) for (let j = 0; j < perByte; j++) lookup[b * perByte + j] = data.levels[(b >> (j * data.bits)) & mask];
      this.table = null;
      this.packed = { ...data, lookup, perByte, rowBytes: (dims * data.bits) / 8 };
    }
  }

  /** Loads a model file; by default the bundled potion-base-8M (256 dimensions at 2 bits, 2.2 MB). */
  static async load(src: Source = DEFAULT_MODEL): Promise<StaticModel> {
    return StaticModel.fromBytes(await loadBytes(src));
  }

  static fromBytes(bytes: Uint8Array): StaticModel {
    return new StaticModel(decodePacked(bytes));
  }

  /** The table of token vectors, tokens.length × dims, row by row (unpacked on first use for a packed model). */
  get vectors(): Float32Array {
    return (this.table ??= unpack(this.packed!).vectors);
  }

  /** Token ids, as Hugging Face's tokenizer gives them without special tokens (unknown words included as UNK). */
  tokenize(text: string): number[] {
    return wordPiece(this.vocab, this.unk, text, this.longestPiece);
  }

  /** Whether `token` (e.g. "voucher" or "##ing") is in the vocabulary. */
  has(token: string): boolean {
    return this.vocab.has(token);
  }

  /** The text's vector: unit length, or all zeros if it has no known tokens. */
  embed(text: string): Float32Array {
    const chars = this.maxTokens * this.medianTokenLength;
    if (text.length > chars) text = Array.from(text).slice(0, chars).join('');
    const ids = this.tokenize(text);
    return this.embedIds(ids, 0, Math.min(ids.length, this.maxTokens));
  }

  /** The vector of the token ids from `start` up to `end` (as `tokenize` gives them): what `embed` makes of a text. */
  embedIds(ids: ArrayLike<number>, start = 0, end = ids.length): Float32Array {
    const { dims, packed } = this;
    const v = new Float32Array(dims);
    if (packed) {
      const { codes, rowScales, lookup, perByte, rowBytes, colScales } = packed;
      for (let t = start; t < end; t++) {
        const id = ids[t];
        if (id === this.unk) continue;
        const r = rowScales[id];
        if (perByte === 4) {
          for (let p = id * rowBytes, d = 0; d < dims; p++, d += 4) {
            const o = codes[p] << 2;
            v[d] += lookup[o] * r;
            v[d + 1] += lookup[o + 1] * r;
            v[d + 2] += lookup[o + 2] * r;
            v[d + 3] += lookup[o + 3] * r;
          }
        } else {
          for (let p = id * rowBytes, d = 0; d < dims; p++, d += 2) {
            const o = codes[p] << 1;
            v[d] += lookup[o] * r;
            v[d + 1] += lookup[o + 1] * r;
          }
        }
      }
      for (let d = 0; d < dims; d++) v[d] *= colScales[d];
    } else {
      const table = this.table!;
      for (let t = start; t < end; t++) {
        const id = ids[t];
        if (id === this.unk) continue;
        for (let d = 0, o = id * dims; d < dims; d++) v[d] += table[o + d];
      }
    }
    let norm = 0;
    for (let d = 0; d < dims; d++) norm += v[d] * v[d];
    norm = Math.sqrt(norm) || 1;
    for (let d = 0; d < dims; d++) v[d] /= norm;
    return v;
  }

  embedMany(texts: string[]): Float32Array[] {
    return texts.map((t) => this.embed(t));
  }
}

/** Dot product: the cosine similarity of two unit vectors, such as `embed` returns (1 = same meaning). */
export function similarity(a: Float32Array, b: Float32Array): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}
