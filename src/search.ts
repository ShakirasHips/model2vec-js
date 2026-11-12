import { DEFAULT_MODEL, type Source } from './load.js';
import { StaticModel } from './model.js';

// As in MiniSearch: documents are any objects, and ids any values a Map can key on.
type AnyDocument = any;
type DocumentId = any;

export interface SearchOptions {
  /** Only these fields count (default: all of `fields`). */
  fields?: string[];
  /** Weights of fields against each other, e.g. `{ title: 2 }` (default 1 each). */
  boost?: Record<string, number>;
  /** Only results for which this is true. */
  filter?: (result: SearchResult) => boolean;
  /** Leave out results scoring under this (default: none). */
  minScore?: number;
  /**
   * How much of the score comes from the query's words being in the document, from 0 (meaning only) to 1 (words
   * only). Default 0.5.
   */
  keywordWeight?: number;
}

export interface Options<T = AnyDocument> {
  /** The fields embedded and searched. */
  fields: string[];
  /** The fields returned with each result (default: none). */
  storeFields?: string[];
  /** The field holding each document's unique id (default "id"). */
  idField?: string;
  /** Reads a field from a document (default: `document[fieldName]`). Arrays are embedded item by item. */
  extractField?: (document: T, fieldName: string) => unknown;
  /** Defaults for `search`. */
  searchOptions?: SearchOptions;
  /**
   * Texts longer than this many tokens (about three quarters of a word each) are split into passages of about this
   * length, at sentence ends where there are any, and the closest passage counts. Default 120; 0 to leave texts whole.
   */
  passageLength?: number;
  /** A loaded model, or where to load one from in `create` (default: the bundled potion-base-8M). */
  model?: StaticModel | Source;
}

export type SearchResult = {
  id: DocumentId;
  /** About −0.1 to 1: how close the document is to the query, in meaning and in the words it shares. */
  score: number;
  [storedField: string]: unknown;
};

/**
 * Every vector in the index, at 8 bits: vector v is values[v × dims ...] × scales[v]. That is a quarter of the memory
 * of 32 bits, and scores move by about 1e-4. They share one array because a typed array of its own costs each
 * vector about as much again in bookkeeping.
 */
class Vectors {
  values: Int8Array;
  scales: Float32Array;
  private count = 0;
  private readonly free: number[] = [];

  constructor(readonly dims: number) {
    this.values = new Int8Array(64 * dims);
    this.scales = new Float32Array(64);
  }

  /** Stores the vector and returns its number. */
  add(v: Float32Array): number {
    const { dims } = this;
    let at = this.free.pop();
    if (at === undefined) {
      at = this.count++;
      if (at === this.scales.length) {
        const size = Math.ceil(at * 1.5);
        const values = new Int8Array(size * dims), scales = new Float32Array(size);
        values.set(this.values);
        scales.set(this.scales);
        this.values = values;
        this.scales = scales;
      }
    }
    let max = 0;
    for (let i = 0; i < dims; i++) max = Math.max(max, Math.abs(v[i]));
    const s = max / 127 || 1;
    for (let i = 0, o = at * dims; i < dims; i++) this.values[o + i] = Math.round(v[i] / s);
    this.scales[at] = s;
    return at;
  }

  /** Frees the vector's place for another. */
  remove(at: number): void {
    this.free.push(at);
  }

  /** q · vector `at`. */
  dot(q: Float32Array, at: number): number {
    const v = this.values, o = at * this.dims;
    let a = 0, b = 0, c = 0, d = 0, i = 0;
    for (; i + 3 < q.length; i += 4) {
      a += q[i] * v[o + i];
      b += q[i + 1] * v[o + i + 1];
      c += q[i + 2] * v[o + i + 2];
      d += q[i + 3] * v[o + i + 3];
    }
    for (; i < q.length; i++) a += q[i] * v[o + i];
    return (a + b + c + d) * this.scales[at];
  }
}

/** q · row `o` of m. */
const dot = (q: Float32Array, m: Float32Array, o: number): number => {
  let a = 0, b = 0, c = 0, d = 0, i = 0;
  for (; i + 3 < q.length; i += 4) {
    a += q[i] * m[o + i];
    b += q[i + 1] * m[o + i + 1];
    c += q[i + 2] * m[o + i + 2];
    d += q[i + 3] * m[o + i + 3];
  }
  for (; i < q.length; i++) a += q[i] * m[o + i];
  return a + b + c + d;
};

interface Indexed {
  id: DocumentId;
  /** Per field, the numbers of its vectors: one for each value, or each passage of a long one (none if it's empty). */
  fields: number[][];
  /** The vector of all the fields' text together, or -1 if there is none. */
  whole: number;
  stored: Record<string, unknown>;
  /** Its row in the current plan, or -1 if it has none. */
  row: number;
  /** Its number in the keyword index. */
  num: number;
  /** Per field, how many words (and pieces of words) it has. */
  lens: number[];
}

/**
 * The keyword index: for each key (a token in a field), the documents that have it and how many times. Every list
 * lives in one array, as a chain of blocks [next block, capacity, entries...] that double in size, because tens of
 * thousands of lists as objects of their own would cost more memory than their entries. An entry is the document's
 * number × 256 + its count (at most 255).
 */
class Postings {
  private pool = new Uint32Array(1024);
  private top = 1; // block 0 would look like "no next block"
  /** Per list: its first block, its last block, and how many entries that one holds. */
  private lists = new Uint32Array(3 * 256);
  private readonly ids = new Map<number, number>();

  private block(capacity: number): number {
    const at = this.top;
    this.top += 2 + capacity;
    if (this.top > this.pool.length) {
      const grown = new Uint32Array(Math.max(this.top, Math.ceil(this.pool.length * 1.25)));
      grown.set(this.pool);
      this.pool = grown;
    }
    this.pool[at] = 0;
    this.pool[at + 1] = capacity;
    return at;
  }

  add(key: number, num: number, count: number): void {
    let l = this.ids.get(key);
    if (l === undefined) {
      l = this.ids.size * 3;
      this.ids.set(key, l);
      if (l + 3 > this.lists.length) {
        const grown = new Uint32Array(this.lists.length * 2);
        grown.set(this.lists);
        this.lists = grown;
      }
      this.lists[l] = this.lists[l + 1] = this.block(2);
      this.lists[l + 2] = 0;
    }
    let last = this.lists[l + 1];
    if (this.lists[l + 2] === this.pool[last + 1]) {
      const next = this.block(Math.min(this.pool[last + 1] * 2, 4096));
      this.pool[last] = next;
      this.lists[l + 1] = last = next;
      this.lists[l + 2] = 0;
    }
    this.pool[last + 2 + this.lists[l + 2]++] = num * 256 + Math.min(count, 255);
  }

  /** Calls `each` with every entry of the key's list, oldest first. */
  forEach(key: number, each: (entry: number) => void): void {
    const l = this.ids.get(key);
    if (l === undefined) return;
    const { pool } = this, last = this.lists[l + 1];
    for (let b = this.lists[l]; b; b = pool[b]) {
      const n = b === last ? this.lists[l + 2] : pool[b + 1];
      for (let i = b + 2; i < b + 2 + n; i++) each(pool[i]);
    }
  }

  /** The same lists with the documents renumbered by `to`, leaving out those it gives -1. */
  renumbered(to: Int32Array): Postings {
    const out = new Postings();
    for (const key of this.ids.keys()) {
      this.forEach(key, (e) => {
        const num = to[e >>> 8];
        if (num >= 0) out.add(key, num, e & 255);
      });
    }
    return out;
  }
}

// BM25's usual settings: how fast repeats of a word stop counting, and how much long documents are marked down.
const K1 = 1.2, B = 0.75;

/** Which fields count, and their weights: one choice of search options. */
interface Weighting {
  key: string;
  use: boolean[];
  weights: number[];
  subset: boolean;
}

/**
 * A document's score as terms, each a weight times the best similarity among some vectors (a single-valued field has
 * one): half the whole vector's similarity plus half the fields' similarities averaged by weight, or just the fields'
 * average when only some fields count. Null if no field with a value counts.
 */
function terms(doc: Indexed, { use, weights, subset }: Weighting): [number, number[]][] | null {
  let weight = 0;
  for (let i = 0; i < doc.fields.length; i++) if (use[i] && doc.fields[i].length) weight += weights[i];
  if (!weight) return null;
  // The whole-document vector covers every field, so it only counts when every field does.
  const whole = subset || doc.whole < 0 ? 0 : 0.5;
  const out: [number, number[]][] = whole ? [[whole, [doc.whole]]] : [];
  for (let i = 0; i < doc.fields.length; i++) {
    if (use[i] && doc.fields[i].length) out.push([((1 - whole) * weights[i]) / weight, doc.fields[i]]);
  }
  return out;
}

/**
 * The scores for one weighting as a matrix. A score is linear in the query vector except where an array field's
 * closest item counts, so each document gets one row, its single-vector terms added up. Searching is then one dot
 * product per document, plus one per item of any array field with several.
 */
interface Plan extends Weighting {
  /** One row per document, `dims` long, with room to grow. */
  rows: Float32Array;
  docs: Indexed[];
  /** Per row, the terms with several vectors. */
  multi: ([number, number[]][] | null)[];
}

const texts = (value: unknown): string[] => {
  if (value === undefined || value === null) return [];
  return (Array.isArray(value) ? value : [value]).map(String).filter((s) => s.trim());
};

/**
 * Semantic search with MiniSearch's API: documents are found by meaning as well as shared words, so "stop sending me
 * newsletters" finds "How do I unsubscribe from your emails?". Each document is embedded once per field and once
 * whole. Its meaning score is half its whole vector's cosine similarity to the query, plus half its fields' (weighted
 * by `boost`; an array field or a long text counts its closest item or passage). Its keyword score is BM25 over the
 * model's tokens, as a share of the most the query could score. The two are averaged, weighted by `keywordWeight`.
 * Searching takes no network and no server.
 */
export class MiniSemanticSearch<T = AnyDocument> {
  readonly model: StaticModel;
  private readonly options: Required<Omit<Options<T>, 'model'>>;
  private readonly index = new Map<DocumentId, Indexed>();
  private plan: Plan | null = null;
  /** The weighting of the last search that had no plan: a plan is made when one is used twice running. */
  private lastKey = '';
  private vectors: Vectors;
  // The keyword index. Documents are numbered as they are added; a removed one leaves its number unused (null in
  // `slots`) until `compact` renumbers them.
  private slots: (Indexed | null)[] = [];
  private dead = 0;
  /** Keyed by token id × number of fields + field. */
  private postings = new Postings();
  /** Per field, the words in all the documents. */
  private readonly fieldWords: number[];
  /** Per token id: 1 for a word or a piece of one, 0 for punctuation and special tokens. */
  private readonly isWord: Uint8Array;
  /** Per token id: 1 if it ends a sentence. */
  private readonly endsSentence: Uint8Array;
  /** Per token id, zeros between uses: counts while a document is added. */
  private readonly counts: Uint32Array;

  /** Loads the model (the bundled one unless `options.model` says otherwise), then makes the index. */
  static async create<T = AnyDocument>(options: Options<T>): Promise<MiniSemanticSearch<T>> {
    const { model = DEFAULT_MODEL } = options;
    return new MiniSemanticSearch({ ...options, model: model instanceof StaticModel ? model : await StaticModel.load(model) });
  }

  /** For a model you have loaded already (`StaticModel.load()`); `create` loads it for you. */
  constructor(options: Options<T> & { model: StaticModel }) {
    const { model, fields, storeFields = [], idField = 'id', extractField = (d, f) => (d as Record<string, unknown>)[f], searchOptions = {}, passageLength = 120 } = options;
    if (!(model instanceof StaticModel)) throw new Error('MiniSemanticSearch: the constructor needs a loaded model; use MiniSemanticSearch.create to load one');
    if (!fields?.length) throw new Error('MiniSemanticSearch: option "fields" must list at least one field');
    this.model = model;
    this.vectors = new Vectors(model.dims);
    this.options = { fields, storeFields, idField, extractField, searchOptions, passageLength };
    const n = model.tokens.length;
    this.fieldWords = fields.map(() => 0);
    this.counts = new Uint32Array(n);
    this.isWord = new Uint8Array(n);
    for (let i = 0; i < n; i++) {
      const t = model.tokens[i];
      this.isWord[i] = i !== model.unk && !/^\[[A-Za-z]+\]$/.test(t) && /[\p{L}\p{N}]/u.test(t) ? 1 : 0;
    }
    this.endsSentence = new Uint8Array(n);
    for (const id of model.tokenize('. ! ?')) if (id !== model.unk) this.endsSentence[id] = 1;
  }

  /** How many documents are indexed. */
  get documentCount(): number {
    return this.index.size;
  }

  /** Adds a document; it can be searched straight away. Embedding takes about 0.05 ms. */
  add(document: T): void {
    const { fields, storeFields, idField, extractField } = this.options;
    const id = extractField(document, idField);
    if (id === undefined) throw new Error(`MiniSemanticSearch: document does not have ID field "${idField}"`);
    if (this.index.has(id)) throw new Error(`MiniSemanticSearch: duplicate ID ${id}`);
    if (this.slots.length >= 2 ** 24) this.compact(); // document numbers have 24 bits
    if (this.slots.length >= 2 ** 24) throw new Error('MiniSemanticSearch: the index is full (16,777,216 documents)');
    // Each text is tokenized once; its passages' vectors, the whole document's and the keyword counts come from that.
    const ids = fields.map((f) => texts(extractField(document, f)).map((s) => this.model.tokenize(s)));
    const vectors = ids.map((values) => values.flatMap((t) => this.passages(t).map(([a, b]) => this.vectors.add(this.model.embedIds(t, a, b)))));
    const flat = ids.flat(), { maxTokens } = this.model;
    let whole = -1;
    // One short text is its field's vector and the whole document's.
    if (flat.length === 1 && flat[0].length <= maxTokens && vectors.flat().length === 1) whole = vectors.flat()[0];
    else if (flat.length) {
      const all: number[] = [];
      for (const t of flat) for (let i = 0; i < t.length && all.length < maxTokens; i++) all.push(t[i]);
      whole = this.vectors.add(this.model.embedIds(all));
    }
    const doc: Indexed = {
      id,
      fields: vectors,
      whole,
      stored: Object.fromEntries(storeFields.map((f) => [f, extractField(document, f)])),
      row: -1,
      num: this.slots.length,
      lens: ids.map((values, f) => this.count(values, f, this.slots.length)),
    };
    this.slots.push(doc);
    this.index.set(id, doc);
    if (this.plan) this.place(this.plan, doc);
  }

  /** Where to cut token ids into passages, as [start, end) pairs: at the last sentence end within `passageLength`. */
  private passages(ids: number[]): [number, number][] {
    const size = this.options.passageLength;
    if (!(size > 0) || ids.length <= size) return [[0, ids.length]];
    const out: [number, number][] = [];
    let start = 0, sentence = 0;
    for (let i = 0; i < ids.length; i++) {
      if (this.endsSentence[ids[i]]) sentence = i + 1;
      if (i + 1 - start < size) continue;
      const cut = sentence > start ? sentence : i + 1;
      out.push([start, cut]);
      start = cut;
    }
    // A short tail joins the passage before it: a few words alone would match too easily.
    if (ids.length - start >= size / 4) out.push([start, ids.length]);
    else if (start < ids.length) out[out.length - 1][1] = ids.length;
    return out;
  }

  /** Adds a field's words to the keyword index for document `num`; returns how many there are. */
  private count(values: number[][], field: number, num: number): number {
    const { counts, isWord, postings } = this, nFields = this.options.fields.length;
    const seen: number[] = [];
    let words = 0;
    for (const ids of values) {
      for (const id of ids) {
        if (!isWord[id]) continue;
        words++;
        if (counts[id]++ === 0) seen.push(id);
      }
    }
    for (const id of seen) {
      postings.add(id * nFields + field, num, counts[id]);
      counts[id] = 0;
    }
    this.fieldWords[field] += words;
    return words;
  }

  /** Renumbers the documents without the removed ones' numbers, and drops those from the keyword index. */
  private compact(): void {
    const to = new Int32Array(this.slots.length).fill(-1);
    const slots: Indexed[] = [];
    for (const doc of this.slots) {
      if (!doc) continue;
      to[doc.num] = doc.num = slots.length;
      slots.push(doc);
    }
    this.postings = this.postings.renumbered(to);
    this.slots = slots;
    this.dead = 0;
  }

  /**
   * Per document number, BM25 for the query's words as a share of the most the query could score (0 to 1), or null
   * if the query has no words. A word's count in a document is its counts in the fields added up by weight (BM25F),
   * the weights averaging 1.
   */
  private keywordScores(query: number[], { use, weights }: Weighting): Float32Array | null {
    const { slots, postings, isWord, fieldWords } = this, nFields = use.length, live = this.index.size;
    const words = [...new Set(query.filter((id) => isWord[id]))];
    let total = 0, used = 0;
    for (let f = 0; f < nFields; f++) if (use[f]) (total += weights[f]), used++;
    if (!words.length || !live || !(total > 0)) return null;
    const w = weights.map((x, f) => (use[f] ? (x * used) / total : 0));
    let mean = 0;
    for (let f = 0; f < nFields; f++) mean += (w[f] * fieldWords[f]) / live;
    if (!mean) return null;
    const scores = new Float32Array(slots.length), tf = new Float32Array(slots.length), found = new Uint32Array(slots.length);
    let most = 0;
    for (const id of words) {
      let df = 0;
      for (let f = 0; f < nFields; f++) {
        const weight = w[f];
        if (!weight) continue;
        postings.forEach(id * nFields + f, (e) => {
          const num = e >>> 8;
          if (!slots[num]) return;
          if (tf[num] === 0) found[df++] = num;
          tf[num] += weight * (e & 255);
        });
      }
      const idf = Math.log(1 + (live - df + 0.5) / (df + 0.5));
      most += idf;
      for (let i = 0; i < df; i++) {
        const num = found[i], { lens } = slots[num]!;
        let len = 0;
        for (let f = 0; f < nFields; f++) len += w[f] * lens[f];
        scores[num] += (idf * tf[num]) / (tf[num] + K1 * (1 - B + (B * len) / mean));
        tf[num] = 0;
      }
    }
    for (let i = 0; i < scores.length; i++) scores[i] /= most;
    return scores;
  }

  addAll(documents: readonly T[]): void {
    for (const d of documents) this.add(d);
  }

  /** Adds documents `chunkSize` at a time, yielding between chunks so a page stays responsive. */
  async addAllAsync(documents: readonly T[], { chunkSize = 100 }: { chunkSize?: number } = {}): Promise<void> {
    for (let i = 0; i < documents.length; i += chunkSize) {
      this.addAll(documents.slice(i, i + chunkSize));
      await new Promise((resolve) => setTimeout(resolve, 0));
    }
  }

  /** Removes a document, found by its id. */
  remove(document: T): void {
    this.discard(this.options.extractField(document, this.options.idField));
  }

  /** Removes the document with this id. */
  discard(id: DocumentId): void {
    const doc = this.index.get(id);
    if (!doc) throw new Error(`MiniSemanticSearch: cannot discard document with ID ${id}: it is not in the index`);
    this.unplace(doc);
    this.index.delete(id);
    for (const v of new Set(doc.fields.flat().concat(doc.whole))) if (v >= 0) this.vectors.remove(v);
    this.slots[doc.num] = null;
    doc.lens.forEach((n, f) => (this.fieldWords[f] -= n));
    if (++this.dead > 32 && this.dead > this.index.size) this.compact();
  }

  discardAll(ids: readonly DocumentId[]): void {
    for (const id of ids) this.discard(id);
  }

  /** Removes the given documents, or every document if none are given. */
  removeAll(documents?: readonly T[]): void {
    if (documents === undefined) {
      this.index.clear();
      this.plan = null;
      this.vectors = new Vectors(this.model.dims);
      this.slots = [];
      this.dead = 0;
      this.postings = new Postings();
      this.fieldWords.fill(0);
    } else for (const d of documents) this.remove(d);
  }

  /** Replaces the document that has the same id. */
  replace(document: T): void {
    this.remove(document);
    this.add(document);
  }

  has(id: DocumentId): boolean {
    return this.index.has(id);
  }

  /** The document's `storeFields`, or undefined if it isn't indexed. */
  getStoredFields(id: DocumentId): Record<string, unknown> | undefined {
    return this.index.get(id)?.stored;
  }

  /** Every document, best first, with its id, score and stored fields. */
  search(query: string, searchOptions: SearchOptions = {}): SearchResult[] {
    const { fields: only, boost = {}, filter, minScore = -Infinity, keywordWeight = 0.5 } = { ...this.options.searchOptions, ...searchOptions };
    const { fields } = this.options;
    const use = fields.map((f) => !only || only.includes(f));
    const weights = fields.map((f, i) => (use[i] ? boost[f] ?? 1 : 0));
    const weighting: Weighting = { key: JSON.stringify([use, weights]), use, weights, subset: use.some((u) => !u) };
    const ids = this.model.tokenize(query);
    const q = this.model.embedIds(ids, 0, Math.min(ids.length, this.model.maxTokens));
    const share = Math.min(Math.max(keywordWeight, 0), 1);
    const keywords = share ? this.keywordScores(ids, weighting) : null;
    const results: SearchResult[] = [];
    const emit = (doc: Indexed, meaning: number): void => {
      const score = (1 - share) * meaning + (keywords ? share * keywords[doc.num] : 0);
      if (score < minScore) return;
      const result: SearchResult = { id: doc.id, score, ...doc.stored }; // stored fields come last, as in MiniSearch
      if (!filter || filter(result)) results.push(result);
    };
    const best = (items: number[]): number => {
      let b = -Infinity;
      for (const v of items) b = Math.max(b, this.vectors.dot(q, v));
      return b;
    };
    const plan = this.planFor(weighting);
    if (plan) {
      const { rows, docs, multi } = plan;
      for (let r = 0, o = 0; r < docs.length; r++, o += q.length) {
        let score = dot(q, rows, o);
        const m = multi[r];
        if (m) for (const [w, items] of m) score += w * best(items);
        emit(docs[r], score);
      }
    } else {
      for (const doc of this.index.values()) {
        const t = terms(doc, weighting);
        if (!t) continue;
        let score = 0;
        for (const [w, items] of t) score += w * best(items);
        emit(doc, score);
      }
    }
    return results.sort((a, b) => b.score - a.score);
  }

  /** The plan for this weighting: the current one, a new one if it was also the last search's, or else none. */
  private planFor(weighting: Weighting): Plan | null {
    if (this.plan?.key === weighting.key) return this.plan;
    if (this.lastKey !== weighting.key) {
      this.lastKey = weighting.key;
      return null;
    }
    const plan: Plan = { ...weighting, rows: new Float32Array(Math.max(this.index.size, 16) * this.model.dims), docs: [], multi: [] };
    this.plan = plan;
    for (const doc of this.index.values()) this.place(plan, doc);
    return plan;
  }

  /** Adds the document's row to the plan. */
  private place(plan: Plan, doc: Indexed): void {
    doc.row = -1;
    const t = terms(doc, plan);
    if (!t) return;
    const dims = this.model.dims, n = plan.docs.length;
    if ((n + 1) * dims > plan.rows.length) {
      const grown = new Float32Array(plan.rows.length * 2);
      grown.set(plan.rows);
      plan.rows = grown;
    }
    const row = plan.rows.subarray(n * dims, (n + 1) * dims).fill(0);
    let multi: [number, number[]][] | null = null;
    const { values, scales } = this.vectors;
    for (const [w, items] of t) {
      if (items.length > 1) {
        (multi ??= []).push([w, items]);
        continue;
      }
      const o = items[0] * dims, k = scales[items[0]] * w;
      for (let i = 0; i < dims; i++) row[i] += values[o + i] * k;
    }
    plan.docs.push(doc);
    plan.multi.push(multi);
    doc.row = n;
  }

  /** Takes the document's row out of the plan, moving the last row into its place. */
  private unplace(doc: Indexed): void {
    const plan = this.plan;
    if (!plan || doc.row < 0) return;
    const dims = this.model.dims, last = plan.docs.length - 1;
    if (doc.row !== last) {
      plan.rows.copyWithin(doc.row * dims, last * dims, (last + 1) * dims);
      const moved = plan.docs[last];
      plan.docs[doc.row] = moved;
      plan.multi[doc.row] = plan.multi[last];
      moved.row = doc.row;
    }
    plan.docs.pop();
    plan.multi.pop();
    doc.row = -1;
  }
}
