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
  /** A loaded model, or where to load one from in `create` (default: the bundled potion-base-8M). */
  model?: StaticModel | Source;
}

export type SearchResult = {
  id: DocumentId;
  /** About −0.1 to 1: how close in meaning the document is to the query. */
  score: number;
  [storedField: string]: unknown;
};

/** A document's vector at 8 bits, value i being q[i] × s: a quarter of the memory, and scores move by about 1e-4. */
interface Vec {
  q: Int8Array;
  s: number;
}

const pack8 = (v: Float32Array): Vec => {
  let max = 0;
  for (let i = 0; i < v.length; i++) max = Math.max(max, Math.abs(v[i]));
  const s = max / 127 || 1, q = new Int8Array(v.length);
  for (let i = 0; i < v.length; i++) q[i] = Math.round(v[i] / s);
  return { q, s };
};

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

const dot8 = (q: Float32Array, { q: v, s }: Vec): number => {
  let a = 0, b = 0, c = 0, d = 0, i = 0;
  for (; i + 3 < q.length; i += 4) {
    a += q[i] * v[i];
    b += q[i + 1] * v[i + 1];
    c += q[i + 2] * v[i + 2];
    d += q[i + 3] * v[i + 3];
  }
  for (; i < q.length; i++) a += q[i] * v[i];
  return (a + b + c + d) * s;
};

interface Indexed {
  id: DocumentId;
  /** Per field, a vector for each of its values (empty if the document has none). */
  fields: Vec[][];
  /** All the fields' text together. */
  whole: Vec | null;
  stored: Record<string, unknown>;
  /** Its row in the current plan, or -1 if it has none. */
  row: number;
}

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
function terms(doc: Indexed, { use, weights, subset }: Weighting): [number, Vec[]][] | null {
  let weight = 0;
  for (let i = 0; i < doc.fields.length; i++) if (use[i] && doc.fields[i].length) weight += weights[i];
  if (!weight) return null;
  // The whole-document vector covers every field, so it only counts when every field does.
  const whole = subset || !doc.whole ? 0 : 0.5;
  const out: [number, Vec[]][] = whole ? [[whole, [doc.whole!]]] : [];
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
  multi: ([number, Vec[]][] | null)[];
}

const texts = (value: unknown): string[] => {
  if (value === undefined || value === null) return [];
  return (Array.isArray(value) ? value : [value]).map(String).filter((s) => s.trim());
};

/**
 * Semantic search with MiniSearch's API: documents are found by meaning rather than shared words, so "stop sending me
 * newsletters" finds "How do I unsubscribe from your emails?". Each document is embedded once per field and once
 * whole. It scores half its whole vector's cosine similarity to the query, plus half its fields' (weighted by
 * `boost`; an array field counts its closest item). Searching takes no network and no server.
 */
export class MiniSemanticSearch<T = AnyDocument> {
  readonly model: StaticModel;
  private readonly options: Required<Omit<Options<T>, 'model'>>;
  private readonly index = new Map<DocumentId, Indexed>();
  private plan: Plan | null = null;
  /** The weighting of the last search that had no plan: a plan is made when one is used twice running. */
  private lastKey = '';

  /** Loads the model (the bundled one unless `options.model` says otherwise), then makes the index. */
  static async create<T = AnyDocument>(options: Options<T>): Promise<MiniSemanticSearch<T>> {
    const { model = DEFAULT_MODEL } = options;
    return new MiniSemanticSearch({ ...options, model: model instanceof StaticModel ? model : await StaticModel.load(model) });
  }

  /** For a model you have loaded already (`StaticModel.load()`); `create` loads it for you. */
  constructor(options: Options<T> & { model: StaticModel }) {
    const { model, fields, storeFields = [], idField = 'id', extractField = (d, f) => (d as Record<string, unknown>)[f], searchOptions = {} } = options;
    if (!(model instanceof StaticModel)) throw new Error('MiniSemanticSearch: the constructor needs a loaded model; use MiniSemanticSearch.create to load one');
    if (!fields?.length) throw new Error('MiniSemanticSearch: option "fields" must list at least one field');
    this.model = model;
    this.options = { fields, storeFields, idField, extractField, searchOptions };
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
    const values = fields.map((f) => texts(extractField(document, f)));
    const all = values.flat().join(' ');
    const doc: Indexed = {
      id,
      fields: values.map((v) => v.map((s) => pack8(this.model.embed(s)))),
      whole: all ? pack8(this.model.embed(all)) : null,
      stored: Object.fromEntries(storeFields.map((f) => [f, extractField(document, f)])),
      row: -1,
    };
    this.index.set(id, doc);
    if (this.plan) this.place(this.plan, doc);
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
  }

  discardAll(ids: readonly DocumentId[]): void {
    for (const id of ids) this.discard(id);
  }

  /** Removes the given documents, or every document if none are given. */
  removeAll(documents?: readonly T[]): void {
    if (documents === undefined) {
      this.index.clear();
      this.plan = null;
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
    const { fields: only, boost = {}, filter, minScore = -Infinity } = { ...this.options.searchOptions, ...searchOptions };
    const { fields } = this.options;
    const use = fields.map((f) => !only || only.includes(f));
    const weights = fields.map((f, i) => (use[i] ? boost[f] ?? 1 : 0));
    const weighting: Weighting = { key: JSON.stringify([use, weights]), use, weights, subset: use.some((u) => !u) };
    const q = this.model.embed(query);
    const results: SearchResult[] = [];
    const emit = (doc: Indexed, score: number): void => {
      if (score < minScore) return;
      const result: SearchResult = { id: doc.id, score, ...doc.stored }; // stored fields come last, as in MiniSearch
      if (!filter || filter(result)) results.push(result);
    };
    const best = (items: Vec[]): number => {
      let b = -Infinity;
      for (const v of items) b = Math.max(b, dot8(q, v));
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
    let multi: [number, Vec[]][] | null = null;
    for (const [w, items] of t) {
      if (items.length > 1) {
        (multi ??= []).push([w, items]);
        continue;
      }
      const { q, s } = items[0], k = s * w;
      for (let i = 0; i < dims; i++) row[i] += q[i] * k;
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
