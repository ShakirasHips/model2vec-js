import { DEFAULT_MODEL, type Source } from './load.js';
import { similarity, StaticModel } from './model.js';

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

interface Indexed {
  /** Per field, a vector for each of its values (empty if the document has none). */
  fields: Float32Array[][];
  /** All the fields' text together. */
  whole: Float32Array | null;
  stored: Record<string, unknown>;
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

  /** Adds a document; it can be searched straight away. Embedding takes about 0.1 ms. */
  add(document: T): void {
    const { fields, storeFields, idField, extractField } = this.options;
    const id = extractField(document, idField);
    if (id === undefined) throw new Error(`MiniSemanticSearch: document does not have ID field "${idField}"`);
    if (this.index.has(id)) throw new Error(`MiniSemanticSearch: duplicate ID ${id}`);
    const values = fields.map((f) => texts(extractField(document, f)));
    const all = values.flat().join(' ');
    this.index.set(id, {
      fields: values.map((v) => v.map((s) => this.model.embed(s))),
      whole: all ? this.model.embed(all) : null,
      stored: Object.fromEntries(storeFields.map((f) => [f, extractField(document, f)])),
    });
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
    if (!this.index.delete(id)) throw new Error(`MiniSemanticSearch: cannot discard document with ID ${id}: it is not in the index`);
  }

  discardAll(ids: readonly DocumentId[]): void {
    for (const id of ids) this.discard(id);
  }

  /** Removes the given documents, or every document if none are given. */
  removeAll(documents?: readonly T[]): void {
    if (documents === undefined) this.index.clear();
    else for (const d of documents) this.remove(d);
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
    const subset = use.some((u) => !u);
    const q = this.model.embed(query);
    const results: SearchResult[] = [];
    for (const [id, doc] of this.index) {
      let sum = 0, weight = 0;
      doc.fields.forEach((vs, i) => {
        if (!use[i] || !vs.length) return;
        const w = boost[fields[i]] ?? 1;
        let best = -Infinity;
        for (const v of vs) best = Math.max(best, similarity(q, v));
        sum += w * best;
        weight += w;
      });
      if (!weight) continue;
      // The whole-document vector covers every field, so it only counts when every field does.
      const score = subset || !doc.whole ? sum / weight : 0.5 * similarity(q, doc.whole) + 0.5 * (sum / weight);
      if (score < minScore) continue;
      const result: SearchResult = { id, score, ...doc.stored }; // stored fields come last, as in MiniSearch
      if (!filter || filter(result)) results.push(result);
    }
    return results.sort((a, b) => b.score - a.score);
  }
}
