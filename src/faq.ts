import { DEFAULT_WORDS, loadText, type Source } from './load.js';
import { similarity, StaticModel } from './model.js';
import { OFF_TOPIC } from './off-topic.js';
import { parseWordList, Speller } from './speller.js';

export interface FaqEntry {
  /** The question the entry answers. */
  q: string;
  /** The answer, in whatever format you display. */
  a: string;
  /** A heading the entry sits under; it helps matching when questions are alike ("What does this screen show?"). */
  section?: string;
  /** Other ways people ask the question; the closest one counts. */
  alt?: string[];
}

export interface Match<E> {
  entry: E;
  /** Cosine similarity, about −0.1 to 1. */
  score: number;
}

export interface Answer<E> {
  /** The question as asked. */
  question: string;
  /** The question with spelling corrected, or null if nothing changed. */
  corrected: string | null;
  /** The best match, or null if even it scored under the threshold. */
  answer: Match<E> | null;
  /** The best matches, best first, whatever their scores. */
  matches: Match<E>[];
}

export interface FaqOptions {
  /** A loaded model, or where to load one from (default: the bundled potion-base-8M). */
  model?: StaticModel | Source;
  /** The spelling corrector's extra known words: a word list's source or the words; false turns correction off. */
  words?: Source | string[] | false;
  /** Weight of the question-and-answer text against the question alone when scoring (default 0.5). */
  mix?: number;
  /**
   * Score under which the best match isn't treated as an answer. 'auto' (the default) sets it from the FAQ: 0.02
   * above the 90th percentile of what 40 unrelated questions ("how do I bake sourdough bread") score against it.
   */
  threshold?: number | 'auto';
}

export interface AskOptions<E> {
  /** How many matches to return (default 3). */
  k?: number;
  /** Only consider entries for which this is true. */
  filter?: (entry: E) => boolean;
}

interface Indexed<E> {
  entry: E;
  questions: Float32Array[];
  whole: Float32Array;
}

const plain = (s: string): string => s.replace(/\*\*|__|`/g, '');

/**
 * Answers questions from an FAQ by finding the entries closest in meaning. Each entry is embedded twice: its
 * question (with its section, and each `alt` phrasing), and its question with the answer. An entry scores
 * (1 − mix) × its closest question + mix × the whole. Nothing is generated, so answers are always your own text.
 */
export class FaqSearch<E extends FaqEntry = FaqEntry> {
  readonly model: StaticModel;
  readonly mix: number;
  private readonly fixedThreshold: number | null;
  private calibrated: number | null = null;
  private readonly known: string[] | null;
  private speller: Speller | null = null;
  private readonly index: Indexed<E>[] = [];

  /** Loads what's needed and indexes the entries. Embedding takes about 0.1 ms an entry. */
  static async create<E extends FaqEntry>(entries: E[], options: FaqOptions = {}): Promise<FaqSearch<E>> {
    const { model = undefined, words = DEFAULT_WORDS } = options;
    const [m, known] = await Promise.all([
      model instanceof StaticModel ? model : StaticModel.load(model),
      words === false ? null : Array.isArray(words) ? words : loadText(words).then(parseWordList),
    ]);
    return new FaqSearch(m, entries, { ...options, words: known ?? false });
  }

  /** For a model you've loaded already; `words` must be the words themselves (or false) here. */
  constructor(model: StaticModel, entries: E[] = [], { mix = 0.5, threshold = 'auto', words = [] }: Omit<FaqOptions, 'model' | 'words'> & { words?: string[] | false } = {}) {
    this.model = model;
    this.mix = mix;
    this.fixedThreshold = threshold === 'auto' ? null : threshold;
    this.known = words === false ? null : words;
    this.add(...entries);
  }

  /** The entries, in the order added. */
  get entries(): E[] {
    return this.index.map((r) => r.entry);
  }

  /** Adds entries; they can be asked about straight away. */
  add(...entries: E[]): void {
    for (const entry of entries) {
      const title = entry.section ? `${entry.section}: ${entry.q}` : entry.q;
      this.index.push({
        entry,
        questions: [title, ...(entry.alt ?? [])].map((q) => this.model.embed(q)),
        whole: this.model.embed(`${title} ${plain(entry.a)}`),
      });
    }
    this.speller = null; // rebuilt with the new entries' words when next needed
    this.calibrated = null;
  }

  /** The score a best match needs to count as an answer. */
  get threshold(): number {
    if (this.fixedThreshold !== null) return this.fixedThreshold;
    if (this.calibrated === null) {
      const scores = OFF_TOPIC.map((q) => this.rank(this.model.embed(q), 1)[0]?.score ?? 0).sort((a, b) => a - b);
      this.calibrated = scores[Math.floor(scores.length * 0.9)] + 0.02;
    }
    return this.calibrated;
  }

  /** Removes the entries for which `test` is true. */
  remove(test: (entry: E) => boolean): void {
    for (let i = this.index.length - 1; i >= 0; i--) if (test(this.index[i].entry)) this.index.splice(i, 1);
    this.speller = null;
    this.calibrated = null;
  }

  /** The question with misspelt words corrected (unchanged if correction is off). */
  correct(question: string): string {
    if (!this.known) return question;
    this.speller ??= new Speller(this.model, {
      preferred: this.index.flatMap(({ entry: e }) => [e.section ?? '', e.q, e.a, ...(e.alt ?? [])]),
      known: this.known,
    });
    return this.speller.correct(question);
  }

  ask(question: string, { k = 3, filter }: AskOptions<E> = {}): Answer<E> {
    const fixed = this.correct(question);
    const matches = this.rank(this.model.embed(fixed), k, filter);
    return {
      question,
      corrected: fixed === question ? null : fixed,
      answer: matches[0] && matches[0].score >= this.threshold ? matches[0] : null,
      matches,
    };
  }

  private rank(v: Float32Array, k: number, filter?: (entry: E) => boolean): Match<E>[] {
    return this.index
      .filter((r) => !filter || filter(r.entry))
      .map((r) => ({
        entry: r.entry,
        score: (1 - this.mix) * Math.max(...r.questions.map((q) => similarity(v, q))) + this.mix * similarity(v, r.whole),
      }))
      .sort((a, b) => b.score - a.score)
      .slice(0, k);
  }
}
