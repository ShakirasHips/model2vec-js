import type { StaticModel } from './model.js';
import { normalize } from './tokenizer.js';

const DIGITS = '0123456789abcdefghijklmnopqrstuvwxyz';
const FRONT_CODED = '#front-coded';

/**
 * Reads a word list: one word per line, or front-coded (`writeWordList`), where each line is the number of letters
 * shared with the word before (one base-36 digit) and the rest.
 */
export function parseWordList(text: string): string[] {
  const lines = text.split('\n').filter((l) => l);
  if (lines[0] !== FRONT_CODED) return lines.map((l) => l.trim());
  const words: string[] = [];
  let prev = '';
  for (const line of lines.slice(1)) {
    prev = prev.slice(0, DIGITS.indexOf(line[0])) + line.slice(1);
    words.push(prev);
  }
  return words;
}

/** Writes a word list front-coded (sorted, without duplicates), about half the size of one word per line. */
export function writeWordList(words: Iterable<string>): string {
  const out = [FRONT_CODED];
  let prev = '';
  for (const w of [...new Set(words)].sort()) {
    let p = 0;
    while (p < prev.length && p < w.length && p < 35 && prev[p] === w[p]) p++;
    out.push(DIGITS[p] + w.slice(p));
    prev = w;
  }
  return out.join('\n') + '\n';
}

export interface SpellerOptions {
  /** Text whose words are known and preferred as corrections, such as the FAQ's questions and answers. */
  preferred?: string[];
  /** More real words to leave alone, but never to correct to (rare words make poor guesses). */
  known?: Iterable<string>;
}

/**
 * Corrects misspelt words before they're embedded: a misspelt word breaks into word pieces that mean something
 * else ("reprots" is rep + ##rot + ##s). Words of under 4 letters and known words are left alone; any other becomes
 * the closest preferred word or whole-word token of the model, by letters dropped, added, changed or swapped. One
 * edit is allowed for words of 4–5 letters and two for longer ones, but two only to a preferred word; ties go to
 * the same first letter (rarely the one mistyped), then to the word the preferred text uses most, then to the
 * commoner word (a BERT vocabulary lists words roughly from common to rare: "stuf" becomes "stuff", not "stu").
 */
export class Speller {
  private readonly preferred = new Map<string, number>();
  private readonly known: Set<string>;
  private readonly byLength: string[][] = [];
  private readonly rank = new Map<string, number>();

  constructor(model: StaticModel, { preferred = [], known = [] }: SpellerOptions = {}) {
    for (const text of preferred) {
      for (const w of normalize(text).match(/[a-z]+/g) ?? []) this.preferred.set(w, (this.preferred.get(w) ?? 0) + 1);
    }
    model.tokens.forEach((t, i) => {
      if (/^[a-z]{2,}$/.test(t)) this.rank.set(t, i);
    });
    this.known = new Set([...this.preferred.keys(), ...this.rank.keys()]);
    for (const w of this.known) (this.byLength[w.length] ??= []).push(w);
    for (const w of known) this.known.add(normalize(w));
  }

  /** The word as it should be spelt (lower-cased), or unchanged if it's known or no close word is. */
  correctWord(word: string): string {
    const w = normalize(word);
    if (w.length < 4 || this.known.has(w) || !/^[a-z]+$/.test(w)) return word;
    const max = w.length <= 5 ? 1 : 2;
    let best = word;
    let key = [Infinity, 0, 0, 0];
    for (let len = w.length - max; len <= w.length + max; len++) {
      for (const c of this.byLength[len] ?? []) {
        const d = distance(w, c, Math.min(max, key[0]));
        if (d === Infinity || (d === 2 && !this.preferred.has(c))) continue;
        const k = [d, c[0] === w[0] ? 0 : 1, -(this.preferred.get(c) ?? 0), this.rank.get(c) ?? Infinity];
        const i = k.findIndex((x, j) => x !== key[j]);
        if (i >= 0 && k[i] < key[i]) {
          best = c;
          key = k;
        }
      }
    }
    return best;
  }

  /** The text with misspelt words corrected, e.g. "can reprots exprot to csv" → "can reports export to csv". */
  correct(text: string): string {
    return text.replace(/\p{L}+/gu, (w) => this.correctWord(w));
  }
}

/** Edit distance counting a swap of neighbouring letters as one edit, or Infinity once it's over max. */
export function distance(a: string, b: string, max = Infinity): number {
  if (Math.abs(a.length - b.length) > max) return Infinity;
  let prev2: number[] | null = null;
  let prev = Array.from({ length: b.length + 1 }, (_, j) => j);
  for (let i = 1; i <= a.length; i++) {
    const row = [i];
    let best = i;
    for (let j = 1; j <= b.length; j++) {
      let d = Math.min(prev[j] + 1, row[j - 1] + 1, prev[j - 1] + (a[i - 1] === b[j - 1] ? 0 : 1));
      if (prev2 && j > 1 && a[i - 1] === b[j - 2] && a[i - 2] === b[j - 1]) d = Math.min(d, prev2[j - 2] + 1);
      row.push(d);
      best = Math.min(best, d);
    }
    if (best > max) return Infinity;
    prev2 = prev;
    prev = row;
  }
  return prev[b.length] <= max ? prev[b.length] : Infinity;
}
