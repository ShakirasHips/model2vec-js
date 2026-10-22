import type { FaqEntry, FaqSearch } from './faq.js';

export interface TestQuestion {
  /** A question as a user might ask it. */
  q: string;
  /** The ids of the entries that answer it (any counts); empty if none does. */
  ids: string[];
}

export interface EvalResult {
  answerable: number;
  offTopic: number;
  /** Share of answerable questions whose first match answers them. */
  top1: number;
  /** Share whose top 3 include an answer. */
  top3: number;
  /** Share the threshold turns away although they have an answer. */
  turnedAway: number;
  /** Share of off-topic questions the threshold turns away. */
  offTopicTurnedAway: number;
  /** Highest best-match score of any off-topic question. */
  offTopicMax: number;
}

/** Scores a search on test questions; `id` names the entry field holding the ids the tests use. */
export function evaluate<E extends FaqEntry>(faq: FaqSearch<E>, tests: TestQuestion[], { id = 'id' as keyof E, typos = 0 } = {}): EvalResult {
  let top1 = 0, top3 = 0, turnedAway = 0, offRejected = 0, offTopicMax = -Infinity;
  const answerable = tests.filter((t) => t.ids.length).length;
  tests.forEach((t, i) => {
    const r = faq.ask(typos ? misspell(t.q, typos, i + 1) : t.q, { k: 3 });
    if (!t.ids.length) {
      offTopicMax = Math.max(offTopicMax, r.matches[0]?.score ?? -Infinity);
      if (!r.answer) offRejected++;
      return;
    }
    const ids = r.matches.map((m) => String(m.entry[id]));
    if (t.ids.includes(ids[0])) top1++;
    if (ids.some((x) => t.ids.includes(x))) top3++;
    if (!r.answer) turnedAway++;
  });
  const offTopic = tests.length - answerable;
  return {
    answerable, offTopic, top1: top1 / answerable, top3: top3 / answerable, turnedAway: turnedAway / answerable,
    offTopicTurnedAway: offTopic ? offRejected / offTopic : NaN, offTopicMax,
  };
}

const rows = ['qwertyuiop', 'asdfghjkl', 'zxcvbnm'];
const near = (c: string): string[] => {
  const r = rows.findIndex((row) => row.includes(c));
  if (r < 0) return [c];
  const i = rows[r].indexOf(c);
  return [rows[r][i - 1], rows[r][i + 1], rows[r - 1]?.[i], rows[r + 1]?.[i]].filter(Boolean) as string[];
};

/**
 * Mistypes `n` words of 4+ letters, the same way for the same seed: a letter (never the first) dropped, doubled,
 * swapped with the one before, or replaced by or added beside a neighbouring key on a QWERTY keyboard.
 */
export function misspell(text: string, n: number, seed = 1): string {
  let s = seed;
  const rand = (k: number): number => {
    s = (s * 1103515245 + 12345) >>> 0;
    return (s >>> 8) % k;
  };
  const words = text.split(' ');
  const long = words.map((w, i) => [w, i] as const).filter(([w]) => /^[a-z]{4,}$/i.test(w));
  for (let t = 0; t < n && long.length; t++) {
    const [w, i] = long.splice(rand(long.length), 1)[0];
    const p = 1 + rand(w.length - 1), keys = near(w[p].toLowerCase()), k = keys[rand(keys.length)];
    words[i] = [
      w.slice(0, p) + w.slice(p + 1),
      w.slice(0, p) + w[p] + w.slice(p),
      w.slice(0, p - 1) + w[p] + w[p - 1] + w.slice(p + 1),
      w.slice(0, p) + k + w.slice(p + 1),
      w.slice(0, p) + k + w.slice(p),
    ][rand(5)];
  }
  return words.join(' ');
}
