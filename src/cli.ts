#!/usr/bin/env node
import { readFile, writeFile } from 'node:fs/promises';
import {
  evaluate, FaqSearch, normalize, packModel, parseWordList, StaticModel, writeWordList, type FaqEntry,
} from './node.js';

const HELP = `model2vec-js: Model2Vec static embeddings and FAQ search

  model2vec-js ask <faq.json> "<question>" [--k 3] [--model m.bin] [--no-spelling]
      The best entries for a question. faq.json is [{ "q", "a", "section"?, "alt"?, ... }].
  model2vec-js eval <faq.json> <tests.json> [--typos 0,1,2] [--mix 0.5] [--threshold auto] [--model m.bin]
      Accuracy on test questions: [{ "q", "ids": [entry ids that answer it, or none] }]. Entries need an "id".
  model2vec-js embed "<text>" [--model m.bin]
      The text's vector as JSON.
  model2vec-js pack <model folder> <out.bin> [--bits 4|8|32] [--group 64] [--dims N]
      Packs a Model2Vec model from Hugging Face (model.safetensors + tokenizer.json).
  model2vec-js words <out.txt> <word list>... [--model m.bin] [--encoding latin1]
      Writes the spelling corrector's word list: the words of 4+ letters in the lists that the model has no
      token for, front-coded. Lists are one word per line, UTF-8 unless --encoding says otherwise.

--model defaults to the bundled potion-base-8M (4-bit).`;

const argv = process.argv.slice(2);
const flags = new Map<string, string>();
const args: string[] = [];
for (let i = 0; i < argv.length; i++) {
  const a = argv[i];
  if (a === '--no-spelling') flags.set('no-spelling', '');
  else if (a.startsWith('--')) flags.set(a.slice(2), argv[++i]);
  else args.push(a);
}
const flag = (name: string): string | undefined => flags.get(name);
const num = (name: string, def: number): number => (flags.has(name) ? Number(flag(name)) : def);
const json = async <T>(path: string): Promise<T> => JSON.parse(await readFile(path, 'utf8')) as T;
const model = (): Promise<StaticModel> => StaticModel.load(flag('model'));
const pct = (x: number): string => `${(100 * x).toFixed(1)}%`;

async function main(): Promise<void> {
  const [cmd, ...rest] = args;
  if (cmd === 'ask' && rest.length === 2) {
    const faq = await FaqSearch.create(await json<FaqEntry[]>(rest[0]), {
      model: flag('model'), words: flags.has('no-spelling') ? false : undefined,
    });
    const r = faq.ask(rest[1], { k: num('k', 3) });
    if (r.corrected) console.log(`(searched for: ${r.corrected})`);
    if (!r.answer) console.log('No confident answer. Closest:');
    for (const m of r.matches) console.log(`${m.score.toFixed(3)}  ${m.entry.section ? `${m.entry.section}: ` : ''}${m.entry.q}`);
    if (r.answer) console.log(`\n${r.answer.entry.a}`);
  } else if (cmd === 'eval' && rest.length === 2) {
    const [entries, tests] = await Promise.all([json<FaqEntry[]>(rest[0]), json<{ q: string; ids: string[] }[]>(rest[1])]);
    const m = await model();
    const t0 = performance.now();
    const faq = await FaqSearch.create(entries, { model: m, mix: num('mix', 0.5), threshold: flags.has('threshold') ? num('threshold', 0) : 'auto' });
    const plain = new FaqSearch(m, entries, { mix: faq.mix, threshold: faq.threshold, words: false });
    const indexMs = performance.now() - t0;
    const first = evaluate(faq, tests);
    console.log(`${entries.length} entries (indexed in ${indexMs.toFixed(0)} ms), threshold ${faq.threshold.toFixed(3)}, ${first.answerable} answerable and ${first.offTopic} off-topic test questions`);
    console.log('typos  spelling  top-1   top-3   answerable turned away  off-topic turned away  off-topic max');
    for (const typos of (flag('typos') ?? '0').split(',').map(Number)) {
      for (const [name, f] of [['off', plain], ['on', faq]] as const) {
        const r = evaluate(f, tests, { typos });
        console.log(`${String(typos).padStart(5)}  ${name.padEnd(8)}  ${pct(r.top1).padStart(6)}  ${pct(r.top3).padStart(6)}  ` +
          `${pct(r.turnedAway).padStart(22)}  ${(r.offTopic ? pct(r.offTopicTurnedAway) : '-').padStart(21)}  ${r.offTopicMax.toFixed(3).padStart(13)}`);
      }
    }
  } else if (cmd === 'embed' && rest.length === 1) {
    console.log(JSON.stringify(Array.from((await model()).embed(rest[0]), (x) => Number(x.toFixed(6)))));
  } else if (cmd === 'pack' && rest.length === 2) {
    const bits = num('bits', 4);
    if (bits !== 4 && bits !== 8 && bits !== 32) throw new Error('--bits must be 4, 8 or 32');
    const bytes = await packModel(rest[0], { bits, group: num('group', 64), dims: flags.has('dims') ? num('dims', 0) : undefined });
    await writeFile(rest[1], bytes);
    const m = StaticModel.fromBytes(bytes);
    console.log(`${m.tokens.length} tokens × ${m.dims} dims at ${bits} bits -> ${rest[1]} (${(bytes.length / 1e6).toFixed(2)} MB)`);
  } else if (cmd === 'words' && rest.length >= 2) {
    const m = await model();
    const words = new Set<string>();
    for (const path of rest.slice(1)) {
      for (const w of parseWordList(await readFile(path, (flag('encoding') ?? 'utf8') as BufferEncoding))) {
        const n = normalize(w);
        if (/^[a-z]{4,}$/.test(n) && !m.has(n)) words.add(n);
      }
    }
    const text = writeWordList(words);
    await writeFile(rest[0], text);
    console.log(`${words.size} words -> ${rest[0]} (${(text.length / 1e3).toFixed(0)} KB)`);
  } else {
    console.log(HELP);
    if (cmd && cmd !== 'help' && cmd !== '--help') process.exitCode = 1;
  }
}

main().catch((err: Error) => {
  console.error(`model2vec-js: ${err.message}`);
  process.exit(1);
});
