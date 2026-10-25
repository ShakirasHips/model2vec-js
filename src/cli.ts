#!/usr/bin/env node
import { readFile, writeFile } from 'node:fs/promises';
import { MiniSemanticSearch, packModel, StaticModel } from './node.js';

const HELP = `mini-semantic-search: semantic search with Model2Vec static embeddings

  mini-semantic-search search <docs.json> "<query>" --fields title,text [--id id] [--k 5] [--model m.bin]
      The documents closest in meaning to the query. docs.json is an array of objects.
  mini-semantic-search embed "<text>" [--model m.bin]
      The text's vector as JSON.
  mini-semantic-search pack <model folder> <out.bin> [--bits 4|8|32] [--group 64] [--dims N]
      Packs a Model2Vec model from Hugging Face (model.safetensors + tokenizer.json).

--model defaults to the bundled potion-base-8M (128 dimensions at 4 bits).`;

const argv = process.argv.slice(2);
const flags = new Map<string, string>();
const args: string[] = [];
for (let i = 0; i < argv.length; i++) {
  const a = argv[i];
  if (a.startsWith('--')) flags.set(a.slice(2), argv[++i]);
  else args.push(a);
}
const flag = (name: string): string | undefined => flags.get(name);
const num = (name: string, def: number): number => (flags.has(name) ? Number(flag(name)) : def);
const model = (): Promise<StaticModel> => StaticModel.load(flag('model'));

async function main(): Promise<void> {
  const [cmd, ...rest] = args;
  if (cmd === 'search' && rest.length === 2 && flag('fields')) {
    const fields = flag('fields')!.split(',');
    const idField = flag('id') ?? 'id';
    const docs = JSON.parse(await readFile(rest[0], 'utf8')) as Record<string, unknown>[];
    const search = await MiniSemanticSearch.create({ fields, storeFields: fields, idField, model: flag('model') });
    search.addAll(docs);
    for (const r of search.search(rest[1]).slice(0, num('k', 5))) {
      console.log(`${r.score.toFixed(3)}  ${String(r.id)}  ${String(r[fields[0]] ?? '')}`);
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
  } else {
    console.log(HELP);
    if (cmd && cmd !== 'help' && cmd !== '--help') process.exitCode = 1;
  }
}

main().catch((err: Error) => {
  console.error(`mini-semantic-search: ${err.message}`);
  process.exit(1);
});
