#!/bin/sh
# Installs the package as npm would publish it into an empty project, then uses it from there: the library, its
# bundled model and word list, and the CLI. Catches files missing from "files" and anything the tests only reach
# through the source tree. Run after `npm run build`.
set -eu
repo=$(cd "$(dirname "$0")/.." && pwd)
dir=$(mktemp -d)
trap 'rm -rf "$dir"' EXIT

tgz=$(cd "$repo" && npm pack --silent --pack-destination "$dir")
cd "$dir"
npm init -y >/dev/null
npm install --silent --no-audit --no-fund "./$tgz"

node --input-type=module -e "
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { FaqSearch } from 'model2vec-js';
const faq = await FaqSearch.create(JSON.parse(readFileSync('$repo/examples/store-faq.json', 'utf8')));
const r = faq.ask('the mug came smahsed');
assert.equal(r.corrected, 'the mug came smashed');
assert.equal(r.answer?.entry.id, 'damaged');
console.log('library: ok');
"

out=$(npx --no-install model2vec-js ask "$repo/examples/store-faq.json" 'do you take paypal')
echo "$out" | grep -q 'What payment methods' || { echo "$out"; echo 'cli: unexpected answer' >&2; exit 1; }
echo 'cli: ok'
