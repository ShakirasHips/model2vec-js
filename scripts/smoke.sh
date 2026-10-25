#!/bin/sh
# Installs the package as npm would publish it into an empty project, then uses it from there: the library, its
# bundled model, and the CLI. Catches files missing from "files" and anything the tests only reach
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
import MiniSemanticSearch from 'mini-semantic-search';
const search = await MiniSemanticSearch.create({ fields: ['section', 'q', 'a'], storeFields: ['q'] });
search.addAll(JSON.parse(readFileSync('$repo/examples/store-faq.json', 'utf8')));
assert.equal(search.search('the mug came smashed')[0].id, 'damaged');
console.log('library: ok');
"

out=$(npx --no-install mini-semantic-search search "$repo/examples/store-faq.json" 'do you take paypal' --fields q,a)
echo "$out" | head -1 | grep -q 'What payment methods' || { echo "$out"; echo 'cli: unexpected result' >&2; exit 1; }
echo 'cli: ok'
