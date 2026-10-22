#!/bin/sh
# Rebuilds models/words.txt, the spelling corrector's list of real words, from SCOWL (http://wordlist.aspell.net/):
# size 60 (the usual spell-checker size) in American, British, Canadian and Australian spelling, with proper names
# up to size 50. Run from the repository root after `npm run build`.
set -e
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
curl -sfL -o "$tmp/scowl.tar.gz" https://downloads.sourceforge.net/project/wordlist/SCOWL/2020.12.07/scowl-2020.12.07.tar.gz
tar -xzf "$tmp/scowl.tar.gz" -C "$tmp"
final="$tmp/scowl-2020.12.07/final"
lists=""
for f in "$final"/*; do
  name=$(basename "$f")
  case "$name" in
    english-*|american-*|british-*|canadian-*|australian-*) ;;
    *) continue ;;
  esac
  case "$name" in
    *-words.*|*-upper.*) max=60 ;;
    *-proper-names.*) max=50 ;;
    *) continue ;;
  esac
  [ "${name##*.}" -le "$max" ] && lists="$lists $f"
done
# shellcheck disable=SC2086
node dist/cli.js words models/words.txt $lists --encoding latin1
