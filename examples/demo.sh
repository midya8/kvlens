#!/usr/bin/env bash
# Fully offline KVLens demo: bitwise-equivalent KV-cached decoding with
# measured FLOP savings. Both gates assert their side of the contract.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== generation: identical tokens, 85%+ FLOP savings at 32 steps =="
python3 -m kvlens.cli generate --steps 32 --require-identical --min-savings 0.85
echo

echo "== equivalence sweep: bitwise logit match across configs and lengths =="
python3 -m kvlens.cli equivalence
echo

echo "== savings curve over generation length =="
python3 -m kvlens.cli curve --min-savings 0.3
echo

echo "Demo complete: cached path is bitwise-identical to full recompute and measured cheaper."
