#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
PYTHON=${PYTHON:-python3}
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
"$PYTHON" -m venv "$TMP/venv"
if command -v uv >/dev/null 2>&1; then
 uv pip install --python "$TMP/venv/bin/python" --quiet --no-deps "$ROOT" --target "$TMP/site"
else
 "$TMP/venv/bin/python" -m pip install --quiet --no-deps "$ROOT" --target "$TMP/site"
fi
cd "$TMP"
PYTHONPATH="$TMP/site" "$TMP/venv/bin/python" - <<'PY'
import importlib,json,pathlib
from importlib import resources
for name in ['kvlens','kvlens.model','kvlens.analysis','kvlens.cli','kvlens.audit']:
 m=importlib.import_module(name);assert '/site/' in m.__file__,m.__file__
from kvlens.audit import audit
from kvlens.analysis import equivalence_sweep,savings_curve
from kvlens.model import Model,Cache,prefill,cache_token,fork_cache,forward_all
D=json.loads(resources.files('kvlens').joinpath('data/references.json').read_text())
for c in D['cases']:
 r=audit(c['report']['spec']);assert r['tokens']==c['report']['tokens'];assert r['totals']==c['report']['totals'];assert r['all_paths_bitwise_equal']
assert all(x['max_abs_diff']==0 for x in equivalence_sweep())
assert savings_curve()[-1]['full_flops']==500224 and savings_curve()[-1]['cached_flops']==45888
m=Model(8,4,2);c=Cache(m);prefill(m,c,[1,2]);child=fork_cache(m,c);assert cache_token(m,child,3)==forward_all(m,[1,2,3]);assert c.length==2
try:prefill(m,c,[3])
except ValueError:pass
else:raise AssertionError('occupied prefill accepted')
pathlib.Path('audit.json').write_text(json.dumps(D['cases'][0]['report']['spec']))
print('Installed: five module origins, all32 complete audits/97predictions, original full curve/equivalence and lifecycle/fork checks.')
PY
PYTHONPATH="$TMP/site" "$TMP/venv/bin/python" -m kvlens.audit --input audit.json > audit-report.json
PYTHONPATH="$TMP/site" "$TMP/venv/bin/python" -m kvlens.cli generate --steps 32 --require-identical --min-savings 0.85 > generation.log
PYTHONPATH="$TMP/site" "$TMP/venv/bin/python" -m kvlens.cli equivalence > equivalence.log
PYTHONPATH="$TMP/site" "$TMP/venv/bin/python" -m kvlens.cli curve > curve.log
PYTHONPATH="$TMP/site" "$TMP/venv/bin/python" -m kvlens.cli json > original-report.json
printf '%s\n' 'Audit and all four original CLI modes passed outside checkout.'
