"""Reproducible full reference exports, no selected logits or tables."""
from pathlib import Path
import json,random,hashlib,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from kvlens.audit import audit,weights
from kvlens.model import Model
ROOT=Path(__file__).resolve().parents[1]
specs=[]
for v,d,h,seed in [(16,8,2,1),(9,12,3,5),(7,4,1,0),(12,8,4,7)]:
 for prompt,steps in [([0,1,2,3],8),([1],1),([2,0],0),([0,1],4),([0,1,2,3,4,5],3)]:
  specs.append(dict(vocab=v,dim=d,heads=h,seed=seed,prompt=prompt,steps=steps))
rng=random.Random(371)
for i in range(12):
 v,d,h,seed=[(8,4,2,3),(9,6,3,11),(5,8,1,23)][i%3]
 specs.append(dict(vocab=v,dim=d,heads=h,seed=seed,prompt=[rng.randrange(v) for _ in range(1+i%7)],steps=1+i%5))
models={};reports=[]
for s in specs:
 key=f'{s["vocab"]}-{s["dim"]}-{s["heads"]}-{s["seed"]}'
 m=Model(s['vocab'],s['dim'],s['heads'],s['seed']);models[key]=weights(m)
 r=audit(s,m)
 reports.append({'model_id':key,'report':{k:r[k] for k in ('spec','tokens','totals')}})
original=json.loads((ROOT/'kvlens/data/original-hashes.json').read_text())
# Fixed originals are tests, analysis, CLI and demo; model changes are explicitly audited.
for path,sha in original['files'].items():
 if path.startswith('tests/') or path in ('kvlens/analysis.py','examples/demo.sh'):
  assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==sha,path
out={'models':models,'cases':reports,'original_commit':original['commit']}
raw=json.dumps(out,separators=(',',':'),allow_nan=False)+'\n'
(ROOT/'kvlens/data/references.json').write_text(raw)
(ROOT/'docs/data.js').write_text('window.KV_DATA='+raw.strip()+';\n')
print(f'{len(reports)} complete audit specifications, {sum(r["report"]["spec"]["steps"] for r in reports)} full decision paths')
