"""Verify every regenerated report cell within declared binary64 tolerance."""
import json,math,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
paths=[ROOT/'kvlens/data/references.json',ROOT/'docs/data.js']
before=[p.read_bytes() for p in paths]
def compare(a,b):
 if isinstance(a,float):assert math.isfinite(b) and abs(a-b)<=1e-12*max(1,abs(a),abs(b))
 elif isinstance(a,dict):
  assert set(a)==set(b)
  for k in a:compare(a[k],b[k])
 elif isinstance(a,list):
  assert len(a)==len(b)
  for x,y in zip(a,b):compare(x,y)
 else:assert a==b,(a,b)
try:
 subprocess.run([sys.executable,'scripts/export_audit.py'],cwd=ROOT,check=True)
 compare(json.loads(before[0]),json.loads(paths[0].read_bytes()))
 compare(json.loads(before[1].decode().removeprefix('window.KV_DATA=').strip().removesuffix(';')),json.loads(paths[1].read_text().removeprefix('window.KV_DATA=').strip().removesuffix(';')))
 print('Complete model/report/browser regeneration verified (1e-12 normalized tolerance).')
finally:
 for p,raw in zip(paths,before):p.write_bytes(raw)
