import json
from importlib import resources
from kvlens.audit import audit
D=json.loads(resources.files('kvlens').joinpath('data/references.json').read_text())
print(json.dumps([audit(c['report']['spec']) for c in D['cases']],allow_nan=False))
