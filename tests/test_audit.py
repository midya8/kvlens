import copy
import hashlib
import json
import math
from pathlib import Path
import random
import subprocess
import sys
import unittest
from kvlens import model as M
from kvlens.audit import audit,attention,ledger,parse,validate

ROOT=Path(__file__).resolve().parents[1]
SPEC=dict(vocab=16,dim=8,heads=2,seed=1,prompt=[0,1,2,3],steps=8)

def assert_close(test,a,b):
 if isinstance(a,float):
  test.assertTrue(math.isfinite(b) and abs(a-b)<=1e-11*max(1,abs(a),abs(b)))
 elif isinstance(a,dict):
  test.assertEqual(set(a),set(b))
  for k in a:assert_close(test,a[k],b[k])
 elif isinstance(a,list):
  test.assertEqual(len(a),len(b))
  for x,y in zip(a,b):assert_close(test,x,y)
 else:test.assertEqual(a,b)

class AuditTests(unittest.TestCase):
 def test_all_four_native_paths(self):
  refs=json.loads((ROOT/'kvlens/data/references.json').read_text())
  for row in refs['cases']:
   r=audit(row['report']['spec'])
   self.assertEqual(r['tokens'],row['report']['tokens']);self.assertEqual(r['totals'],row['report']['totals'])
 def test_random_full_vs_stronger_vs_cached(self):
  rng=random.Random(404)
  for i in range(36):
   d,h=[(4,1),(6,2),(8,4),(12,3)][i%4];v=2+i%13
   m=M.Model(v,d,h,i);ts=[rng.randrange(v) for _ in range(1+i%9)]
   base=M.generate_full(m,ts,1+i%6)[0]
   for cached in [False,True]:self.assertEqual(base,M.generate_last(m,ts,1+i%6,cached)[0])
 def test_ledger_independent_instrumentation(self):
  for d,h in [(4,1),(8,2),(12,3)]:
   m=M.Model(9,d,h,3)
   for n in range(1,11):
    for kind,fn in [('full',M.forward_all),('last',M.forward_last)]:
     c={'flops':0};fn(m,[1]*n,c);self.assertEqual(c['flops'],ledger(m,n,kind)['total'])
    c={'flops':0};cache=M.Cache(m)
    for _ in range(n-1):M.cache_token(m,cache,1)
    M.cache_token(m,cache,1,c);self.assertEqual(c['flops'],ledger(m,n,'append')['total'])
 def test_ledger_one_hand_case(self):
  m=M.Model(16,8,2);self.assertEqual(ledger(m,4,'full'),dict(q_projection=512,k_projection=512,v_projection=512,attention_scores=160,weighted_values=160,output_projection=128,vocabulary_projection=256,total=2240))
  self.assertEqual(ledger(m,4,'last')['total'],1664)
 def test_original_curve_unchanged(self):
  from kvlens.analysis import savings_curve
  self.assertEqual(savings_curve(),[dict(new_tokens=n,full_flops=f,cached_flops=c,saved=s) for n,f,c,s in [(2,5024,3168,.3694),(4,12352,5120,.5855),(8,34944,9408,.7308),(16,119040,19520,.836),(32,500224,45888,.9083)]])
 def test_zero_generation_no_work(self):
  m=M.Model(9,4,2)
  for fn in [M.generate_full,M.generate_cached,M.generate_last]:self.assertEqual(fn(m,[0,1],0),([0,1],0))
  r=audit(dict(SPEC,steps=0));self.assertEqual(r['paths'],[]);self.assertEqual(r['peak_kv_elements'],0);self.assertIsNone(r['optimized_saved_vs_last'])
 def test_one_generation_no_reuse(self):
  r=audit(dict(SPEC,steps=1));self.assertEqual(r['totals']['full'],r['totals']['historical_cache']);self.assertEqual(r['totals']['last'],r['totals']['optimized_cache']);self.assertEqual(r['optimized_saved_vs_last'],0)
 def test_stronger_baseline_cost_order(self):
  r=audit(SPEC);self.assertLess(r['totals']['last'],r['totals']['full']);self.assertLess(r['totals']['optimized_cache'],r['totals']['last']);self.assertLess(r['optimized_saved_vs_last'],r['saved_vs_full'])
 def test_cache_final_token_not_processed(self):
  r=audit(SPEC);self.assertEqual(r['paths'][-1]['cache_after']['length'],len(r['tokens'])-1);self.assertFalse(r['final_emitted_token_cached']);self.assertEqual(r['peak_kv_elements'],2*(4+8-1)*8)
 def test_cache_states_reconstruct_attention(self):
  for row in audit(SPEC)['paths']:
   for h,head in enumerate(row['attention']['heads']):
    self.assertEqual(head['keys'],row['cache_after']['keys'][h]);self.assertEqual(head['values'],row['cache_after']['values'][h])
 def test_uniform_attention_hand_reference(self):
  m=M.Model(3,2,1);m.wq=[[[0,0],[0,0]]];m.wk=[[[0,0],[0,0]]];m.wv=[[[1,0],[0,1]]];m.wo=[[1,0],[0,1]];m.wout=[[1,0,1],[0,1,-1]]
  r=attention(m,[0,1,2]);expected=[sum(row[c] for row in r['embedded'])/3 for c in range(2)]
  self.assertEqual(r['heads'][0]['probabilities'],[1/3]*3)
  for x,y in zip(r['logits'],[expected[0],expected[1],expected[0]-expected[1]]):self.assertAlmostEqual(x,y)
 def test_all_projection_contributions_sum(self):
  for path in audit(SPEC)['paths']:
   a=path['attention']
   self.assertEqual([sum(row) for row in a['output_terms']],a['output'])
   self.assertEqual([sum(row) for row in a['logit_terms']],a['logits'])
   for h in a['heads']:
    self.assertAlmostEqual(sum(h['probabilities']),1)
    for score,terms in zip(h['scores'],h['score_terms']):self.assertAlmostEqual(score,sum(terms),14)
 def test_prefill_occupied_rejected_atomically(self):
  m=M.Model(7,4,2);c=M.Cache(m);M.prefill(m,c,[1,2]);before=copy.deepcopy((c.k,c.v,c.length));ctr={'flops':0}
  with self.assertRaisesRegex(ValueError,'empty cache'):M.prefill(m,c,[3],ctr)
  self.assertEqual((c.k,c.v,c.length),before);self.assertEqual(ctr['flops'],0)
 def test_wrong_model_rejected(self):
  a=M.Model(8,4,2,1);b=M.Model(8,4,2,2);c=M.Cache(a)
  for fn in [lambda:M.prefill(b,c,[1]),lambda:M.cache_token(b,c,1),lambda:M.forward_last(b,[1],cache=c),lambda:M.fork_cache(b,c)]:
   with self.assertRaisesRegex(ValueError,'different model'):fn()
  self.assertEqual(c.length,0)
 def test_inconsistent_shape_rejected(self):
  m=M.Model(8,4,2);c=M.Cache(m);c.length=1
  with self.assertRaisesRegex(ValueError,'inconsistent'):M.cache_token(m,c,1)
 def test_cache_fork_is_not_alias(self):
  m=M.Model(8,4,2);root=M.Cache(m);M.prefill(m,root,[1,2]);left=M.fork_cache(m,root);right=M.fork_cache(m,root)
  a=M.cache_token(m,left,3);b=M.cache_token(m,right,4)
  self.assertEqual(a,M.forward_all(m,[1,2,3]));self.assertEqual(b,M.forward_all(m,[1,2,4]));self.assertEqual(root.length,2)
  left.k[0][0][0]+=1;self.assertNotEqual(left.k[0][0],root.k[0][0]);self.assertEqual(right.k[0][:2],root.k[0])
 def test_bad_tokens_do_not_mutate(self):
  m=M.Model(8,4,2);c=M.Cache(m)
  for bad in [-1,8,True,1.0]:
   with self.assertRaises(ValueError):M.cache_token(m,c,bad)
   self.assertEqual(c.length,0)
  for bad in [[],[1,9],[True],(1,)]:
   with self.assertRaises(ValueError):M.prefill(m,c,bad)
   self.assertEqual(c.length,0)
 def test_bad_generation_steps(self):
  m=M.Model(8,4,2)
  for fn in [M.generate_full,M.generate_cached,M.generate_last]:
   for bad in [-1,True,1.5]:
    with self.assertRaises(ValueError):fn(m,[0],bad)
 def test_bad_model_dimensions(self):
  for args in [(8,0,1),(8,-2,1),(8,4,True),(8,4,2.0),(True,4,1)]:
   with self.assertRaises(ValueError):M.Model(*args)
 def test_schema_bounds(self):
  for key,bad in [('dim',0),('dim',17),('heads',3),('seed',-1),('steps',13),('prompt',[]),('vocab',True),('prompt',[1.0]),('prompt',[16])]:
   with self.assertRaises(ValueError):validate(dict(SPEC,**{key:bad}))
  with self.assertRaises(ValueError):validate(dict(SPEC,extra=1))
 def test_strict_json(self):
  good=json.dumps(SPEC);self.assertEqual(parse(good),SPEC)
  for raw in ['{"steps":1,"steps":2}',good.replace('8,','1e999,',1),'NaN','['*9+'0'+']'*9,' '*100001,good+'x']:
   with self.assertRaises(ValueError):parse(raw)
 def test_cli_all_outputs(self):
  p=subprocess.run([sys.executable,'-m','kvlens.audit','--input','-'],input=json.dumps(SPEC),text=True,capture_output=True);self.assertEqual(p.returncode,0,p.stderr);self.assertEqual(json.loads(p.stdout),audit(SPEC))
 def test_cli_rejects_no_partial(self):
  for raw in ['NaN','{}',json.dumps(dict(SPEC,steps=True))]:
   p=subprocess.run([sys.executable,'-m','kvlens.audit','--input','-'],input=raw,text=True,capture_output=True);self.assertEqual(p.returncode,2);self.assertEqual(p.stdout,'');self.assertIn('kvlens-audit:',p.stderr)
 def test_original_test_and_gate_files_untouched(self):
  original=json.loads((ROOT/'kvlens/data/original-hashes.json').read_text())
  for path,sha in original['files'].items():
   if path.startswith('tests/') or path in ['kvlens/analysis.py','examples/demo.sh']:
    self.assertEqual(hashlib.sha256((ROOT/path).read_bytes()).hexdigest(),sha,path)

class MoreLifecycleTests(unittest.TestCase):
 def test_original_cli_zero_budget_is_actionable(self):
  p=subprocess.run([sys.executable,'-m','kvlens.cli','generate','--steps','0'],capture_output=True,text=True)
  self.assertEqual(p.returncode,2);self.assertEqual(p.stdout,'');self.assertIn('kvlens.audit',p.stderr);self.assertNotIn('Traceback',p.stderr)
 def test_optimized_prefill_guard_and_position_continuation(self):
  m=M.Model(9,8,2);c=M.Cache(m);M.forward_last(m,[1,2,3],cache=c)
  with self.assertRaises(ValueError):M.forward_last(m,[4],cache=c)
  self.assertEqual(M.cache_token(m,c,4),M.forward_all(m,[1,2,3,4]))
 def test_cache_corruption_checked_before_append(self):
  for bad in [None,[[[True]]],[[['x']]],'bad']:
   m=M.Model(8,4,2);c=M.Cache(m);c.k=bad
   with self.assertRaises(ValueError):M.cache_token(m,c,1)

if __name__=='__main__':unittest.main()
