"""Complete last-query attention and four-path counted-operation audit."""
from __future__ import annotations
import argparse
import copy
import json
import math
import sys
from .model import (Model, Cache, _embed_row, _project_row, softmax, forward_all,
                    forward_last, prefill, cache_token, generate_full, generate_cached,
                    generate_last)

FIELDS = {'vocab', 'dim', 'heads', 'seed', 'prompt', 'steps'}


def validate(spec):
    if not isinstance(spec, dict) or set(spec) != FIELDS:
        raise ValueError('expected exactly vocab,dim,heads,seed,prompt,steps')
    for key, lo, hi in [('vocab',2,32),('dim',2,16),('heads',1,8),('seed',0,1000000),('steps',0,12)]:
        v=spec[key]
        if type(v) is not int or not lo <= v <= hi:
            raise ValueError(f'{key} must be an integer in {lo}..{hi}')
    if spec['dim'] % spec['heads']:
        raise ValueError('heads must divide dim')
    prompt=spec['prompt']
    if not isinstance(prompt,list) or not 1 <= len(prompt) <= 20 or any(type(t) is not int or not 0 <= t < spec['vocab'] for t in prompt):
        raise ValueError('prompt needs 1..20 in-vocabulary integers')
    return spec


def weights(model):
    return {key:copy.deepcopy(getattr(model,key)) for key in ('vocab','dim','heads','embed','wq','wk','wv','wo','wout','bout')}


def ledger(model,n,kind):
    d=model.dim
    qrows=n if kind=='full' else 1
    kvrows=1 if kind=='append' else n
    visits=n*(n+1)//2 if kind=='full' else n
    r={'q_projection':2*qrows*d*d,'k_projection':2*kvrows*d*d,'v_projection':2*kvrows*d*d,
       'attention_scores':2*visits*d,'weighted_values':2*visits*d,
       'output_projection':2*d*d,'vocabulary_projection':2*d*model.vocab}
    r['total']=sum(r.values())
    return r


def attention(model,tokens):
    x=[_embed_row(model,t,i) for i,t in enumerate(tokens)]
    heads=[];joined=[]
    for h in range(model.heads):
        q=_project_row(model,x[-1],model.wq[h])
        k=[_project_row(model,row,model.wk[h]) for row in x]
        v=[_project_row(model,row,model.wv[h]) for row in x]
        score_terms=[[q[c]*row[c]/math.sqrt(model.head_dim) for c in range(model.head_dim)] for row in k]
        # Match original: sum products, then scale. Term sums may round differently.
        scores=[sum(q[c]*row[c] for c in range(model.head_dim))/math.sqrt(model.head_dim) for row in k]
        # Original uses multiplication by reciprocal, preserve that order exactly.
        scores=[sum(q[c]*row[c] for c in range(model.head_dim))*(1/math.sqrt(model.head_dim)) for row in k]
        probs=softmax(scores)
        contributions=[[probs[j]*v[j][c] for c in range(model.head_dim)] for j in range(len(tokens))]
        out=[sum(row[c] for row in contributions) for c in range(model.head_dim)]
        joined.extend(out)
        heads.append({'query':q,'keys':k,'values':v,'score_terms':score_terms,'scores':scores,
                      'probabilities':probs,'value_contributions':contributions,'output':out})
    output_terms=[[joined[d]*model.wo[d][c] for d in range(model.dim)] for c in range(model.dim)]
    output=[sum(row) for row in output_terms]
    logit_terms=[[output[d]*model.wout[d][c] for d in range(model.dim)] for c in range(model.vocab)]
    logits=[sum(row)+model.bout[c] for c,row in enumerate(logit_terms)]
    return {'embedded':x,'heads':heads,'joined':joined,'output_terms':output_terms,'output':output,
            'logit_terms':logit_terms,'logits':logits}


def cache_state(cache):
    return {'length':cache.length,'keys':copy.deepcopy(cache.k),'values':copy.deepcopy(cache.v)}


def audit(spec,model=None):
    validate(spec)
    model=model or Model(spec['vocab'],spec['dim'],spec['heads'],spec['seed'])
    tokens=list(spec['prompt']);historical=Cache(model);optimized=Cache(model)
    counters={kind:{'flops':0} for kind in ('full','last','historical_cache','optimized_cache')}
    paths=[];totals={key:[] for key in counters}
    for step in range(spec['steps']):
        before=cache_state(optimized)
        full=forward_all(model,tokens,counters['full'])
        last=forward_last(model,tokens,counters['last'])
        old=(prefill(model,historical,tokens,counters['historical_cache']) if step==0 else cache_token(model,historical,tokens[-1],counters['historical_cache']))
        new=(forward_last(model,tokens,counters['optimized_cache'],optimized) if step==0 else cache_token(model,optimized,tokens[-1],counters['optimized_cache']))
        detail=attention(model,tokens)
        if not (full==last==old==new==detail['logits']):
            raise AssertionError('native paths differ')
        ledgers={key:ledger(model,len(tokens),'append' if step and 'cache' in key else ('full' if key in ('full','historical_cache') else 'last')) for key in counters}
        for key in counters:
            totals[key].append(ledgers[key]['total'])
            if sum(totals[key]) != counters[key]['flops']:
                raise AssertionError('counter/ledger mismatch')
        nxt=max(range(model.vocab),key=lambda c:full[c])
        sorted_logits=sorted(full,reverse=True)
        n=len(tokens);paths.append({'step':step,'prefix':list(tokens),'next_token':nxt,
             'top_margin':sorted_logits[0]-sorted_logits[1], 'attention':detail,
             'ledgers':ledgers,'cumulative_flops':{k:v['flops'] for k,v in counters.items()},
             'cache_before':before,'cache_after':cache_state(optimized),
             'historical_cache':cache_state(historical),
             'kv_elements':2*n*model.dim,'ideal_binary64_kv_bytes':16*n*model.dim})
        tokens.append(nxt)
    counts={k:v['flops'] for k,v in counters.items()}
    generated={}
    for name,fn in [('full',generate_full),('last',generate_last),('historical_cache',generate_cached)]:
        result,cost=fn(model,spec['prompt'],spec['steps']);assert result==tokens and cost==counts[name];generated[name]=result
    result,cost=generate_last(model,spec['prompt'],spec['steps'],cached=True)
    assert result==tokens and cost==counts['optimized_cache'];generated['optimized_cache']=result
    return {'version':1,'spec':copy.deepcopy(spec),'tokens':tokens,'paths':paths,'totals':counts,
            'all_paths_bitwise_equal':True,'generated':generated,
            'saved_vs_full':(1-counts['historical_cache']/counts['full']) if counts['full'] else None,
            'optimized_saved_vs_last':(1-counts['optimized_cache']/counts['last']) if counts['last'] else None,
            'peak_kv_elements':paths[-1]['kv_elements'] if paths else 0,
            'ideal_binary64_kv_bytes':paths[-1]['ideal_binary64_kv_bytes'] if paths else 0,
            'final_emitted_token_cached':False if paths else None}


def parse(raw):
    if len(raw.encode('utf8'))>100000:raise ValueError('input exceeds 100000 UTF-8 bytes')
    depth=0;quoted=False;escape=False
    for ch in raw:
        if quoted:
            if escape:escape=False
            elif ch=='\\':escape=True
            elif ch=='"':quoted=False
        elif ch=='"':quoted=True
        elif ch in '[{':
            depth+=1
            if depth>8:raise ValueError('nesting exceeds 8')
        elif ch in ']}':depth-=1
    def pairs(items):
        out={}
        for k,v in items:
            if k in out:raise ValueError('duplicate JSON key')
            out[k]=v
        return out
    def finite(v):raise ValueError('nonfinite JSON number')
    return validate(json.loads(raw,object_pairs_hook=pairs,parse_constant=finite))


def main(argv=None):
    ap=argparse.ArgumentParser();ap.add_argument('--input',required=True);args=ap.parse_args(argv)
    try:
        if args.input=='-':raw=sys.stdin.read(100001)
        else:
            with open(args.input,encoding='utf8') as f:raw=f.read(100001)
        result=audit(parse(raw))
    except (ValueError,OSError,TypeError,OverflowError) as exc:
        print(f'kvlens-audit: {exc}',file=sys.stderr);return 2
    print(json.dumps(result,allow_nan=False,separators=(',',':')));return 0


if __name__=='__main__':raise SystemExit(main())
