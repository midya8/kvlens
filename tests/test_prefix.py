"""Prefix branches checked against independently rebuilt token histories."""
import copy
import random
import unittest

from kvlens.model import (Model, Cache, prefill, forward_last, forward_all,
                          cache_token, fork_cache)
from kvlens.prefix import prefix_cache


def snapshot(cache):
    return copy.deepcopy((cache.k, cache.v, cache.length))


class PrefixTests(unittest.TestCase):
    def test_every_prefix_rebuild_and_counter_continuation(self):
        rng = random.Random(921)
        cases = 0
        for seed in range(7):
            for dim, heads in ((4,1),(6,2),(8,4),(12,3)):
                model = Model(9,dim,heads,seed)
                tokens = [rng.randrange(9) for _ in range(8)]
                for optimized in (False, True):
                    root = Cache(model)
                    if optimized:
                        forward_last(model,tokens,cache=root)
                    else:
                        prefill(model,root,tokens)
                    before = snapshot(root)
                    for length in range(9):
                        child = prefix_cache(model,root,length)
                        reference = Cache(model)
                        if length: prefill(model,reference,tokens[:length])
                        self.assertEqual(snapshot(child),snapshot(reference))
                        self.assertIs(child.model,model)
                        branch = [rng.randrange(9) for _ in range(3)]
                        context = tokens[:length]
                        for token in branch:
                            a,b = {'flops':0},{'flops':0}
                            actual = cache_token(model,child,token,a)
                            expected = cache_token(model,reference,token,b)
                            context = context+[token]
                            self.assertEqual(actual,expected)
                            self.assertEqual(actual,forward_all(model,context))
                            self.assertEqual(a,b)
                            self.assertEqual(snapshot(child),snapshot(reference))
                        self.assertEqual(snapshot(root),before)
                        cases += 1
        self.assertEqual(cases,504)

    def test_sibling_ancestor_rows_are_independent_in_both_directions(self):
        model=Model(8,8,2,4);root=Cache(model);prefill(model,root,[1,2,3,4])
        left=prefix_cache(model,root,2);right=prefix_cache(model,root,3)
        child=prefix_cache(model,left,1);full=prefix_cache(model,root,4)
        self.assertEqual(snapshot(full),snapshot(fork_cache(model,root)))
        all_caches=[root,left,right,child,full]
        for a in all_caches:
            for b in all_caches:
                if a is b:continue
                for x,y in ((a.k,b.k),(a.v,b.v)):
                    self.assertIsNot(x,y)
                    for hx,hy in zip(x,y):
                        self.assertIsNot(hx,hy)
                        for rx,ry in zip(hx,hy):self.assertIsNot(rx,ry)
        before=[snapshot(c) for c in all_caches]
        left.k[0][0][0]+=1;left.v[1][1][0]-=1
        for i in (0,2,3,4):self.assertEqual(snapshot(all_caches[i]),before[i])
        frozen=snapshot(left);root.k[1][0][0]+=2;root.v[0][0][0]-=2
        self.assertEqual(snapshot(left),frozen)
        for i in (2,3,4):self.assertEqual(snapshot(all_caches[i]),before[i])

    def test_empty_prefix_and_reprefill_vs_stale_position(self):
        model=Model(8,8,2,9);root=Cache(model);prefill(model,root,[1,2,3,4])
        empty=prefix_cache(model,root,0)
        self.assertEqual(prefill(model,empty,[5,6]),forward_all(model,[5,6]))
        empty=prefix_cache(model,root,0)
        self.assertEqual(forward_last(model,[5,6],cache=empty),forward_all(model,[5,6]))
        shortened=prefix_cache(model,root,2)
        self.assertEqual(cache_token(model,shortened,7),forward_all(model,[1,2,7]))
        full=fork_cache(model,root)
        self.assertNotEqual(cache_token(model,full,7),forward_all(model,[1,2,7]))
        self.assertEqual(root.length,4)

    def test_bad_requests_are_atomic_and_use_existing_ownership_guards(self):
        model=Model(8,4,2,1);other=Model(8,4,2,1)
        root=Cache(model);prefill(model,root,[1,2,3]);before=snapshot(root)
        for length in (-1,4,True,1.,None):
            with self.assertRaises(ValueError):prefix_cache(model,root,length)
            self.assertEqual(snapshot(root),before)
        with self.assertRaises(ValueError):prefix_cache(other,root,1)
        self.assertEqual(snapshot(root),before)
        for field,value in [('length',True),('length',2),('k',None),('v',[[[float('nan')]]])]:
            bad=fork_cache(model,root);setattr(bad,field,value)
            before_bad=snapshot(bad)
            with self.assertRaises(ValueError):prefix_cache(model,bad,0)
            self.assertEqual(repr(snapshot(bad)),repr(before_bad))
        with self.assertRaises(ValueError):prefix_cache(model,None,0)


if __name__=='__main__':unittest.main()
