"""Check allocation and paired scoring against independent, small constructions."""
import itertools
import tempfile
import unittest
from pathlib import Path
import json

from search import allocate_exact, compare, nll, swap_neighbors
from reference import assert_equivalent


class SearchTests(unittest.TestCase):
    def test_reference_allows_only_run_specific_differences(self):
        a=['vllm','serve','model','--port','18074','--max-model-len','163840',
           '--extra-body',json.dumps({'temperature':1.,'cache_salt':'old'})]
        b=a.copy();b[4]='18076';b[-1]=json.dumps({'cache_salt':'new','temperature':1.})
        assert_equivalent(a,b)
        b[6]='65536'
        with self.assertRaises(ValueError):assert_equivalent(a,b)
        b[6]='163840';b[-1]=json.dumps({'temperature':.7,'cache_salt':'new'})
        with self.assertRaises(ValueError):assert_equivalent(a,b)

    def test_budget_matches_exhaustive_search(self):
        curves=[{k:{'gen':v} for k,v in d.items()} for d in (
            {3:-.2,4:0.,5:.3,6:0.}, {3:.8,4:.15,5:.1,6:0.}, {3:.4,4:.3,5:.05,6:0.})]
        for budget in range(9,19):
            expected=min(sum(curves[i][k]['gen'] for i,k in enumerate(ks))
                         for ks in itertools.product(range(3,7),repeat=3) if sum(ks)==budget)
            loss,ks=allocate_exact(curves,budget,'gen')
            self.assertEqual(sum(ks),budget)
            self.assertAlmostEqual(loss,expected)

    def test_slot_swaps_preserve_budget_and_bounds(self):
        curves=[{k:{'balanced':(6-k)**2*(i+1)} for k in range(3,7)} for i in range(4)]
        for x in swap_neighbors([3,4,5,6],curves,100):
            self.assertEqual(sum(x),18)
            self.assertTrue(all(3<=k<=6 for k in x))

    def test_identity_and_modality_pairing(self):
        r={'batch':'e065_example','task_id':'t','pair':0,'attempt':0,
           'n_gen':10,'nll_gen':5.,'n_blk':2,'nll_blk':.4}
        b={m:{'gen':.5,'tool':.2,'rows':[dict(r,batch='e065t_example' if m=='txt' else 'e065_example')]}
           for m in ('img','txt')}
        result=compare(b,b)
        self.assertEqual(result['joint_score'],0.)
        self.assertEqual(result['stderr'],0.)
        c=json.loads(json.dumps(b))
        for m in c:
            c[m]['gen']*=1.1;c[m]['tool']*=1.2
            c[m]['rows'][0]['nll_gen']*=1.1;c[m]['rows'][0]['nll_blk']*=1.2
        self.assertAlmostEqual(compare(c,b)['joint_score'],.175)

    def test_duplicate_rows_do_not_pass_completeness(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);(p/'data').mkdir();(p/'results/x').mkdir(parents=True)
            (p/'data/manifest.json').write_text(json.dumps({'paired_keys':{'screen':[['e065_x','a',0,0],['e065_x','b',0,0]]}}))
            r={'batch':'e065_x','task_id':'a','pair':0,'attempt':0,'n_gen':10,'n_blk':2,'nll_gen':1.,'nll_blk':1.}
            (p/'results/x/screen_img.jsonl').write_text((json.dumps(r)+'\n')*2)
            with self.assertRaises(AssertionError):nll(p,'x','screen')


if __name__=='__main__':unittest.main()
