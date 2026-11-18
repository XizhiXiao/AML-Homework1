"""Add five targeted runs without changing the original scaling plans."""
import csv
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aml import scaling

scaling.MODELS = [('n1p78m',192,4,3),('n2p67m',192,6,3),('n16p63m',480,6,6)]
plan = scaling.make_plan(scaling.read('configs/main.json'), 'full')
selected = {'n1p78m': [1e16], 'n2p67m': [1e16,3e16], 'n16p63m': [3e16,1e17]}
plan['runs'] = [r for r in plan['runs'] if r['cohort']=='isoflop' and r['budget'] in selected[r['id'].split('_')[0]]]
plan['profile'] = 'isoflop_refinement'
plan['total_C'] = sum(r['C'] for r in plan['runs'])
plan['notes'] += ['Widths/layers/heads: 192/4/3, 192/6/3, 480/6/6. Use exact counts, not nominal model labels.', 'D counts processed tokens including repeats; 6ND omits embedding/head and attention overhead.']
out = Path('experiments/path1_refinement')
for r in plan['runs']:
 scaling.write(out/'configs'/(r['id']+'.json'),r['config'])
 print(r['id'], 'N=',r['N'],'total=',r['N_total'],'steps=',r['config']['train']['max_steps'])
scaling.write(out/'plan.json',plan)
with (out/'budget.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=['id','cohort','budget','N','N_total','D','C','budget_relative_error']);w.writeheader();w.writerows({k:v for k,v in r.items() if k!='config'} for r in plan['runs'])
