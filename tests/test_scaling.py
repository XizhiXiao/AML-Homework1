import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from aml.scaling import make_plan, counts, fit_power, predict, final_row, analyze_command
from aml.model import GPT, ModelConfig


class ScalingTests(unittest.TestCase):
    def test_parameters_and_budgets(self):
        m=dict(vocab_size=257,context_length=32,d_model=32,n_layers=2,n_heads=4,d_ff=80,dropout=0.,tie_weights=True)
        actual=GPT(ModelConfig(**m)).parameter_counts()
        self.assertEqual(counts(m),(actual['non_embedding'],actual['total']))
        base=json.loads(Path('configs/main.json').read_text())
        full=make_plan(base,'full');pilot=make_plan(base,'pilot')
        self.assertEqual(len(full['runs']),20)
        for r in full['runs']:
            t=r['config']['train'];self.assertEqual(r['D'],t['max_steps']*32768)
            self.assertLess(t['warmup_steps'],t['max_steps'])
            if r['cohort']=='fixed_d':self.assertEqual(r['D'],full['reference_D'])
            else:self.assertLess(abs(r['budget_relative_error']),0.002)
        shared={r['id']:r['config'] for r in full['runs']}
        for r in pilot['runs']:
            if r['id'] in shared:self.assertEqual(shared[r['id']],r['config'])

    def test_recover_power(self):
        x=np.geomspace(5e6,1e8,8);y=1.5+40/x**0.25
        fit=fit_power(x,y)
        self.assertAlmostEqual(fit['alpha'],0.25,places=3)
        self.assertAlmostEqual(predict(fit,2e8),1.5+40/(2e8)**0.25,places=4)
        with self.assertRaises(ValueError):fit_power(x[:3],y[:3])

    def test_incomplete_not_collected(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp);cfg=json.loads(Path('configs/smoke.json').read_text())
            (p/'config.json').write_text(json.dumps(cfg))
            (p/'metrics.jsonl').write_text(json.dumps({'kind':'val','step':4,'loss':1.0})+'\n')
            (p/'last.pt').touch()
            with self.assertRaisesRegex(ValueError,'not complete'):final_row(p)

    def test_analysis_synthetic_not_real_results(self):
        plan=make_plan(json.loads(Path('configs/main.json').read_text()),'full')
        rows=[];curves={}
        for run in plan['runs']:
            loss=1.5+40/run['N']**0.25+30/run['D']**0.3
            row={**run,'loss':loss,'ppl':float(np.exp(loss)),'bpb':loss/3}
            rows.append(row);curves[run['id']]=[{'kind':'val','tokens_seen':run['D'],'loss':loss}]
        main_loss=1.5+40/plan['reference_N']**0.25+30/plan['reference_D']**0.3
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp);source=p/'synthetic.json'
            source.write_text(json.dumps({'plan':plan,'rows':rows,'curves':curves,'missing':[],
                                         'main':{'N':plan['reference_N'],'D':plan['reference_D'],'loss':main_loss}}))
            with contextlib.redirect_stdout(io.StringIO()):
                analyze_command(SimpleNamespace(results=source,output=p/'out',allow_partial=False,target_n=None))
            report=json.loads((p/'out/analysis.json').read_text())
            self.assertLess(report['main_comparison']['absolute_error'],1e-4)
            self.assertEqual(len(report['sampled_frontier']),3)
            self.assertEqual(len(list((p/'out').glob('*.png'))),5)
            self.assertTrue(any(r['boundary_optimum'] for r in report['sampled_frontier']))


if __name__=='__main__':unittest.main()
