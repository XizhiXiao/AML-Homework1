"""Part 4: plan, run, collect and fit controlled scaling experiments.

Training remains in aml.train. This module does not change the main-model run.
"""
import argparse
import copy
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

MODELS = [
    ('n5m', 256, 6, 4), ('n11m', 384, 6, 6),
    ('n25m', 512, 8, 8), ('n50m', 512, 16, 8), ('n99m', 768, 14, 12),
]
GROUP_GPUS = {
    '0': 'GPU-0a24f921-95ca-a59a-91f6-15d0d83901bd',
    '1': 'GPU-6de7363b-d633-c651-5e43-7bf2ecfca355',
}

GPU5_GPUS = {
    '1': 'GPU-28304eda-a96c-7bd9-5506-4df2d3a48523',
    '6': 'GPU-6d559c06-4544-db21-e20e-914cc85d62f8',
    '7': 'GPU-b083383a-89a9-47fa-aa29-59c50a7a7cc2',
}


def read(path):
    return json.loads(Path(path).read_text())


def write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def counts(m):
    d, f, layers = m['d_model'], m['d_ff'], m['n_layers']
    # Attention matrices+biases, FFN matrices+biases, two LayerNorms per block.
    n = layers * (4*d*d + 2*d*f + 9*d + f) + 2*d
    total = n + m['vocab_size']*d + m['context_length']*d
    if not m['tie_weights']:
        n += m['vocab_size']*d
        total += m['vocab_size']*d
    return n, total


def make_plan(base, profile='pilot'):
    batch = 32768
    reference_d = base['train']['max_steps'] * base['train']['micro_batch_size'] * base['train']['gradient_accumulation'] * base['model']['context_length']
    fixed_d = 67108864 if profile == 'pilot' else reference_d
    budgets = [1e16, 3e16] if profile == 'pilot' else [1e16, 3e16, 1e17]
    entries = []
    for name, width, layers, heads in MODELS:
        m = copy.deepcopy(base['model'])
        m.update(d_model=width, n_layers=layers, n_heads=heads, d_ff=4*width)
        n, total = counts(m)
        for cohort, target in [('fixed_d', fixed_d)] + [('isoflop', c) for c in budgets]:
            requested_d = target if cohort == 'fixed_d' else target/(6*n)
            steps = max(1, round(requested_d / batch))
            if steps < 32:
                raise ValueError('Budget too small: fewer than 32 optimizer updates')
            cfg = copy.deepcopy(base)
            cfg['model'] = m.copy()
            t = cfg['train']
            t.update(micro_batch_size=4, gradient_accumulation=batch//(4*m['context_length']),
                     max_steps=steps, warmup_steps=max(1, round(steps*0.02)),
                     eval_interval=max(1, steps//20), log_interval=max(1, steps//100),
                     save_interval=max(1, steps//5))
            if t['gradient_accumulation'] * 4 * m['context_length'] != batch:
                raise ValueError('Context does not divide experimental batch')
            # A documented starting recipe, not a claim of optimal hyperparameters.
            t['learning_rate'] = base['train']['learning_rate'] * math.sqrt(batch / (base['train']['micro_batch_size']*base['train']['gradient_accumulation']*m['context_length']))
            d = steps*batch
            entries.append({'id': f'{name}_d{d}_s{t["seed"]}', 'cohort': cohort,
                            'budget': target, 'N': n, 'N_total': total, 'D': d,
                            'C': 6*n*d, 'budget_relative_error': 0 if cohort == 'fixed_d' else (6*n*d-target)/target,
                            'config': cfg})
    main_n, _ = counts(base['model'])
    return {'version': 1, 'profile': profile, 'parameter_convention': 'non_embedding',
            'reference_config': base, 'reference_N': main_n, 'reference_D': reference_d,
            'reference_C': 6*main_n*reference_d, 'runs': entries,
            'total_C': sum(r['C'] for r in entries),
            'notes': ['Independent complete cosine schedule for EACH budget.',
                      'Pilot fixed-D curve cannot predict main model at a different D.',
                      'Smaller batch and scaled LR are a starting recipe; tune before interpreting exponents.',
                      'Use final-step validation, not best checkpoint or intermediate cosine checkpoint.']}


def plan_command(args):
    out = Path(args.output)
    if out.exists() and any(out.iterdir()):
        raise ValueError('Plan output must be empty')
    manifest = make_plan(read(args.base), args.profile)
    for run in manifest['runs']:
        write(out / 'configs' / (run['id'] + '.json'), run['config'])
    write(out / 'plan.json', manifest)
    with open(out / 'budget.csv', 'w') as stream:
        writer = csv.DictWriter(stream, fieldnames=['id', 'cohort', 'budget', 'N', 'N_total', 'D', 'C', 'budget_relative_error'])
        writer.writeheader()
        writer.writerows({k: v for k, v in r.items() if k != 'config'} for r in manifest['runs'])
    print(json.dumps({'runs': len(manifest['runs']), 'total_estimated_FLOPs': manifest['total_C'],
                      'relative_to_main': manifest['total_C']/manifest['reference_C']}, indent=2))


def final_row(directory, expected=None):
    directory = Path(directory)
    cfg = read(directory/'config.json')
    if expected is not None and cfg != expected:
        raise ValueError(f'Configuration mismatch: {directory}')
    records = [json.loads(line) for line in (directory/'metrics.jsonl').read_text().splitlines() if line.strip()]
    step = cfg['train']['max_steps']
    vals = [r for r in records if r['kind'] == 'val' and r['step'] == step]
    if not vals or not (directory/'last.pt').exists():
        raise ValueError(f'Run is not complete: {directory}')
    row = vals[-1]
    n, total = counts(cfg['model'])
    d = step*cfg['train']['micro_batch_size']*cfg['train']['gradient_accumulation']*cfg['model']['context_length']
    if row['tokens_seen'] != d or not math.isfinite(row['loss']):
        raise ValueError('Invalid token accounting/loss')
    meta = read(directory/'data_metadata.json')
    signature = {'metadata_sha256': hashlib.sha256((directory/'data_metadata.json').read_bytes()).hexdigest(),
                 'tokenizer': meta['tokenizer_sha256'], 'policy': meta['validation_policy'],
                 'eval_documents': cfg['train']['eval_documents'],
                 'eval_tokens': row['tokens'], 'eval_bytes': row['bytes'],
                 'context': cfg['model']['context_length']}
    return dict(row, N=n, N_total=total, D=d, C=6*n*d, signature=signature, directory=str(directory)), records


def run_command(args):
    plan = read(args.plan)
    chosen = [r for r in plan['runs'] if not args.only or r['id'] in args.only]
    if not chosen:
        raise ValueError('No matching runs')
    if args.only and set(args.only) - {r['id'] for r in chosen}:
        raise ValueError('Unknown run ID')
    if args.only:
        by_id = {r['id']: r for r in chosen}
        chosen = [by_id[name] for name in dict.fromkeys(args.only)]
    # Execution is opt-in. A dry run never imports torch or allocates a GPU.
    if not args.execute:
        print('\n'.join(r['id'] for r in chosen)); return
    allowed = GPU5_GPUS if args.host_profile == 'gpu5' else GROUP_GPUS
    if args.gpu not in allowed:
        raise ValueError('GPU is not authorized for this host profile')
    uuid = allowed[args.gpu]
    busy = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader'], text=True)
    if uuid in busy:
        raise RuntimeError('Selected group GPU has a compute process; do not interrupt it')
    import fcntl
    root = Path(args.runs); root.mkdir(parents=True, exist_ok=True)
    # Global per-GPU advisory lock catches concurrent runners from this project.
    lock = open('/tmp/aml-scaling-' + args.host_profile + '-gpu' + args.gpu + '.lock', 'w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=uuid, OMP_NUM_THREADS='4', TOKENIZERS_PARALLELISM='false')
    config_root = root / '_configs'
    for run in chosen:
        dest = root / run['id']
        config = config_root / (run['id']+'.json')
        write(config, run['config'])
        if (dest/'last.pt').exists():
            try:
                final_row(dest, run['config']); print('SKIP complete:', run['id']); continue
            except ValueError as error:
                if 'not complete' not in str(error): raise
        command = [sys.executable, '-m', 'aml.train', '--config', str(config), '--data', args.data, '--output', str(dest)]
        if (dest/'last.pt').exists():
            command += ['--resume', str(dest/'last.pt')]
        elif dest.exists() and any(dest.iterdir()):
            raise RuntimeError('Incomplete directory without recovery checkpoint: ' + str(dest))
        busy = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader'], text=True)
        if uuid in busy:
            raise RuntimeError('Selected group GPU became occupied; remaining runs were not started')
        print('START', run['id'], flush=True)
        with open(root/(run['id']+'.log'), 'a') as log:
            subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        final_row(dest, run['config'])
    lock.close()


def collect_command(args):
    manifest = read(args.plan)
    rows, curves, missing = [], {}, []
    signature = None
    for r in manifest['runs']:
        directory = Path(args.runs)/r['id']
        if not directory.exists():
            missing.append(r['id']); continue
        try:
            row, history = final_row(directory, r['config'])
        except (FileNotFoundError, ValueError) as error:
            if isinstance(error, ValueError) and 'not complete' not in str(error): raise
            missing.append(r['id']); continue
        if signature is not None and row['signature'] != signature:
            raise ValueError('Data/validation protocol mismatch')
        signature = row['signature']
        rows.append(dict(row, id=r['id'], cohort=r['cohort'], budget=r['budget']))
        curves[r['id']] = history
    main = None
    if args.main:
        main, _ = final_row(args.main, manifest['reference_config'])
        if signature is not None and main['signature'] != signature:
            raise ValueError('Main model uses different data or validation protocol')
    write(args.output, {'plan': manifest, 'rows': rows, 'curves': curves, 'missing': missing, 'main': main})
    print(f'Collected {len(rows)} completed runs; {len(missing)} missing/incomplete')


def fit_power(x, y, offset=True):
    import numpy as np
    from scipy.optimize import least_squares
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(np.unique(x)) < (4 if offset else 3):
        raise ValueError('Not enough distinct scales for this fit')
    scale = float(np.exp(np.mean(np.log(x))))
    z = x/scale
    best = None
    for alpha in (0.05, 0.15, 0.4, 0.8):
        if offset:
            start = [float(min(y)*0.5), float(max(y)-min(y)*0.5), alpha]
            bounds = ([0, 0, 0.001], [float(min(y)), float(max(y)*100), 3])
            fun = lambda p: p[0]+p[1]*z**(-p[2])-y
        else:
            start = [float(np.mean(y)), alpha]
            bounds = ([0, 0.001], [float(max(y)*100), 3])
            fun = lambda p: p[0]*z**(-p[1])-y
        result = least_squares(fun, start, bounds=bounds, max_nfev=10000)
        if result.success and (best is None or sum(result.fun**2) < sum(best.fun**2)):
            best = result
    if best is None:
        raise RuntimeError('Fit failed to converge')
    e, a, alpha = best.x if offset else (0, *best.x)
    return {'E': float(e), 'A': float(a*scale**alpha), 'alpha': float(alpha),
            'rmse': float(np.sqrt(np.mean(best.fun**2))),
            'jacobian_condition': float(np.linalg.cond(best.jac)),
            'bound_hit': bool(np.any(best.active_mask)), 'n_points': len(x)}


def predict(fit, x):
    return fit['E'] + fit['A']/x**fit['alpha']


def analyze_command(args):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    if args.target_n is not None and args.target_n <= 0:
        raise ValueError('target-n must be positive')
    bundle = read(args.results)
    if bundle['missing'] and not args.allow_partial:
        raise ValueError('Incomplete design; complete runs or explicitly --allow-partial')
    rows = bundle['rows']
    if not rows: raise ValueError('No completed experiments')
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    report = {'partial': bool(bundle['missing']), 'missing': bundle['missing'], 'warnings': [
        'Fits describe this recipe and data, not universal constants.',
        'Main model is held out from all fits.',
        'Single-seed results do not provide a reliable statistical confidence interval.',
        '6ND ignores embedding/head and attention overhead, especially important for tiny models.']}
    fixed = sorted([r for r in rows if r['cohort']=='fixed_d'], key=lambda r:r['N'])
    if len({r['D'] for r in fixed}) > 1: raise ValueError('Fixed-D group has unequal token budgets')
    if len(fixed) >= 4:
        fit = fit_power([r['N'] for r in fixed], [r['loss'] for r in fixed])
        report['L_N'] = dict(fit, D=fixed[0]['D'])
        target_n = args.target_n or bundle['plan']['reference_N']
        prediction = predict(fit, target_n)
        # Leave-one-scale-out sensitivity, not a confidence interval.
        sensitivity = []
        if len(fixed) >= 5:
            for i in range(len(fixed)):
                keep = fixed[:i]+fixed[i+1:]
                f = fit_power([r['N'] for r in keep], [r['loss'] for r in keep])
                sensitivity.append(predict(f, target_n))
        report['prediction'] = {'N': target_n, 'D': fixed[0]['D'], 'loss': prediction,
                                'leave_one_scale_out_predictions': sensitivity,
                                'extrapolation_ratio': target_n/max(r['N'] for r in fixed)}
        main = bundle['main']
        if main and target_n == main['N'] and fixed[0]['D'] == main['D']:
            report['main_comparison'] = {'measured_loss': main['loss'], 'predicted_loss': prediction,
                                        'absolute_error': abs(prediction-main['loss']),
                                        'relative_error': abs(prediction-main['loss'])/main['loss']}
        else:
            report['warnings'].append('No matched completed main result at this N and D; do not claim a validated main-model prediction.')
        xx = np.geomspace(min(r['N'] for r in fixed), max(target_n,max(r['N'] for r in fixed)),200)
        fig, ax = plt.subplots()
        ax.scatter([r['N'] for r in fixed],[r['loss'] for r in fixed],label='Small models, fixed D')
        ax.plot(xx,[predict(fit,x) for x in xx],label='E + A/N^alpha')
        ax.scatter([target_n],[prediction],marker='x',label='Held-out prediction')
        ax.set(xscale='log',xlabel='Non-embedding parameters N',ylabel='Validation loss');ax.legend()
        fig.tight_layout();fig.savefig(out/'loss_vs_parameters.png',dpi=160);plt.close(fig)
    else: report['warnings'].append('Need at least four fixed-D scales for offset fit.')
    budgets = sorted({r['budget'] for r in rows if r['cohort']=='isoflop'})
    frontier=[]
    fig, ax = plt.subplots()
    for budget in budgets:
        group=sorted([r for r in rows if r['cohort']=='isoflop' and r['budget']==budget],key=lambda r:r['N'])
        winner=min(group,key=lambda r:r['loss'])
        edge=winner['N'] in (group[0]['N'],group[-1]['N'])
        frontier.append({'budget':budget,'actual_C':winner['C'],'N':winner['N'],'D':winner['D'],'loss':winner['loss'],'boundary_optimum':edge})
        ax.plot([r['N'] for r in group],[r['loss'] for r in group],'-o',label=f'C≈{budget:.2g}')
        if edge:report['warnings'].append(f'Budget {budget:g}: best sampled model is at grid boundary; widen search before claiming a true optimum.')
    ax.set(xscale='log',xlabel='N',ylabel='Validation loss')
    if budgets:ax.legend()
    fig.tight_layout();fig.savefig(out/'isoflop.png',dpi=160);plt.close(fig)
    report['sampled_frontier']=frontier
    if len(frontier)>=3:
        report['L_C']=fit_power([r['budget'] for r in frontier],[r['loss'] for r in frontier],offset=False)
        report['allocation_exponents']={
            'N_vs_C':float(np.polyfit(np.log([r['budget'] for r in frontier]),np.log([r['N'] for r in frontier]),1)[0]),
            'D_vs_C':float(np.polyfit(np.log([r['budget'] for r in frontier]),np.log([r['D'] for r in frontier]),1)[0]),
            'exploratory_only':any(r['boundary_optimum'] for r in frontier)}
    else:report['warnings'].append('Pilot has fewer than three compute budgets: show frontier points, defer exponent fitting.')
    fig,ax=plt.subplots()
    if frontier:ax.plot([r['actual_C'] for r in frontier],[r['loss'] for r in frontier],'-o',label='Sampled compute-optimal envelope')
    if 'L_C' in report:
        xx=np.geomspace(min(r['budget'] for r in frontier),max(r['budget'] for r in frontier),100)
        ax.plot(xx,[predict(report['L_C'],x) for x in xx],'--',label='A/C^alpha')
    ax.set(xscale='log',xlabel='Approximate training FLOPs',ylabel='Validation loss')
    if frontier:ax.legend()
    fig.tight_layout();fig.savefig(out/'compute_frontier.png',dpi=160);plt.close(fig)
    for axis in ['tokens_seen','compute']:
        fig,ax=plt.subplots(figsize=(9,6))
        for row in rows:
            history=[r for r in bundle['curves'][row['id']] if r['kind']=='val' and r['tokens_seen']>0]
            xs=[r['tokens_seen']*(6*row['N'] if axis=='compute' else 1) for r in history]
            ax.plot(xs,[r['loss'] for r in history],label=row['id'],alpha=0.7)
        ax.set(xscale='log',xlabel=axis,ylabel='Validation loss');ax.legend(fontsize=5)
        fig.tight_layout();fig.savefig(out/f'curves_{axis}.png',dpi=160);plt.close(fig)
    write(out/'analysis.json',report)
    with open(out/'results.csv','w') as f:
        keys=['id','cohort','budget','N','N_total','D','C','loss','ppl','bpb']
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows({k:r[k] for k in keys} for r in rows)
    print(json.dumps(report,indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    q=sub.add_parser('plan');q.add_argument('--base',default='configs/main.json');q.add_argument('--profile',choices=['pilot','full'],default='pilot');q.add_argument('--output',required=True);q.set_defaults(func=plan_command)
    q=sub.add_parser('run');q.add_argument('--plan',required=True);q.add_argument('--data',required=True);q.add_argument('--runs',required=True);q.add_argument('--host-profile',choices=['course','gpu5'],default='course');q.add_argument('--gpu',choices=['0','1','6','7'],default='1');q.add_argument('--only',nargs='+');q.add_argument('--execute',action='store_true');q.set_defaults(func=run_command)
    q=sub.add_parser('collect');q.add_argument('--plan',required=True);q.add_argument('--runs',required=True);q.add_argument('--main');q.add_argument('--output',required=True);q.set_defaults(func=collect_command)
    q=sub.add_parser('analyze');q.add_argument('--results',required=True);q.add_argument('--output',required=True);q.add_argument('--allow-partial',action='store_true');q.add_argument('--target-n',type=int,help='Optional path-2 larger NON-embedding parameter target, at fixed D');q.set_defaults(func=analyze_command)
    args=p.parse_args();args.func(args)


if __name__=='__main__':main()
