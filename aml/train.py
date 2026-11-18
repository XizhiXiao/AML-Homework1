"""Single-device pretraining with accumulation, AMP, evaluation and exact RNG resume."""
import argparse
from contextlib import nullcontext
from dataclasses import asdict
import hashlib
import inspect
import json
import math
from pathlib import Path
import random
import time
import numpy as np
import torch
from .data import Corpus
from .metrics import evaluate
from .model import GPT, ModelConfig


def select_device(name):
    if name != 'auto':
        return torch.device(name)
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def amp_context(device, precision):
    if precision == 'fp32':
        return nullcontext
    if device.type != 'cuda':
        raise ValueError('Mixed precision configuration requires CUDA; use fp32 for CPU/MPS')
    if precision == 'bf16' and not torch.cuda.is_bf16_supported():
        raise ValueError('Device does not support BF16; select fp16 or fp32')
    if precision not in ('bf16', 'fp16'):
        raise ValueError('Unknown precision')
    return lambda: torch.autocast('cuda', dtype=torch.bfloat16 if precision == 'bf16' else torch.float16)


def learning_rate(step, cfg):
    warmup, total = cfg['warmup_steps'], cfg['max_steps']
    peak, floor = cfg['learning_rate'], cfg['learning_rate'] * cfg['min_lr_ratio']
    if step < warmup:
        return peak * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup - 1)
    return floor + (peak - floor) * (1 + math.cos(math.pi * progress)) / 2


def atomic_save(obj, path):
    path = Path(path)
    temp = path.with_suffix('.tmp')
    torch.save(obj, temp)
    temp.replace(path)


def load_checkpoint(path, device='cpu'):
    # Only load checkpoints you created/trust; they contain Python RNG states.
    kwargs = {'map_location': device}
    if 'weights_only' in inspect.signature(torch.load).parameters:
        kwargs['weights_only'] = False
    return torch.load(path, **kwargs)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--data', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--resume', help='Trusted last.pt, with the SAME config and data')
    p.add_argument('--device', help='Override config device')
    p.add_argument('--stop-after', type=int, help='Stop after this absolute step, preserving original LR horizon')
    args = p.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    t = cfg['train']
    if args.device:
        t['device'] = args.device
    for key in ('micro_batch_size', 'gradient_accumulation', 'max_steps', 'eval_interval', 'log_interval', 'save_interval'):
        if t[key] <= 0:
            raise ValueError(key + ' must be positive')
    if t['eval_documents'] < 0:
        raise ValueError('eval_documents must be nonnegative')
    if not 0 <= t['warmup_steps'] < t['max_steps']:
        raise ValueError('warmup_steps must be in [0, max_steps)')
    out = Path(args.output)
    if out.exists() and any(out.iterdir()) and not args.resume:
        raise ValueError('Use an empty output directory or --resume')
    out.mkdir(parents=True, exist_ok=True)
    random.seed(t['seed']); np.random.seed(t['seed']); torch.manual_seed(t['seed'])
    device = select_device(t['device'])
    autocast = amp_context(device, t['precision'])
    corpus = Corpus(args.data, cfg['model']['context_length'])
    if corpus.meta['vocab_size'] != cfg['model']['vocab_size']:
        raise ValueError('Config vocabulary differs from prepared tokenizer; inspect metadata.json')
    data_fingerprint = hashlib.sha256((corpus.path / 'metadata.json').read_bytes()).hexdigest()
    generator = torch.Generator().manual_seed(t['seed'] + 1)
    model = GPT(ModelConfig(**cfg['model'])).to(device)
    decay = [p for p in model.parameters() if p.ndim >= 2]
    no_decay = [p for p in model.parameters() if p.ndim < 2]
    optimizer = torch.optim.AdamW([{'params': decay, 'weight_decay': t['weight_decay']},
                                  {'params': no_decay, 'weight_decay': 0.0}],
                                 lr=t['learning_rate'], betas=tuple(t['betas']))
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == 'cuda' and t['precision'] == 'fp16')
    step, best, tokens_seen = 0, float('inf'), 0
    if args.resume:
        checkpoint = load_checkpoint(args.resume)
        if checkpoint['config'] != cfg or checkpoint['data_fingerprint'] != data_fingerprint:
            raise ValueError('Resume requires identical configuration and data metadata')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scaler.load_state_dict(checkpoint['scaler'])
        step, best, tokens_seen = checkpoint['step'], checkpoint['best_loss'], checkpoint['tokens_seen']
        generator.set_state(checkpoint['batch_rng'])
        torch.set_rng_state(checkpoint['torch_rng'])
        random.setstate(checkpoint['python_rng']); np.random.set_state(checkpoint['numpy_rng'])
        if device.type == 'cuda' and checkpoint['cuda_rng'] is not None:
            torch.cuda.set_rng_state_all(checkpoint['cuda_rng'])
        # Remove log records newer than the checkpoint when replaying interrupted work.
        log_path = out / 'metrics.jsonl'
        if log_path.exists():
            records = [json.loads(line) for line in log_path.read_text().splitlines() if line]
            log_path.write_text(''.join(json.dumps(r) + '\n' for r in records if r['step'] <= step))
    (out / 'config.json').write_text(json.dumps(cfg, indent=2))
    (out / 'data_metadata.json').write_text(json.dumps(corpus.meta, indent=2))
    environment = {'torch': torch.__version__, 'numpy': np.__version__, 'device': str(device),
                   'cuda_visible_devices': __import__('os').environ.get('CUDA_VISIBLE_DEVICES'),
                   'gpu_name': torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
                   'source_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in Path('aml').glob('*.py')}}
    (out / 'environment.json').write_text(json.dumps(environment, indent=2))
    counts = model.parameter_counts()
    (out / 'model_info.json').write_text(json.dumps(counts, indent=2))
    batch_tokens = t['micro_batch_size'] * t['gradient_accumulation'] * model.config.context_length
    print(json.dumps({'parameters': counts, 'device': str(device), 'tokens_per_step': batch_tokens,
                      'planned_tokens': t['max_steps'] * batch_tokens}), flush=True)

    def log(record):
        with open(out / 'metrics.jsonl', 'a') as stream:
            stream.write(json.dumps(record) + '\n')
        print(json.dumps(record), flush=True)

    def save(name):
        atomic_save({'config': cfg, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                     'scaler': scaler.state_dict(), 'step': step, 'tokens_seen': tokens_seen, 'best_loss': best,
                     'data_fingerprint': data_fingerprint, 'tokenizer_sha256': corpus.meta['tokenizer_sha256'],
                     'batch_rng': generator.get_state(), 'torch_rng': torch.get_rng_state(),
                     'python_rng': random.getstate(), 'numpy_rng': np.random.get_state(),
                     'cuda_rng': torch.cuda.get_rng_state_all() if device.type == 'cuda' else None}, out / name)

    def validate():
        nonlocal best
        result = evaluate(model, corpus, device, autocast, t['micro_batch_size'],
                          t['eval_documents'])
        log({'kind': 'val', 'step': step, 'tokens_seen': tokens_seen, **result})
        if result['loss'] < best:
            best = result['loss']
            save('best.pt')
        return result

    if step == 0:
        validate()
    model.train()
    stop = min(t['max_steps'], args.stop_after if args.stop_after is not None else t['max_steps'])
    if stop <= step:
        raise ValueError('Requested stop must exceed checkpoint step')
    interval_loss, interval_steps, timer = 0.0, 0, time.perf_counter()
    while step < stop:
        optimizer.zero_grad(set_to_none=True)
        lr = learning_rate(step, t)
        for group in optimizer.param_groups:
            group['lr'] = lr
        step_loss = 0.0
        for _ in range(t['gradient_accumulation']):
            x, y = corpus.batch(t['micro_batch_size'], generator, device)
            with autocast():
                _, loss = model(x, y)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite loss; recover from last.pt and investigate')
            step_loss += loss.item() / t['gradient_accumulation']
            scaler.scale(loss / t['gradient_accumulation']).backward()
        scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), t['grad_clip'])
        if not torch.isfinite(norm):
            raise FloatingPointError('Nonfinite gradients; optimizer update cancelled')
        scaler.step(optimizer); scaler.update()
        step += 1; tokens_seen += batch_tokens
        interval_loss += step_loss; interval_steps += 1
        if step % t['log_interval'] == 0 or step == stop:
            elapsed = time.perf_counter() - timer
            log({'kind': 'train', 'step': step, 'tokens_seen': tokens_seen,
                 'loss': interval_loss / interval_steps, 'lr': lr, 'grad_norm': float(norm),
                 'tokens_per_second': interval_steps * batch_tokens / elapsed,
                 'compute_flops_estimate': 6 * counts['non_embedding'] * tokens_seen})
            interval_loss, interval_steps = 0.0, 0
            timer = time.perf_counter()
        if step % t['eval_interval'] == 0 or step == stop:
            validate()
        if step % t['save_interval'] == 0 or step == stop:
            save('last.pt')
    print(f'Finished step {step}; best validation loss {best:.6f}. Checkpoints: {out}', flush=True)


if __name__ == '__main__':
    main()
