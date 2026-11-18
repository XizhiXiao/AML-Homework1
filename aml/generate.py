import argparse
import hashlib
import json
from pathlib import Path
import torch
from .data import Tokenizer
from .model import GPT, ModelConfig
from .train import load_checkpoint, select_device


def main():
    p = argparse.ArgumentParser(description='Generate reproducible samples from a trusted checkpoint')
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--data', required=True)
    p.add_argument('--prompt', action='append', help='Repeat for multiple prompts')
    p.add_argument('--output', required=True)
    p.add_argument('--device', default='auto')
    p.add_argument('--max-new-tokens', type=int, default=160)
    p.add_argument('--temperature', type=float, default=0.8)
    p.add_argument('--top-k', type=int, default=50)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    torch.manual_seed(args.seed)
    device = select_device(args.device)
    checkpoint = load_checkpoint(args.checkpoint)
    path = Path(args.data)
    if hashlib.sha256((path / 'tokenizer.json').read_bytes()).hexdigest() != checkpoint['tokenizer_sha256']:
        raise ValueError('Tokenizer differs from checkpoint')
    tok = Tokenizer(path / 'tokenizer.json')
    eos_id = json.loads((path / 'metadata.json').read_text())['eos_id']
    model = GPT(ModelConfig(**checkpoint['config']['model'])).to(device)
    model.load_state_dict(checkpoint['model'])
    results = []
    for prompt in args.prompt or ['Once upon a time,', 'The little girl found', 'One day, a dog']:
        ids = tok.encode(prompt) or [eos_id]
        outputs = model.generate(torch.tensor([ids], device=device), args.max_new_tokens, args.temperature, args.top_k, eos_id)
        completion = tok.decode(outputs[0, len(ids):].tolist())
        results.append({'prompt': prompt, 'completion': completion})
        print(prompt + completion + '\n')
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps({'settings': vars(args), 'samples': results}, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
