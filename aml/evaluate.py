import argparse
import json
from pathlib import Path
from .data import Corpus
from .metrics import evaluate
from .model import GPT, ModelConfig
from .train import amp_context, load_checkpoint, select_device


def main():
    p = argparse.ArgumentParser(description='Evaluate complete validation documents; default: entire validation set')
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--data', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--device', default='auto')
    p.add_argument('--precision', default='fp32', choices=['fp32', 'bf16', 'fp16'])
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--max-documents', type=int, default=0)
    args = p.parse_args()
    device = select_device(args.device)
    ckpt = load_checkpoint(args.checkpoint)
    corpus = Corpus(args.data, ckpt['config']['model']['context_length'])
    if corpus.meta['tokenizer_sha256'] != ckpt['tokenizer_sha256']:
        raise ValueError('Tokenizer differs from training')
    model = GPT(ModelConfig(**ckpt['config']['model'])).to(device)
    model.load_state_dict(ckpt['model'])
    result = evaluate(model, corpus, device, amp_context(device, args.precision), args.batch_size, args.max_documents)
    result.update({'checkpoint': str(Path(args.checkpoint).resolve()), 'step': ckpt['step'],
                   'max_documents': args.max_documents, 'policy': corpus.meta['validation_policy']})
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
