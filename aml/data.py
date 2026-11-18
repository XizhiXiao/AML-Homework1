"""Stream documents, split by content hash, encode to disk, preserve byte counts."""
import argparse
import hashlib
import json
from itertools import islice
from pathlib import Path
import shutil
import numpy as np
from .bpe import ByteBPE


class Tokenizer:
    def __init__(self, path):
        self.path = Path(path)
        obj = json.loads(self.path.read_text(encoding='utf-8'))
        if obj.get('type') == 'aml_byte_bpe':
            self.impl = ByteBPE.load(path)
            self.custom = True
            self.eos_id, self.vocab_size = self.impl.eos_id, self.impl.vocab_size
        else:
            from tokenizers import Tokenizer as HFTokenizer
            self.impl = HFTokenizer.from_file(str(path))
            self.impl.no_truncation()
            self.impl.no_padding()
            self.custom = False
            self.vocab_size = self.impl.get_vocab_size(with_added_tokens=True)
            self.eos_id = next((self.impl.token_to_id(s) for s in ['<|endoftext|>', '</s>', '<eos>', '<|end_of_text|>'] if self.impl.token_to_id(s) is not None), None)

    def encode(self, text):
        return self.impl.encode(text) if self.custom else self.impl.encode(text, add_special_tokens=False).ids

    def decode(self, ids):
        return self.impl.decode(ids)


def documents(args):
    if args.input:
        with open(args.input, encoding='utf-8') as stream:
            if args.input_format == 'tinystories':
                lines = []
                for line in stream:
                    if line.strip() == '<|endoftext|>':
                        if lines:
                            yield ''.join(lines)
                        lines = []
                    else:
                        lines.append(line)
                if lines:
                    yield ''.join(lines)
            else:
                for line in stream:
                    if line.strip():
                        yield json.loads(line)[args.text_field]
    else:
        from datasets import load_dataset
        kwargs = {'split': args.split, 'streaming': True}
        if args.dataset_config:
            kwargs['name'] = args.dataset_config
        if args.revision:
            kwargs['revision'] = args.revision
        for row in load_dataset(args.dataset, **kwargs):
            yield row[args.text_field]


def encoded_documents(args, tokenizer):
    source = documents(args)
    if args.max_documents:
        source = islice(source, args.max_documents)
    while True:
        batch = list(islice(source, 256))
        if not batch:
            return
        batch = [text for text in batch if isinstance(text, str) and text]
        if tokenizer.custom:
            encoded = [tokenizer.encode(text) for text in batch]
        else:
            encoded = [item.ids for item in tokenizer.impl.encode_batch(batch, add_special_tokens=False)]
        yield from zip(batch, encoded)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument('--input', help='Local JSONL documents')
    source.add_argument('--dataset', help='HF dataset ID')
    p.add_argument('--input-format', choices=['jsonl', 'tinystories'], default='jsonl')
    p.add_argument('--dataset-config')
    p.add_argument('--split', default='train')
    p.add_argument('--revision')
    p.add_argument('--text-field', default='text')
    p.add_argument('--tokenizer', help='Local tokenizer JSON; defaults to course tokenizer')
    p.add_argument('--tokenizer-revision', default='main')
    p.add_argument('--eos-id', type=int)
    p.add_argument('--output', required=True)
    p.add_argument('--val-fraction', type=float, default=0.01)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--max-documents', type=int, default=0)
    p.add_argument('--max-train-tokens', type=int, default=0, help='Stop after full document crossing budget; 0 = all')
    p.add_argument('--bpe-sample-documents', type=int, default=1000)
    args = p.parse_args()
    if not 0 < args.val_fraction < 1:
        p.error('val-fraction must be between 0 and 1')
    out = Path(args.output)
    if out.exists() and any(out.iterdir()):
        p.error('Output directory must be empty to avoid overwriting prepared data')
    out.mkdir(parents=True, exist_ok=True)
    tokenizer_path = args.tokenizer
    if not tokenizer_path:
        from huggingface_hub import hf_hub_download
        tokenizer_path = hf_hub_download('flexitok/bpe_fw_edu_16000', 'tokenizer.json', revision=args.tokenizer_revision)
    shutil.copyfile(tokenizer_path, out / 'tokenizer.json')
    tok = Tokenizer(out / 'tokenizer.json')
    eos = args.eos_id if args.eos_id is not None else tok.eos_id
    if eos is None or not 0 <= eos < tok.vocab_size:
        raise ValueError('Specify an existing valid --eos-id; no token is silently added')
    dtype = np.dtype('<u2' if tok.vocab_size <= 65536 else '<u4')
    stats = {s: {'documents': 0, 'tokens': 0, 'bytes': 0} for s in ('train', 'val')}
    streams = {s: open(out / (s + '.bin'), 'wb', buffering=8 * 1024 * 1024) for s in stats}
    try:
        with open(out / 'val_documents.jsonl', 'w') as val_index, open(out / 'bpe_sample.jsonl', 'w', encoding='utf-8') as sample:
            for i, (text, content_ids) in enumerate(encoded_documents(args, tok)):
                if args.max_documents and i >= args.max_documents:
                    break
                if not isinstance(text, str) or not text:
                    continue
                raw = text.encode('utf-8')
                digest = hashlib.sha256(str(args.seed).encode() + b'\0' + raw).digest()
                split = 'val' if int.from_bytes(digest[:8], 'big') / 2**64 < args.val_fraction else 'train'
                ids = content_ids + [eos]
                if min(ids) < 0 or max(ids) >= tok.vocab_size:
                    raise ValueError('Tokenizer produced out-of-range token IDs')
                if split == 'val':
                    val_index.write(json.dumps({'offset': stats[split]['tokens'], 'length': len(ids), 'bytes': len(raw)}) + '\n')
                elif stats['train']['documents'] < args.bpe_sample_documents:
                    sample.write(json.dumps({'text': text}, ensure_ascii=False) + '\n')
                streams[split].write(np.asarray(ids, dtype=dtype).tobytes())
                stats[split]['documents'] += 1
                stats[split]['tokens'] += len(ids)
                stats[split]['bytes'] += len(raw)
                if (i + 1) % 10000 == 0:
                    print(json.dumps({'seen_documents': i + 1, **stats}), flush=True)
                if args.max_train_tokens and stats['train']['tokens'] >= args.max_train_tokens:
                    break
    finally:
        for stream in streams.values():
            stream.close()
    if any(v['tokens'] == 0 for v in stats.values()):
        raise ValueError('Empty train/val split. Use more documents or larger val-fraction; remove incomplete output before retrying.')
    meta = {'format_version': 1, 'dtype': dtype.str, 'vocab_size': tok.vocab_size, 'eos_id': eos,
            'tokenizer_sha256': hashlib.sha256((out / 'tokenizer.json').read_bytes()).hexdigest(),
            'splits': stats, 'source': vars(args),
            'split_method': 'sha256(seed + document UTF-8), identical documents always share split',
            'validation_policy': 'Each document starts with EOS context; predict all content tokens plus final EOS. BPB includes EOS NLL, divided by original UTF-8 bytes.'}
    (out / 'metadata.json').write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(meta, indent=2, ensure_ascii=False))


class Corpus:
    def __init__(self, directory, context_length):
        self.path = Path(directory)
        self.meta = json.loads((self.path / 'metadata.json').read_text())
        self.context = context_length
        self.train = np.memmap(self.path / 'train.bin', dtype=self.meta['dtype'], mode='r')
        self.val = np.memmap(self.path / 'val.bin', dtype=self.meta['dtype'], mode='r')
        if len(self.train) <= context_length:
            raise ValueError('Training corpus must contain more tokens than context length')

    def batch(self, size, generator, device):
        import torch
        starts = torch.randint(len(self.train) - self.context, (size,), generator=generator).tolist()
        chunks = np.stack([self.train[i:i+self.context+1] for i in starts]).astype(np.int64)
        tensor = torch.from_numpy(chunks).to(device)
        return tensor[:, :-1], tensor[:, 1:]

    def validation_chunks(self, max_documents=0):
        # Whole documents keep NLL and original byte denominators exactly aligned.
        with open(self.path / 'val_documents.jsonl') as stream:
            for i, line in enumerate(stream):
                if max_documents and i >= max_documents:
                    break
                record = json.loads(line)
                start, length = record['offset'], record['length']
                ids = self.val[start:start+length].astype(np.int64)
                for pos in range(0, length, self.context):
                    y = ids[pos:pos+self.context]
                    previous = self.meta['eos_id'] if pos == 0 else ids[pos-1]
                    x = np.concatenate([[previous], y[:-1]])
                    yield x, y, record['bytes'] if pos == 0 else 0


if __name__ == '__main__':
    main()
