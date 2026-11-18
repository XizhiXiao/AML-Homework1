"""Byte-level BPE. No library BPE trainer or encoder is used here.

This deliberately readable implementation rescans unique documents per merge;
it is intended for a bounded training sample, not a multi-billion-token corpus.
"""
import argparse
import base64
from collections import Counter
import json
from pathlib import Path


class ByteBPE:
    def __init__(self, merges=()):
        self.merges = [tuple(pair) for pair in merges]
        self.vocab = {i: bytes([i]) for i in range(256)}
        for i, (a, b) in enumerate(self.merges, 256):
            if a not in self.vocab or b not in self.vocab:
                raise ValueError('Invalid merge dependency')
            self.vocab[i] = self.vocab[a] + self.vocab[b]
        self.eos_id = len(self.vocab)
        self.vocab_size = self.eos_id + 1

    @staticmethod
    def merge(ids, pair, new_id):
        out, i = [], 0
        while i < len(ids):
            if i + 1 < len(ids) and (ids[i], ids[i + 1]) == pair:
                out.append(new_id)
                i += 2
            else:
                out.append(ids[i])
                i += 1
        return out

    @classmethod
    def train(cls, texts, vocab_size=1024):
        if vocab_size < 257:
            raise ValueError('Need 256 byte tokens and one EOS token')
        sequences = Counter(tuple(text.encode('utf-8')) for text in texts if text)
        merges = []
        for new_id in range(256, vocab_size - 1):
            counts = Counter()
            for ids, frequency in sequences.items():
                for pair in zip(ids, ids[1:]):
                    counts[pair] += frequency
            if not counts:
                break
            # Deterministic tie-breaking makes saved artifacts reproducible.
            pair = min(counts, key=lambda p: (-counts[p], p))
            updated = Counter()
            for ids, frequency in sequences.items():
                updated[tuple(cls.merge(ids, pair, new_id))] += frequency
            sequences = updated
            merges.append(pair)
        return cls(merges)

    def encode(self, text):
        ids = list(text.encode('utf-8'))
        for new_id, pair in enumerate(self.merges, 256):
            ids = self.merge(ids, pair, new_id)
        return ids

    def decode(self, ids):
        return b''.join(self.vocab[i] for i in ids if i != self.eos_id).decode('utf-8', errors='replace')

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps({
            'type': 'aml_byte_bpe', 'version': 1, 'merges': self.merges,
            'eos_id': self.eos_id, 'vocab_size': self.vocab_size,
            'vocab_base64': {str(i): base64.b64encode(b).decode() for i, b in self.vocab.items()}
        }, indent=2), encoding='utf-8')

    @classmethod
    def load(cls, path):
        obj = json.loads(Path(path).read_text(encoding='utf-8'))
        if obj.get('type') != 'aml_byte_bpe':
            raise ValueError('Not an AML BPE tokenizer')
        return cls(obj['merges'])


def main():
    parser = argparse.ArgumentParser(description='Train self-written byte BPE on a JSONL sample')
    parser.add_argument('--input', required=True, help='JSONL, one document per row')
    parser.add_argument('--text-field', default='text')
    parser.add_argument('--output', required=True)
    parser.add_argument('--vocab-size', type=int, default=1024)
    parser.add_argument('--max-documents', type=int, default=1000)
    parser.add_argument('--max-bytes', type=int, default=500000)
    args = parser.parse_args()
    texts, size = [], 0
    with open(args.input, encoding='utf-8') as stream:
        for line in stream:
            if not line.strip():
                continue
            text = json.loads(line)[args.text_field]
            if len(texts) >= args.max_documents or size >= args.max_bytes:
                break
            text = text.encode('utf-8')[:args.max_bytes-size].decode('utf-8', errors='ignore')
            texts.append(text)
            size += len(text.encode('utf-8'))
    if not texts or not size:
        raise ValueError('Empty BPE training sample')
    tokenizer = ByteBPE.train(texts, args.vocab_size)
    for text in texts:
        assert tokenizer.decode(tokenizer.encode(text)) == text
    tokenizer.save(args.output)
    tokens = sum(len(tokenizer.encode(t)) for t in texts)
    report = {'documents': len(texts), 'utf8_bytes': size, 'tokens': tokens,
              'bytes_per_token': size / max(tokens, 1), 'vocab_size': tokenizer.vocab_size,
              'merges': len(tokenizer.merges), 'roundtrip_passed': True}
    Path(args.output + '.report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
