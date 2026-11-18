import contextlib
import io
import json
from pathlib import Path
import sys
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import torch
from aml.bpe import ByteBPE
from aml import data, train, evaluate, generate, plot
from aml.metrics import evaluate as evaluate_model
from aml.model import GPT, ModelConfig


class PipelineTest(unittest.TestCase):
    def call(self, module, args):
        with patch.object(sys, 'argv', ['test'] + list(map(str, args))), contextlib.redirect_stdout(io.StringIO()):
            module.main()

    def test_end_to_end_and_exact_resume(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tok = root / 'byte.json'; ByteBPE().save(tok)
            source = root / 'input.jsonl'
            source.write_text('\n'.join(json.dumps({'text': f'Story {i}: a cat sees a dog. 中文🙂.'}) for i in range(100)), encoding='utf-8')
            corpus = root / 'data'
            subprocess.run([sys.executable, '-m', 'aml.data', '--input', str(source), '--tokenizer', str(tok), '--output', str(corpus), '--val-fraction', '0.2'], check=True, capture_output=True, text=True)
            raw = root / 'stories.txt'
            raw.write_text('first\n<|endoftext|>\nsecond\n<|endoftext|>\n', encoding='utf-8')
            import argparse
            source_args = argparse.Namespace(input=str(raw), input_format='tinystories')
            self.assertEqual(list(data.documents(source_args)), ['first\n', 'second\n'])
            cfg = json.loads(Path('configs/smoke.json').read_text())
            cfg['model']['dropout'] = 0.1  # Ensure RNG restoration matters.
            cfg_path = root / 'config.json'; cfg_path.write_text(json.dumps(cfg))
            full, resumed = root / 'full', root / 'resumed'
            self.call(train, ['--config', cfg_path, '--data', corpus, '--output', full])
            self.call(train, ['--config', cfg_path, '--data', corpus, '--output', resumed, '--stop-after', 6])
            self.call(train, ['--config', cfg_path, '--data', corpus, '--output', resumed, '--resume', resumed / 'last.pt'])
            a = train.load_checkpoint(full / 'last.pt'); b = train.load_checkpoint(resumed / 'last.pt')
            self.assertEqual(a['step'], 12)
            self.assertEqual(a['tokens_seen'], b['tokens_seen'])
            for name in a['model']:
                self.assertTrue(torch.equal(a['model'][name], b['model'][name]), name)
            result_path = root / 'metrics.json'
            self.call(evaluate, ['--checkpoint', full / 'best.pt', '--data', corpus, '--output', result_path, '--device', 'cpu'])
            result = json.loads(result_path.read_text())
            meta = json.loads((corpus / 'metadata.json').read_text())
            self.assertEqual(result['tokens'], meta['splits']['val']['tokens'])
            self.assertEqual(result['bytes'], meta['splits']['val']['bytes'])
            self.call(generate, ['--checkpoint', full / 'best.pt', '--data', corpus, '--output', root / 'samples.json', '--max-new-tokens', 4, '--device', 'cpu'])
            self.assertEqual(len(json.loads((root / 'samples.json').read_text())['samples']), 3)
            self.call(plot, ['--run', full])
            self.assertGreater((full / 'curves.png').stat().st_size, 1000)
            self.assertGreater((full / 'curves.pdf').stat().st_size, 1000)
            # Padding/batch boundaries must not change aggregated evaluation metrics.
            model = GPT(ModelConfig(**cfg['model'])); model.load_state_dict(a['model'])
            ds = data.Corpus(corpus, 32)
            one = evaluate_model(model, ds, torch.device('cpu'), contextlib.nullcontext, 1, 0)
            many = evaluate_model(model, ds, torch.device('cpu'), contextlib.nullcontext, 7, 0)
            self.assertAlmostEqual(one['loss'], many['loss'], places=5)
            self.assertAlmostEqual(one['bpb'], many['bpb'], places=5)


if __name__ == '__main__':
    unittest.main()
