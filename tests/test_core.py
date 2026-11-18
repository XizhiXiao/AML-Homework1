import math
import tempfile
import unittest
from pathlib import Path
import torch
from aml.bpe import ByteBPE
from aml.model import GPT, ModelConfig
from aml.metrics import metrics
from aml.train import learning_rate


class CoreTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(12)

    def test_bpe_order_and_roundtrip(self):
        texts = ['the cat the car', '中文，🙂\n\t café', 'aaaaaaa', '']
        tok = ByteBPE.train(texts, 280)
        self.assertGreater(len(tok.merges), 0)
        for text in texts + ['never seen: 🐕\x00']:
            self.assertEqual(tok.decode(tok.encode(text)), text)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'bpe.json'
            tok.save(path)
            loaded = ByteBPE.load(path)
            self.assertEqual(loaded.encode(texts[0]), tok.encode(texts[0]))
        self.assertEqual(ByteBPE.merge([1, 1, 1], (1, 1), 256), [256, 1])

    def model(self):
        return GPT(ModelConfig(vocab_size=32, context_length=8, d_model=32, n_layers=2, n_heads=4, d_ff=64))

    def test_future_cannot_affect_prefix(self):
        model = self.model().eval()
        a = torch.randint(0, 32, (2, 8)); b = a.clone(); b[:, 4:] = (b[:, 4:] + 1) % 32
        self.assertTrue(torch.allclose(model(a)[0][:, :4], model(b)[0][:, :4], atol=1e-6))

    def test_overfit_and_checkpoint(self):
        model = self.model()
        x = torch.tensor([[1, 2, 3, 4, 1, 2, 3, 4]])
        y = torch.tensor([[2, 3, 4, 1, 2, 3, 4, 1]])
        opt = torch.optim.AdamW(model.parameters(), lr=0.01)
        first = model(x, y)[1].item()
        for _ in range(35):
            opt.zero_grad(); loss = model(x, y)[1]; loss.backward(); opt.step()
        self.assertLess(model(x, y)[1].item(), first * 0.2)
        clone = self.model(); clone.load_state_dict(model.state_dict())
        self.assertTrue(torch.equal(model(x)[0], clone(x)[0]))
        self.assertIs(model.head.weight, model.token_embedding.weight)
        self.assertEqual(model.generate(x, 3, temperature=0).shape, (1, 11))

    def test_metric_denominators(self):
        result = metrics(6 * math.log(2), 3, 6)
        self.assertAlmostEqual(result['ppl'], 4)
        self.assertAlmostEqual(result['bpb'], 1)

    def test_lr_endpoints(self):
        cfg = {'warmup_steps': 2, 'max_steps': 10, 'learning_rate': 0.001, 'min_lr_ratio': 0.1}
        self.assertAlmostEqual(learning_rate(0, cfg), 0.0005)
        self.assertAlmostEqual(learning_rate(2, cfg), 0.001)
        self.assertAlmostEqual(learning_rate(9, cfg), 0.0001)


if __name__ == '__main__':
    unittest.main()
