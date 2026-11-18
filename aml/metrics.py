import math
import numpy as np
import torch
from torch.nn import functional as F


def metrics(nll, tokens, byte_count):
    if tokens <= 0 or byte_count <= 0:
        raise ValueError('Evaluation requires tokens and UTF-8 bytes')
    loss = nll / tokens
    return {'loss': loss, 'ppl': math.exp(loss) if loss < 700 else float('inf'),
            'bpb': nll / (math.log(2) * byte_count), 'tokens': tokens, 'bytes': byte_count}


@torch.no_grad()
def evaluate(model, corpus, device, autocast, batch_size=4, max_documents=128):
    if batch_size <= 0 or max_documents < 0:
        raise ValueError("batch_size must be positive and max_documents nonnegative")
    was_training = model.training
    model.eval()
    total_nll, total_tokens, total_bytes = 0.0, 0, 0
    pending = []

    def consume(chunks):
        max_len = max(len(x) for x, _, _ in chunks)
        x = np.zeros((len(chunks), max_len), dtype=np.int64)
        y = np.full_like(x, -100)
        for i, (inputs, targets, _) in enumerate(chunks):
            x[i, :len(inputs)], y[i, :len(targets)] = inputs, targets
        with autocast():
            logits = model(torch.from_numpy(x).to(device))[0]
        nll = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), torch.from_numpy(y).to(device).reshape(-1), reduction='sum', ignore_index=-100)
        return nll.item(), sum(len(t) for _, t, _ in chunks), sum(b for _, _, b in chunks)

    try:
        for chunk in corpus.validation_chunks(max_documents):
            pending.append(chunk)
            if len(pending) == batch_size:
                nll, tokens, byte_count = consume(pending)
                total_nll += nll; total_tokens += tokens; total_bytes += byte_count
                pending = []
        if pending:
            nll, tokens, byte_count = consume(pending)
            total_nll += nll; total_tokens += tokens; total_bytes += byte_count
    finally:
        model.train(was_training)
    return metrics(total_nll, total_tokens, total_bytes)
