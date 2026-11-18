import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description='Plot train/validation loss, perplexity and BPB')
    p.add_argument('--run', required=True)
    args = p.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    run = Path(args.run)
    records = [json.loads(line) for line in (run / 'metrics.jsonl').read_text().splitlines() if line]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, key in zip(axes, ['loss', 'ppl', 'bpb']):
        for kind in ['train', 'val']:
            rows = [r for r in records if r['kind'] == kind and key in r]
            if rows:
                ax.plot([r['tokens_seen'] for r in rows], [r[key] for r in rows], label=kind)
        ax.set(xlabel='Training tokens', ylabel=key)
        ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout()
    fig.savefig(run / 'curves.png', dpi=180)
    fig.savefig(run / 'curves.pdf')
    print(run / 'curves.png')


if __name__ == '__main__':
    main()
