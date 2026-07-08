# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
"""Turn an I-JEPA training CSV log (<tag>_r<rank>.csv) into a loss-curve PNG
and a cleaned per-iteration CSV.

Usage:
    python scripts/plot_loss.py --log experiments/in1k_vitb16_ep600/jepa_r0.csv \
        --out experiments/in1k_vitb16_ep600/loss_curve
"""
import argparse
import csv
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def read_log(path):
    rows = []
    with open(path, newline='') as f:
        reader = csv.reader(f)
        for line in reader:
            if len(line) < 6:
                continue
            epoch, itr, loss, mask_a, mask_b, time_ms = line[:6]
            rows.append({
                'epoch': int(epoch),
                'itr': int(itr),
                'loss': float(loss),
                'mask_a': float(mask_a),
                'mask_b': float(mask_b),
                'time_ms': int(time_ms),
            })
    return rows


def write_csv(rows, out_csv):
    with open(out_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['epoch', 'itr', 'loss', 'mask_a', 'mask_b', 'time_ms'])
        writer.writeheader()
        writer.writerows(rows)


def plot(rows, out_png, title):
    global_step = list(range(len(rows)))
    loss = [r['loss'] for r in rows]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(global_step, loss, linewidth=0.8)
    ax.set_xlabel('iteration')
    ax.set_ylabel('loss')
    ax.set_title(title)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', required=True, help='path to <tag>_r<rank>.csv produced by src/train.py')
    parser.add_argument('--out', required=True, help='output path prefix (writes <out>.png and <out>.csv)')
    parser.add_argument('--title', default=None, help='plot title (defaults to the log filename)')
    args = parser.parse_args()

    rows = read_log(args.log)
    if not rows:
        raise SystemExit(f'no rows found in {args.log}')

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    write_csv(rows, args.out + '.csv')
    plot(rows, args.out + '.png', args.title or os.path.basename(args.log))
    print(f'wrote {args.out}.csv and {args.out}.png ({len(rows)} logged iterations)')


if __name__ == '__main__':
    main()
