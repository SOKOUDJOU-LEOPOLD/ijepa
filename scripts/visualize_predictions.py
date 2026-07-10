# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
"""Visualize what an I-JEPA checkpoint predicts for masked regions of an image.

I-JEPA's predictor outputs representation vectors, not pixels -- there is no
pixel decoder anywhere in this codebase (the paper's own "sketch"
visualizations come from a separate generative decoder that was never
released). As a proxy, this script retrieves the *nearest-neighbor real
image patch* (by cosine similarity, in that checkpoint's own target-encoder
representation space) for each predicted target patch, and pastes it in.
This is an honest reflection of what the representation encodes, not a
learned reconstruction -- expect a patchwork/collage look, not a smooth
image.

For each sample image and each checkpoint, produces one row of:
    [original | context (visible region only) | prediction (nearest-neighbor fill)]

Usage:
    python scripts/visualize_predictions.py \
        --checkpoints experiments/in1k_vitb16_ep600/jepa-ep100.pth.tar \
                       experiments/in1k_vitb16_ep600/jepa-ep250.pth.tar \
                       experiments/in1k_vitb16_ep600/jepa-ep400.pth.tar \
        --out-dir experiments/in1k_vitb16_ep600/viz
"""
import argparse
import os
import re
import sys
from collections import OrderedDict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import src.models.vision_transformer as vit
from src.datasets.imagenet1k import ImageNet
from src.masks.multiblock import MaskCollator

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
GRID = 14   # 224 / patch_size(16)
PATCH = 16


def checkpoint_label(path):
    m = re.search(r'ep(\d+)', os.path.basename(path))
    return f'epoch {m.group(1)}' if m else os.path.basename(path)


def load_models(checkpoint_path, model_name, patch_size, crop_size, pred_depth, pred_emb_dim, device):
    ckpt = torch.load(checkpoint_path, map_location='cpu')

    def strip(sd):
        return OrderedDict((k.replace('module.', '', 1), v) for k, v in sd.items())

    encoder = vit.__dict__[model_name](img_size=[crop_size], patch_size=patch_size)
    encoder.load_state_dict(strip(ckpt['encoder']))
    target_encoder = vit.__dict__[model_name](img_size=[crop_size], patch_size=patch_size)
    target_encoder.load_state_dict(strip(ckpt['target_encoder']))
    predictor = vit.__dict__['vit_predictor'](
        num_patches=encoder.patch_embed.num_patches,
        embed_dim=encoder.embed_dim,
        predictor_embed_dim=pred_emb_dim,
        depth=pred_depth,
        num_heads=encoder.num_heads)
    predictor.load_state_dict(strip(ckpt['predictor']))

    for m in (encoder, target_encoder, predictor):
        m.to(device).eval()
        for p in m.parameters():
            p.requires_grad = False

    return encoder, predictor, target_encoder, ckpt.get('epoch')


def load_image(path, resize_crop):
    img = Image.open(path).convert('RGB')
    img = resize_crop(img)
    raw = transforms.ToTensor()(img)  # [3,224,224] in [0,1]
    norm = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)(raw.clone())
    return raw, norm


def sample_mask(seed):
    collator = MaskCollator(
        input_size=224, patch_size=PATCH,
        enc_mask_scale=(0.85, 1.0), pred_mask_scale=(0.15, 0.2),
        aspect_ratio=(0.75, 1.5), nenc=1, npred=1, min_keep=10, allow_overlap=False)
    collator._itr_counter.value = seed - 1  # .step() pre-increments
    dummy = torch.zeros(3, 224, 224)
    _, masks_enc, masks_pred = collator([(dummy, 0)])
    return masks_enc[0][0], masks_pred[0][0]  # [Ne], [Np] patch indices for the single image


def patch_pixels(raw_img, idx):
    r, c = divmod(int(idx), GRID)
    return raw_img[:, r * PATCH:(r + 1) * PATCH, c * PATCH:(c + 1) * PATCH]


def build_canvas(raw_img, fill):
    """fill: dict patch_idx -> [3,PATCH,PATCH] tensor"""
    canvas = torch.full_like(raw_img, 0.5)
    for idx, px in fill.items():
        r, c = divmod(int(idx), GRID)
        canvas[:, r * PATCH:(r + 1) * PATCH, c * PATCH:(c + 1) * PATCH] = px
    return canvas


@torch.no_grad()
def build_patch_bank(target_encoder, bank_norm_imgs, device):
    """bank_norm_imgs: [N,3,224,224] normalized. Returns [N*196, D] features."""
    feats = []
    for i in range(0, len(bank_norm_imgs), 32):
        batch = bank_norm_imgs[i:i + 32].to(device)
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=device.startswith('cuda')):
            tokens = target_encoder(batch)  # [b,196,D]
        feats.append(tokens.float().cpu())
    return torch.cat(feats, dim=0).reshape(-1, feats[0].shape[-1])  # [N*196, D]


@torch.no_grad()
def nearest_neighbor_fill(pred_repr, bank_feats, bank_patches):
    """pred_repr: [Np, D]. bank_feats: [M, D]. bank_patches: [M,3,PATCH,PATCH]."""
    q = F.normalize(pred_repr, dim=-1)
    b = F.normalize(bank_feats, dim=-1)
    sims = q @ b.T  # [Np, M]
    best = sims.argmax(dim=-1)  # [Np]
    return bank_patches[best]  # [Np,3,PATCH,PATCH]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoints', nargs='+', required=True)
    parser.add_argument('--model-name', default='vit_base')
    parser.add_argument('--patch-size', type=int, default=16)
    parser.add_argument('--crop-size', type=int, default=224)
    parser.add_argument('--pred-depth', type=int, default=6)
    parser.add_argument('--pred-emb-dim', type=int, default=384)
    parser.add_argument('--root-path', default='/data/hkzhang/imagenet_raw')
    parser.add_argument('--image-folder', default='')
    parser.add_argument('--num-samples', type=int, default=4)
    parser.add_argument('--bank-size', type=int, default=64, help='number of reference images for nearest-neighbor bank')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--out-dir', required=True)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    resize_crop = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(args.crop_size)])

    val_set = ImageNet(root=args.root_path, image_folder=args.image_folder,
                        transform=None, train=False, copy_data=False, index_targets=False)
    paths = [s[0] for s in val_set.samples]

    g = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(len(paths), generator=g).tolist()
    sample_paths = [paths[i] for i in perm[:args.num_samples]]
    bank_paths = [paths[i] for i in perm[args.num_samples:args.num_samples + args.bank_size]]

    print(f'loading {len(sample_paths)} sample images + {len(bank_paths)} bank images...')
    sample_raw, sample_norm = zip(*[load_image(p, resize_crop) for p in sample_paths])
    bank_raw, bank_norm = zip(*[load_image(p, resize_crop) for p in bank_paths])
    bank_raw = torch.stack(bank_raw)
    bank_norm = torch.stack(bank_norm)
    bank_patches_pixels = torch.stack([
        patch_pixels(bank_raw[i], idx)
        for i in range(len(bank_raw)) for idx in range(GRID * GRID)
    ])  # [N*196, 3, PATCH, PATCH]

    # fixed mask per sample image, reused across all checkpoints for a fair comparison
    sample_masks = [sample_mask(seed=args.seed * 1000 + i) for i in range(args.num_samples)]

    labels = [checkpoint_label(c) for c in args.checkpoints]
    n_rows = len(args.checkpoints)

    for si in range(args.num_samples):
        raw_img = sample_raw[si]
        norm_img = sample_norm[si]
        mask_enc_idx, mask_pred_idx = sample_masks[si]

        fig, axes = plt.subplots(n_rows, 3, figsize=(9, 3 * n_rows))
        if n_rows == 1:
            axes = axes[None, :]

        for row, (ckpt_path, label) in enumerate(zip(args.checkpoints, labels)):
            encoder, predictor, target_encoder, ckpt_epoch = load_models(
                ckpt_path, args.model_name, args.patch_size, args.crop_size,
                args.pred_depth, args.pred_emb_dim, args.device)

            bank_feats = build_patch_bank(target_encoder, bank_norm, args.device)

            with torch.no_grad():
                x = norm_img.unsqueeze(0).to(args.device)
                me = mask_enc_idx.unsqueeze(0).to(args.device)
                mp = mask_pred_idx.unsqueeze(0).to(args.device)
                with torch.autocast('cuda', dtype=torch.bfloat16, enabled=args.device.startswith('cuda')):
                    ctx_repr = encoder(x, masks=[me])
                    pred_repr = predictor(ctx_repr, masks_x=[me], masks=[mp])
                pred_repr = pred_repr[0].float().cpu()  # [Np, D]

            nn_patches = nearest_neighbor_fill(pred_repr, bank_feats, bank_patches_pixels)

            context_fill = {idx: patch_pixels(raw_img, idx) for idx in mask_enc_idx.tolist()}
            context_canvas = build_canvas(raw_img, context_fill)

            pred_fill = dict(context_fill)
            pred_fill.update({idx: nn_patches[j] for j, idx in enumerate(mask_pred_idx.tolist())})
            pred_canvas = build_canvas(raw_img, pred_fill)

            for col, (img, title) in enumerate([
                (raw_img, 'original'),
                (context_canvas, 'context'),
                (pred_canvas, 'prediction (NN)'),
            ]):
                ax = axes[row, col]
                ax.imshow(img.permute(1, 2, 0).numpy())
                ax.set_xticks([])
                ax.set_yticks([])
                if col == 0:
                    ax.set_ylabel(label, fontsize=11)
                if row == 0:
                    ax.set_title(title, fontsize=11)

            del encoder, predictor, target_encoder, bank_feats
            if args.device.startswith('cuda'):
                torch.cuda.empty_cache()

        fig.tight_layout()
        out_path = os.path.join(args.out_dir, f'sample_{si}.png')
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f'wrote {out_path}')


if __name__ == '__main__':
    main()
