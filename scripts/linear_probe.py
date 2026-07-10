# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
"""Standalone linear-probe evaluation for an I-JEPA checkpoint.

The official ijepa repo ships no evaluation code. This is a lightweight,
from-scratch linear probe: freeze the target encoder, extract avg-pooled
patch-token features over a stratified subset of ImageNet-1K train (to keep
this fast to run after every checkpoint) plus the full val set, then train a
single linear classifier on the cached features and report top-1 accuracy.
This is a simplification of the paper's own linear-eval protocol (which uses
the full train set and a more elaborate multi-crop procedure) chosen to keep
turnaround fast; treat resulting numbers as directionally comparable across
this project's own checkpoints, not as literal reproductions of the paper's
reported numbers.

Usage:
    python scripts/linear_probe.py \
        --checkpoint experiments/in1k_vitb16_ep600/jepa-latest.pth.tar \
        --out experiments/in1k_vitb16_ep600/linear_probe.json
"""
import argparse
import json
import os
import sys
import time
from collections import OrderedDict

import torch
import torch.nn as nn
import torchvision.transforms as transforms

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import src.models.vision_transformer as vit
from src.datasets.imagenet1k import ImageNet

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_target_encoder(checkpoint_path, model_name, patch_size, crop_size, device):
    encoder = vit.__dict__[model_name](img_size=[crop_size], patch_size=patch_size)
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    state_dict = ckpt['target_encoder']
    # weights were saved from a DistributedDataParallel-wrapped module
    stripped = OrderedDict((k.replace('module.', '', 1), v) for k, v in state_dict.items())
    msg = encoder.load_state_dict(stripped)
    print(f'loaded target_encoder from epoch {ckpt.get("epoch")}: {msg}')
    encoder.to(device)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False
    return encoder, ckpt.get('epoch')


def stratified_subset_indices(dataset, frac, seed):
    g = torch.Generator().manual_seed(seed)
    indices = []
    for class_indices in dataset.target_indices:
        class_indices = torch.as_tensor(class_indices)
        n_keep = max(1, int(round(len(class_indices) * frac)))
        perm = class_indices[torch.randperm(len(class_indices), generator=g)[:n_keep]]
        indices.extend(perm.tolist())
    return indices


@torch.no_grad()
def extract_features(encoder, loader, device):
    feats, labels = [], []
    for imgs, targets in loader:
        imgs = imgs.to(device, non_blocking=True)
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=device.startswith('cuda')):
            tokens = encoder(imgs)  # [B, N, D]
        pooled = tokens.mean(dim=1).float().cpu()
        feats.append(pooled)
        labels.append(targets)
    return torch.cat(feats), torch.cat(labels)


def train_linear_probe(train_feats, train_labels, val_feats, val_labels, num_classes, device, epochs, lr):
    dim = train_feats.shape[1]
    classifier = nn.Linear(dim, num_classes).to(device)
    train_feats = train_feats.to(device)
    train_labels = train_labels.to(device)
    val_feats = val_feats.to(device)
    val_labels = val_labels.to(device)

    # standardize features using train statistics
    mu, sigma = train_feats.mean(0, keepdim=True), train_feats.std(0, keepdim=True) + 1e-6
    train_feats = (train_feats - mu) / sigma
    val_feats = (val_feats - mu) / sigma

    opt = torch.optim.SGD(classifier.parameters(), lr=lr, momentum=0.9, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.CrossEntropyLoss()

    n = train_feats.shape[0]
    batch_size = 4096
    for epoch in range(epochs):
        perm = torch.randperm(n, device=device)
        total_loss = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            logits = classifier(train_feats[idx])
            loss = loss_fn(logits, train_labels[idx])
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(idx)
        sched.step()
        print(f'  linear-probe epoch {epoch + 1}/{epochs} loss {total_loss / n:.4f}')

    classifier.eval()
    with torch.no_grad():
        val_logits = classifier(val_feats)
        top1 = (val_logits.argmax(dim=1) == val_labels).float().mean().item()
    return top1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--model-name', default='vit_base')
    parser.add_argument('--patch-size', type=int, default=16)
    parser.add_argument('--crop-size', type=int, default=224)
    parser.add_argument('--root-path', default='/data/hkzhang/imagenet_raw')
    parser.add_argument('--image-folder', default='')
    parser.add_argument('--train-frac', type=float, default=0.1, help='fraction of the train set to use for the probe')
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--num-workers', type=int, default=16)
    parser.add_argument('--probe-epochs', type=int, default=20)
    parser.add_argument('--probe-lr', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--out', required=True)
    args = parser.parse_args()

    eval_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(args.crop_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    encoder, ckpt_epoch = load_target_encoder(
        args.checkpoint, args.model_name, args.patch_size, args.crop_size, args.device)

    t0 = time.time()
    train_full = ImageNet(root=args.root_path, image_folder=args.image_folder,
                           transform=eval_transform, train=True, copy_data=False, index_targets=True)
    subset_idx = stratified_subset_indices(train_full, args.train_frac, args.seed)
    train_subset = torch.utils.data.Subset(train_full, subset_idx)
    val_set = ImageNet(root=args.root_path, image_folder=args.image_folder,
                        transform=eval_transform, train=False, copy_data=False, index_targets=False)
    print(f'train subset: {len(train_subset)} images ({args.train_frac:.0%} of {len(train_full)}), '
          f'val: {len(val_set)} images')

    train_loader = torch.utils.data.DataLoader(
        train_subset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True)

    print('extracting train features...')
    train_feats, train_labels = extract_features(encoder, train_loader, args.device)
    print('extracting val features...')
    val_feats, val_labels = extract_features(encoder, val_loader, args.device)
    print(f'feature extraction took {time.time() - t0:.0f}s')

    num_classes = len(train_full.classes)
    top1 = train_linear_probe(
        train_feats, train_labels, val_feats, val_labels, num_classes,
        args.device, args.probe_epochs, args.probe_lr)

    result = {
        'checkpoint': args.checkpoint,
        'checkpoint_epoch': ckpt_epoch,
        'train_frac': args.train_frac,
        'num_train_images': len(train_subset),
        'num_val_images': len(val_set),
        'top1_accuracy': top1,
    }
    print(result)
    with open(args.out, 'w') as f:
        json.dump(result, f, indent=2)
    print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
