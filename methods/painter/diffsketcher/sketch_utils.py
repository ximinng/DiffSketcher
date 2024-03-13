# -*- coding: utf-8 -*-
# Copyright (c) XiMing Xing. All rights reserved.
# Author: XiMing Xing
# Description:
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import torch
from torchvision.utils import make_grid


def plt_batch(
        photos: torch.Tensor,
        sketch: torch.Tensor,
        step: int,
        prompt: str,
        save_path: str,
        name: str,
        dpi: int = 300
):
    if photos.shape != sketch.shape:
        raise ValueError("photos and sketch must have the same dimensions")

    plt.figure()
    plt.subplot(1, 2, 1)  # nrows=1, ncols=2, index=1
    grid = make_grid(photos, normalize=True, pad_value=2)
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    plt.imshow(ndarr)
    plt.axis("off")
    plt.title("Generated sample")

    plt.subplot(1, 2, 2)  # nrows=1, ncols=2, index=2
    grid = make_grid(sketch, normalize=False, pad_value=2)
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    plt.imshow(ndarr)
    plt.axis("off")
    plt.title(f"Rendering result - {step} steps")

    plt.suptitle(insert_newline(prompt), fontsize=10)

    plt.tight_layout()
    plt.savefig(f"{save_path}/{name}.png", dpi=dpi)
    plt.close()


def plt_triplet(
        photos: torch.Tensor,
        sketch: torch.Tensor,
        style: torch.Tensor,
        step: int,
        prompt: str,
        save_path: str,
        name: str,
        dpi: int = 300
):
    if photos.shape != sketch.shape:
        raise ValueError("photos and sketch must have the same dimensions")

    plt.figure()
    plt.subplot(1, 3, 1)  # nrows=1, ncols=3, index=1
    grid = make_grid(photos, normalize=True, pad_value=2)
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    plt.imshow(ndarr)
    plt.axis("off")
    plt.title("Generated sample")

    plt.subplot(1, 3, 2)  # nrows=1, ncols=3, index=2
    # style = (style + 1) / 2
    grid = make_grid(style, normalize=False, pad_value=2)
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    plt.imshow(ndarr)
    plt.axis("off")
    plt.title(f"Style")

    plt.subplot(1, 3, 3)  # nrows=1, ncols=3, index=2
    # sketch = (sketch + 1) / 2
    grid = make_grid(sketch, normalize=False, pad_value=2)
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    plt.imshow(ndarr)
    plt.axis("off")
    plt.title(f"Rendering result - {step} steps")

    plt.suptitle(insert_newline(prompt), fontsize=10)

    plt.tight_layout()
    plt.savefig(f"{save_path}/{name}.png", dpi=dpi)
    plt.close()


def insert_newline(string, point=9):
    # split by blank
    words = string.split()
    if len(words) <= point:
        return string

    word_chunks = [words[i:i + point] for i in range(0, len(words), point)]
    new_string = "\n".join(" ".join(chunk) for chunk in word_chunks)
    return new_string


def log_tensor_img(inputs, output_dir, output_prefix="input", norm=False, dpi=300):
    grid = make_grid(inputs, normalize=norm, pad_value=2)
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    plt.imshow(ndarr)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{output_prefix}.png", dpi=dpi, bbox_inches='tight')
    plt.close()


def plt_tensor_img(tensor, title, save_path, name, dpi=500):
    grid = make_grid(tensor, normalize=True, pad_value=2)
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    plt.imshow(ndarr)
    plt.axis("off")
    plt.title(f"{title}")
    plt.savefig(f"{save_path}/{name}.png", dpi=dpi)
    plt.close()


def save_tensor_img(tensor, save_path, name, dpi=500):
    grid = make_grid(tensor, normalize=True, pad_value=2)
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    plt.imshow(ndarr)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(f"{save_path}/{name}.png", dpi=dpi)
    plt.close()


def plt_attn(attn, threshold_map, inputs, inds, output_path):
    # currently supports one image (and not a batch)
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 3, 1)
    main_im = make_grid(inputs, normalize=True, pad_value=2)
    main_im = np.transpose(main_im.cpu().numpy(), (1, 2, 0))
    plt.imshow(main_im, interpolation='nearest')
    plt.scatter(inds[:, 1], inds[:, 0], s=10, c='red', marker='o')
    plt.title("input img")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(attn, interpolation='nearest', vmin=0, vmax=1)
    plt.title("attn map")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    threshold_map_ = (threshold_map - threshold_map.min()) / \
                     (threshold_map.max() - threshold_map.min())
    plt.imshow(np.nan_to_num(threshold_map_), interpolation='nearest', vmin=0, vmax=1)
    plt.title("prob softmax")
    plt.scatter(inds[:, 1], inds[:, 0], s=10, c='red', marker='o')
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def fix_image_scale(im):
    im_np = np.array(im) / 255
    height, width = im_np.shape[0], im_np.shape[1]
    max_len = max(height, width) + 20
    new_background = np.ones((max_len, max_len, 3))
    y, x = max_len // 2 - height // 2, max_len // 2 - width // 2
    new_background[y: y + height, x: x + width] = im_np
    new_background = (new_background / new_background.max()
                      * 255).astype(np.uint8)
    new_im = Image.fromarray(new_background)
    return new_im
