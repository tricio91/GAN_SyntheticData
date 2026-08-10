"""Inference: generate synthetic images from a trained checkpoint."""

from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from .device import DEVICE
from .networks import GeneratorStyleGAN2


def generate_samples(
    checkpoint_path: str,
    output_dir: str,
    num_samples: int = 100,
    class_idx: int = 0,
    truncation_psi: float = 1.0,
    batch_size: int = 16,
    seed: Optional[int] = None,
    show_preview: bool = True,
    preview_count: int = 20,
    preview_cols: int = 5,
) -> List[str]:
    """Load EMA weights and generate ``num_samples`` images with truncation.

    Images are saved as PNG files named ``<class>_<index>.png``. Returns the
    list of written file paths.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    print(f"Loading model: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=DEVICE)
    config = ckpt.get("config", {})
    z_dim = config.get("z_dim", 512)
    w_dim = config.get("w_dim", 512)
    img_size = config.get("img_size", 256)
    class_names = ckpt.get("class_names", ["class_0"])
    num_classes = len(class_names)

    if class_idx >= num_classes:
        print(f"  class_idx={class_idx} out of range, using 0")
        class_idx = 0

    class_name = class_names[class_idx]
    print(f"  Class: {class_name} | Truncation: {truncation_psi}")

    generator = GeneratorStyleGAN2(z_dim, w_dim, num_classes, img_size).to(DEVICE)
    if "generator_ema" in ckpt:
        generator.load_state_dict(ckpt["generator_ema"])
        print("  Using EMA weights")
    else:
        generator.load_state_dict(ckpt["generator"])
        print("  Using raw weights (no EMA)")
    generator.eval()

    print(f"  Generating {num_samples} images...")
    generated_paths = []
    num_generated = 0

    with torch.no_grad():
        pbar = tqdm(total=num_samples, desc="Generating")
        while num_generated < num_samples:
            curr_batch = min(batch_size, num_samples - num_generated)
            z = torch.randn(curr_batch, z_dim, device=DEVICE)
            labels = torch.full((curr_batch,), class_idx,
                                dtype=torch.long, device=DEVICE)

            fake_imgs = generator(z, labels, truncation_psi=truncation_psi)
            fake_imgs = ((fake_imgs + 1) / 2 * 255).clamp(0, 255).to(torch.uint8)

            for i in range(curr_batch):
                img_idx = num_generated + i
                img_array = fake_imgs[i].cpu().permute(1, 2, 0).numpy()
                img_pil = Image.fromarray(img_array)
                filepath = output_path / f"{class_name}_{img_idx:05d}.png"
                img_pil.save(filepath)
                generated_paths.append(str(filepath))

            num_generated += curr_batch
            pbar.update(curr_batch)
        pbar.close()

    print(f"  {num_samples} images saved to {output_dir}")

    if show_preview:
        _preview_grid(generated_paths, class_name, truncation_psi,
                      num_samples, preview_count, preview_cols)

    return generated_paths


def _preview_grid(generated_paths, class_name, truncation_psi,
                  num_samples, preview_count, preview_cols):
    """Display a grid preview of the generated images."""
    n_show = min(preview_count, num_samples)
    n_cols = preview_cols
    n_rows = (n_show + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 2.5, n_rows * 2.5))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    for i in range(n_rows):
        for j in range(n_cols):
            idx = i * n_cols + j
            ax = axes[i, j]
            if idx < n_show:
                img = Image.open(generated_paths[idx])
                ax.imshow(img)
                ax.set_title(f"#{idx}", fontsize=8)
            ax.axis("off")

    plt.suptitle(
        f"Class: {class_name} | psi={truncation_psi} | N={num_samples}",
        fontsize=11,
    )
    plt.tight_layout()
    plt.show()
