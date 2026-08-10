"""Device selection and GPU memory utilities."""

import gc

import torch

# Global device used across the package. Import as: from .device import DEVICE
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def gpu_report() -> None:
    """Print the active device and, if CUDA is available, a VRAM summary.

    Also clears the CUDA cache and resets peak-memory statistics so that
    reported numbers reflect the current run.
    """
    print(f"Device: {DEVICE}")

    if DEVICE.type != "cuda":
        return

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"VRAM total: {total_mem:.2f} GB")
    print(f"VRAM reserved: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")
    print(f"VRAM in use: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
