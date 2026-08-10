#!/usr/bin/env python
"""Training entry point. Run with: python -m scripts.train --data-dir ...

The actual argument parsing lives in ``stylegan2_ada.cli`` so the same code
backs both this wrapper and the installed ``stylegan2-train`` command.
"""

from stylegan2_ada.cli import train

if __name__ == "__main__":
    train()
