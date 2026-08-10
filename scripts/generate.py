#!/usr/bin/env python
"""Generation entry point. Run with: python -m scripts.generate --checkpoint-path ...

The actual argument parsing lives in ``stylegan2_ada.cli`` so the same code
backs both this wrapper and the installed ``stylegan2-generate`` command.
"""

from stylegan2_ada.cli import generate

if __name__ == "__main__":
    generate()
