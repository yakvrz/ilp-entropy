"""Entry point for running ilp_entropy as a module.

This allows the package to be run as:
    python -m ilp_entropy [args...]
"""

from .cli import main

if __name__ == "__main__":
    exit(main())
