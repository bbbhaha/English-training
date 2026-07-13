#!/usr/bin/env python
"""Convert English text to word-level and phone-level ARPAbet mappings."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pronunciation.g2p import text_to_phones, write_g2p_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Run G2P for an English target text.")
    parser.add_argument("--text", required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/demo/g2p.json")
    args = parser.parse_args()
    result = text_to_phones(args.text)
    write_g2p_json(result, args.output)
    print(f"Wrote G2P result to {args.output}")
    print(" ".join(result.phone_sequence))


if __name__ == "__main__":
    main()
