"""Generic offline worker for a registered checkpoint preparation extension."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .training_plugins import extension_symbol, preparation_for_checkpoint


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--dtype', default='bfloat16')
    parser.add_argument('--loader', help='registered loader; defaults to the checkpoint manifest')
    args = parser.parse_args(argv)
    checkpoint_path = args.checkpoint / 'checkpoint.json'
    checkpoint = json.loads(checkpoint_path.read_text()) if checkpoint_path.is_file() else {}
    if args.loader:
        checkpoint['expected_loader'] = args.loader
    preparer, declaration = preparation_for_checkpoint(checkpoint)
    os.umask(0o007)
    preparer.prepare(args.base, args.checkpoint, args.output, dtype=args.dtype)
    print(json.dumps({'status': 'completed', 'converter': declaration['converter_version']}))
    return 0


def __getattr__(name):
    return extension_symbol('checkpoint_preparation', name)


if __name__ == '__main__':
    raise SystemExit(main())
