#!/usr/bin/env python3
"""Build any registered image role without a model-specific dispatch table."""
import argparse
import json
from pathlib import Path
from ovlab_benchctl.plugin_build import build_images


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('roles', nargs='*')
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    print(json.dumps(build_images(args.root, args.roles or None, offline=args.offline, dry_run=args.dry_run), indent=2))


if __name__ == '__main__':
    main()
