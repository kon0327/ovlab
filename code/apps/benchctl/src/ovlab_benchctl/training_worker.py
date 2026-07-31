"""Foreground container workers for training and checkpoint finalization."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

from .training_runs import (
    CheckpointBundleStore,
    OpenVlaTrainerAdapter,
    TrainingRunContext,
    TrainingRunStore,
)


def _context(root: Path) -> TrainingRunContext:
    plan = json.loads((root / "training-plan.json").read_text(encoding="utf-8"))
    return TrainingRunContext(run_id=root.name, root=root, plan=plan)


def train(root: Path) -> int:
    context = _context(root)
    store = TrainingRunStore(root.parent.parent)
    adapter = OpenVlaTrainerAdapter()
    try:
        adapter.preflight(context.plan)
        store.event(root, "preparing")
        adapter.initialize(context.plan, context)
        store.event(root, "running")
        adapter.train(context.plan, context)
        adapter.finalize(context)
        return 0
    except KeyboardInterrupt as exc:
        adapter.interrupt(context)
        store.fail(context, exc, interrupted=True)
        return 130
    except Exception as exc:
        store.fail(context, exc)
        print(f"ovlab training worker: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    finally:
        adapter.close()
        if context.close_count != 1:
            print(f"ovlab training worker: trainer close count was {context.close_count}, expected 1", file=sys.stderr)


def finalize(root: Path, model_data_root: Path) -> int:
    context = _context(root)
    try:
        result = CheckpointBundleStore(model_data_root).finalize(context)
        print(json.dumps(result, sort_keys=True, separators=(",", ":")))
        return 0
    except Exception as exc:
        TrainingRunStore(model_data_root).fail(context, exc)
        print(f"ovlab checkpoint finalizer: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="ovlab-training-worker")
    subcommands = parser.add_subparsers(dest="command", required=True)
    training = subcommands.add_parser("train")
    training.add_argument("--run-root", required=True)
    finalizer = subcommands.add_parser("finalize")
    finalizer.add_argument("--run-root", required=True)
    finalizer.add_argument("--model-data-root", required=True)
    args = parser.parse_args(argv)
    if args.command == "train":
        return train(Path(args.run_root).resolve())
    return finalize(Path(args.run_root).resolve(), Path(args.model_data_root).resolve())


if __name__ == "__main__":
    raise SystemExit(main())
