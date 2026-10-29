"""
S0: rebuild everything from the raw dataset with one command.

    python scripts/bootstrap.py --tracebench "<path>/HDFS_v3_TraceBench/tracebench"
    python scripts/bootstrap.py --tracebench "<...>" --train      # also retrain (hours on CPU)
    python scripts/bootstrap.py --check                           # verify only, build nothing

Steps (each is skipped if its output already exists; --force redoes them):

    1. corpus      data/processed/corpus.jsonl.gz     src/data/build_sequences.py
    2. model       src/models/saved/triage_lstm.pth   src/models/train.py   (only with --train;
                                                       the trained checkpoint is committed)
    3. evaluation  reports/evaluation.json            src/models/evaluate.py
    4. rca         src/models/saved/op_latency_baseline.json   scripts/run_rca_eval.py
    5. samples     samples/*.json                     scripts/make_samples.py
    6. tests       pytest -q

The dataset itself (TraceBench, ~3 GB) is not in the repository and is not
downloaded here: obtain it from the TraceBench authors and point --tracebench at
the `tracebench` folder that contains the scenario directories.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable

CHECKS = {
    "corpus": ROOT / "data/processed/corpus.jsonl.gz",
    "model checkpoint": ROOT / "src/models/saved/triage_lstm.pth",
    "vocab": ROOT / "src/models/saved/vocab.json",
    "evaluation": ROOT / "reports/evaluation.json",
    "rca baseline": ROOT / "src/models/saved/op_latency_baseline.json",
    "samples": ROOT / "samples/canary_good.json",
}


def run(label, *cmd):
    print(f"\n=== {label}\n$ {' '.join(map(str, cmd))}", flush=True)
    r = subprocess.run([str(c) for c in cmd], cwd=ROOT)
    if r.returncode not in (0, 2):          # build_sequences returns 2 = "rerun to continue"
        sys.exit(f"step failed: {label} (exit {r.returncode})")
    return r.returncode


def status():
    print("artifact status:")
    ok = True
    for name, path in CHECKS.items():
        here = path.exists()
        ok &= here
        print(f"  [{'x' if here else ' '}] {name:<18} {path.relative_to(ROOT)}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracebench", help="path to the TraceBench `tracebench` folder")
    ap.add_argument("--train", action="store_true", help="retrain the model (slow)")
    ap.add_argument("--force", action="store_true", help="redo steps whose output exists")
    ap.add_argument("--check", action="store_true", help="report status and exit")
    ap.add_argument("--skip-tests", action="store_true")
    args = ap.parse_args()

    ok = status()
    if args.check:
        sys.exit(0 if ok else 1)

    need = lambda key: args.force or not CHECKS[key].exists()   # noqa: E731

    if need("corpus"):
        if not args.tracebench:
            sys.exit("the corpus is missing: pass --tracebench <path to dataset>")
        out = ROOT / "data/processed"
        base = [PY, "src/data/build_sequences.py", "--tracebench", args.tracebench,
                "--out-dir", out]
        # resumable: rerun until every scenario shard exists
        while run("build corpus shards", *base, "--per-class", 20000) == 2:
            pass
        run("merge shards", *base, "--merge")

    if args.train:
        run("train triage model", PY, "src/models/train.py", "--epochs", 20,
            "--batch-size", 256, "--lr", 0.003, "--hidden", 64, "--emb-dim", 48,
            "--max-len", 256, "--patience", 4, "--weight-power", 0.5)
    elif need("model checkpoint"):
        sys.exit("no trained checkpoint: it is committed to the repo - restore "
                 "src/models/saved/, or rerun with --train")

    if need("evaluation"):
        run("evaluate vs baselines", PY, "src/models/evaluate.py")
    if need("rca baseline"):
        run("RCA baseline + localisation eval", PY, "scripts/run_rca_eval.py")
    if need("samples"):
        run("generate samples", PY, "scripts/make_samples.py")
    if not args.skip_tests:
        run("tests", PY, "-m", "pytest", "-q")
    print("\nbootstrap complete")
    status()


if __name__ == "__main__":
    main()
