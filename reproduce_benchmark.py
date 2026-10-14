#!/usr/bin/env python3
"""Reproduce the DTE-aerial-bench results of the released model in one command.

    python reproduce_benchmark.py                    # download, verify, evaluate, compare
    python reproduce_benchmark.py --download-only    # only download and unpack (e.g. on a login node)
    python reproduce_benchmark.py --skip-download    # only evaluate (data already in --work-dir)

Steps:
1. Downloads DTE-aerial-bench (metadata CSV, tiles and masks archives, SHA256SUMS) from
   huggingface.co/datasets/ayushi3536/deadtree.earth-aerial-bench and the released MiT-B3 model from
   huggingface.co/ayushi3536/DTE-aerial-model, and verifies the benchmark files against SHA256SUMS.
2. Unpacks the benchmark into <work-dir>/DTE-aerial-bench.
3. Writes an evaluation config pointing to it and runs eval.py.
4. Compares the macro F1 (overall, per resolution, per biome) with the values of the released model
   (MiT-B3, seed 200, as used in the paper) and exits with status 1 if any differs by more than --tolerance.

Note: the paper's main tables report the mean over three seeds; the released model is the seed-200 run, so its
numbers differ slightly from those means (e.g. mortality F1 0.598 vs. 0.59 in Table 3).
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import tarfile

BENCH_REPO = "ayushi3536/deadtree.earth-aerial-bench"
MODEL_REPO = "ayushi3536/DTE-aerial-model"
MODEL_FILE = "DTE_aerial_model.safetensors"
BENCH_FILES = ["DTE-aerial-bench-meta.csv", "DTE-aerial-bench-tiles.tar", "DTE-aerial-bench-masks.tar"]

# Macro F1 of the released model (MiT-B3, seed 200) on DTE-aerial-bench: (mortality, tree cover).
EXPECTED = {
    "overall": (0.5976, 0.8916),
    "5cm": (0.6094, 0.8919),
    "10cm": (0.5797, 0.8957),
    "20cm": (0.4523, 0.8901),
    "Temperate Forests": (0.5971, 0.8509),
    "Tropical and Subtropical Forests": (0.6698, 0.9192),
    "Mediterranean Forests": (0.5658, 0.9228),
    "Boreal Forests": (0.5743, 0.9293),
}

REPO_DIR = os.path.dirname(os.path.abspath(__file__))


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(work_dir):
    from huggingface_hub import hf_hub_download

    dl = os.path.join(work_dir, "downloads")
    files = {}
    for name in BENCH_FILES + ["SHA256SUMS"]:
        print(f"Downloading {BENCH_REPO}/{name} ...", flush=True)
        files[name] = hf_hub_download(BENCH_REPO, name, repo_type="dataset", local_dir=dl)
    sums = dict(line.split()[::-1] for line in open(files["SHA256SUMS"]) if line.strip())
    for name in BENCH_FILES:
        got = sha256(files[name])
        if got != sums[name]:
            sys.exit(f"Checksum mismatch for {name}: {got} != {sums[name]}")
        print(f"  {name}: sha256 OK")

    print(f"Downloading {MODEL_REPO}/{MODEL_FILE} ...", flush=True)
    hf_hub_download(MODEL_REPO, MODEL_FILE, local_dir=os.path.join(work_dir, "DTE-aerial-model"))

    bench = os.path.join(work_dir, "DTE-aerial-bench")
    os.makedirs(bench, exist_ok=True)
    for name in ["DTE-aerial-bench-tiles.tar", "DTE-aerial-bench-masks.tar"]:
        print(f"Unpacking {name} ...", flush=True)
        with tarfile.open(files[name]) as tar:
            for m in tar.getmembers():                    # only plain files inside tiles/ or masks/
                if not (m.isfile() or m.isdir()) or m.name.startswith(("/", "..")) or ".." in m.name.split("/"):
                    sys.exit(f"Unexpected archive member: {m.name}")
            tar.extractall(bench)
    meta = os.path.join(bench, "DTE-aerial-bench-meta.csv")
    with open(files["DTE-aerial-bench-meta.csv"], "rb") as src, open(meta, "wb") as dst:
        dst.write(src.read())


def check_data(work_dir):
    import csv

    bench = os.path.join(work_dir, "DTE-aerial-bench")
    rows = list(csv.DictReader(open(os.path.join(bench, "DTE-aerial-bench-meta.csv"))))
    missing = [r[c] for r in rows for c in ("tile_path", "mask_path") if not os.path.exists(os.path.join(bench, r[c]))]
    if len(rows) != 525 or missing:
        sys.exit(f"Benchmark incomplete: {len(rows)} rows, {len(missing)} missing files (e.g. {missing[:2]})")
    print(f"Benchmark complete: {len(rows)} patches with image and mask.")
    return bench


def evaluate(work_dir, bench):
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(os.path.join(REPO_DIR, "config", "evaluation.yml"))
    cfg._base_ = os.path.join(REPO_DIR, "config", cfg._base_)   # absolute, so the config can live in work_dir
    cfg.data.input_dir = bench
    cfg_path = os.path.join(work_dir, "evaluation.yml")
    OmegaConf.save(cfg, cfg_path)

    out = os.path.join(work_dir, "results", "results")
    ckpt = os.path.join(work_dir, "DTE-aerial-model", MODEL_FILE)
    cmd = [sys.executable, "eval.py", "--cfg", cfg_path, "--checkpoint", ckpt, "--output", out]
    print("Running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=REPO_DIR, check=True)
    return json.load(open(os.path.join(os.path.dirname(out), "results.json")))


def compare(res, tol):
    got = {"overall": (res["deadtree_macro_f1"], res["forest_cover_macro_f1"])}
    for group in ("per_resolution", "per_biome"):
        for name, m in res[group].items():
            got[name] = (m["deadtree_f1"], m["forest_f1"])

    print(f"\n{'Group':<34}{'Mortality F1':>22}{'Tree-cover F1':>22}")
    print(f"{'':<34}{'got / expected':>22}{'got / expected':>22}")
    failed = []
    for name, (em, et) in EXPECTED.items():
        if name not in got:
            failed.append(name)
            print(f"{name:<34}{'missing':>22}")
            continue
        gm, gt = got[name]
        ok = abs(gm - em) <= tol and abs(gt - et) <= tol
        if not ok:
            failed.append(name)
        print(f"{name:<34}{gm:>12.4f} / {em:.4f}{gt:>12.4f} / {et:.4f}  {'OK' if ok else 'MISMATCH'}")
    if failed:
        print(f"\n{len(failed)} group(s) differ by more than {tol}: {failed}")
        sys.exit(1)
    print(f"\nAll {len(EXPECTED)} groups match the released model's results (tolerance {tol}).")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--work-dir", default="reproduce", help="folder for downloads, data and results")
    p.add_argument("--download-only", action="store_true")
    p.add_argument("--skip-download", action="store_true")
    p.add_argument("--tolerance", type=float, default=0.005)
    a = p.parse_args()
    work_dir = os.path.abspath(a.work_dir)
    os.makedirs(work_dir, exist_ok=True)

    if not a.skip_download:
        download(work_dir)
    bench = check_data(work_dir)
    if a.download_only:
        return
    compare(evaluate(work_dir, bench), a.tolerance)


if __name__ == "__main__":
    main()
