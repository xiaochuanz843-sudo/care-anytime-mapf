#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Retry pass for failed runs (audit fix: runner freezes status=fail results on resume).

Deletes result JSONs with status=fail so the next runner resume re-executes them, at most
MAX retries per run_id (ledger kept in <workdir>/retries.json).

--only PHASES is MANDATORY discipline when the follow-up runner uses --only: without it this
script would delete fail records of OTHER phases that the follow-up runner never re-executes,
erasing them permanently (package-review confirmed finding). run_all.sh always passes it.

Permanent failure reasons (never retried): csv_bad_vals, negative_delay, csv_missing_cols
(deterministic solver-output properties) and rc=86/rc=87 (the layer's PP-replan and unknown-env-value guards: deterministic
configuration errors, retrying burns budget for nothing).

Usage:  python3 retry_failed.py --config configs/m1.json --only E0,A [--root PKG] [--max 2] [--dry]
"""
import argparse, json, os, sys

PERMANENT = {"csv_bad_vals", "negative_delay", "csv_missing_cols", "rc=86", "rc=87"}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--only", default="",
                    help="comma list of phases to retry (match runner --only); empty = all")
    ap.add_argument("--root", default="")
    ap.add_argument("--max", type=int, default=2)
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()
    cfg = json.load(open(args.config))
    # root default: configs live at <root>/experiments/configs/<x>.json (same as runner.py)
    root = os.path.abspath(args.root) if args.root else \
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(args.config))))
    def resolve(p): return p if os.path.isabs(p) else os.path.join(root, p)
    rdir, wdir = resolve(cfg["results_dir"]), resolve(cfg["workdir"])
    phases = {p for p in args.only.split(",") if p}
    ledger_path = os.path.join(wdir, "retries.json")
    ledger = json.load(open(ledger_path)) if os.path.isfile(ledger_path) else {}
    retried = skipped_perm = skipped_max = skipped_phase = 0
    for fn in os.listdir(rdir) if os.path.isdir(rdir) else []:
        if not fn.endswith(".json"): continue
        p = os.path.join(rdir, fn)
        try: rec = json.load(open(p))
        except (ValueError, OSError): continue
        if rec.get("status") != "fail": continue
        if phases and str(rec.get("block_id", "")).split(".")[0] not in phases:
            skipped_phase += 1; continue
        rid = rec.get("run_id", fn[:-5])
        if rec.get("reason") in PERMANENT:
            skipped_perm += 1; continue
        if ledger.get(rid, 0) >= args.max:
            skipped_max += 1; continue
        ledger[rid] = ledger.get(rid, 0) + 1
        retried += 1
        if not args.dry:
            os.remove(p)
    if not args.dry:
        os.makedirs(wdir, exist_ok=True)
        tmp = ledger_path + ".tmp"
        with open(tmp, "w") as f: json.dump(ledger, f)
        os.replace(tmp, ledger_path)
    print(f"[retry_failed] queued_for_retry={retried} permanent={skipped_perm} "
          f"max_retries_reached={skipped_max} other_phase={skipped_phase} "
          f"(only={args.only or 'ALL'}, max={args.max}, dry={args.dry})")
    sys.exit(0)

if __name__ == "__main__":
    main()
