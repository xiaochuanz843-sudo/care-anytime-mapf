#!/bin/bash
# ============================================================================
# One-shot battery orchestration.  Usage:
#   ./run_all.sh smoke     # smoke battery only (ALWAYS run this first on every machine)
#   ./run_all.sh m1        # machine 1: generates configs, runs LOCK first, then its shard
#   ./run_all.sh m2        # machine 2: REQUIRES configs copied from machine 1 (see below)
#
# Two-machine protocol (review finding: configs must be generated ONCE, on machine 1):
#   1. on m1:  setup steps + ./run_all.sh smoke + nohup ./run_all.sh m1 &
#   2. copy m1's experiments/gen_n_final.json + experiments/configs/ to m2 (same paths)
#   3. on m2:  setup steps + ./run_all.sh smoke + nohup ./run_all.sh m2 &
# Everything is checkpointed: re-running the same command resumes where it stopped.
# ============================================================================
set -e
PKG="$(cd "$(dirname "$0")" && pwd)"
cd "$PKG"
ROLE="${1:?usage: run_all.sh smoke|m1|m2}"
PY=python3

stage() { echo; echo "==== [$(date '+%F %T')] $* ===="; }

run_phase() {  # $1 = config, $2 = --only phases ('' = all): runner + one retry pass + resume
  local CFG="$1" PH="$2" ONLY=()
  [ -n "$PH" ] && ONLY=(--only "$PH")
  if ! $PY experiments/runner.py --root "$PKG" --config "$CFG" "${ONLY[@]}"; then
    echo "FATAL: runner aborted (circuit breaker or hard error) on $CFG $PH -- fix before continuing"
    exit 2
  fi
  $PY experiments/retry_failed.py --root "$PKG" --config "$CFG" ${PH:+--only "$PH"}
  if ! $PY experiments/runner.py --root "$PKG" --config "$CFG" "${ONLY[@]}"; then
    echo "FATAL: runner aborted on the retry pass of $CFG $PH"
    exit 2
  fi
}

# ---------- stage 0: preflight ----------
stage "preflight (role=$ROLE)"
for b in src/lns2/lns src/balance/balance src/address/address src/tackle/tackle; do
  [ -x "$b" ] || { echo "FATAL: $b not built - run setup/01_build_all.sh"; exit 1; }
done
[ -f data/maps/random-32-32-20.map ] || { echo "FATAL: maps missing - run setup/00_download_movingai.sh"; exit 1; }
[ -f data/gen_maps/MANIFEST.json ] || { echo "FATAL: generated maps missing - run setup/02_gen_maps.py --install"; exit 1; }

# ---------- stage 1: smoke gate (mandatory on every machine) ----------
stage "smoke battery (gate: must pass before anything else; ~30-40 min on 16 cores)"
[ -f experiments/gen_n_final.json ] || echo "WARN: gen_n_final.json missing - smoke's generated-map cells use the un-laddered design N and may fail; run setup/03_nprobe.py first (m1) or copy the file from m1 (m2)"
$PY experiments/gen_config_smoke.py
run_phase experiments/configs/smoke.json ""
$PY analysis/check_smoke.py || { echo "FATAL: smoke FAILED - do not launch the battery"; exit 1; }
if [ "$ROLE" = smoke ]; then echo "smoke PASSED"; exit 0; fi

# ---------- stage 2: configs (generated ONCE, on m1; m2 uses the copies) ----------
stage "battery configs"
if [ "$ROLE" = m1 ]; then
  [ -f experiments/gen_n_final.json ] || { echo "FATAL: experiments/gen_n_final.json missing - run setup/03_nprobe.py on THIS machine first"; exit 1; }
  # Regenerate whenever the battery configs are missing OR OLDER than gen_n_final.json. A stale
  # config shipped in the package (generated before nprobe) must NEVER be silently reused - it
  # would run the G batch with un-laddered design N (wasted compute + off-preregistration).
  # Regeneration is deterministic (same gen_n_final => byte-identical configs + md5), so this is
  # a no-op on a genuine resume.
  if [ ! -f experiments/configs/m1.json ] || [ experiments/gen_n_final.json -nt experiments/configs/m1.json ]; then
    echo "generating fresh battery configs from gen_n_final.json"
    rm -f experiments/configs/m1.json experiments/configs/m1.json.md5 \
          experiments/configs/m2.json experiments/configs/m2.json.md5 \
          experiments/configs/lock.json experiments/configs/lock.json.md5
    $PY experiments/gen_config_aaai.py --m1-cores "${M1_CORES:-80}" --m2-cores "${M2_CORES:-64}"
  else
    echo "battery configs are newer than gen_n_final.json - reusing (deterministic)"
  fi
  echo "REMINDER: copy experiments/gen_n_final.json + experiments/configs/ to machine 2 (same paths)"
else
  for f in experiments/configs/m2.json experiments/gen_n_final.json; do
    [ -f "$f" ] || { echo "FATAL: $f missing - copy it from machine 1 (configs are generated ONCE, on m1; see header)"; exit 1; }
  done
  ( cd experiments/configs && md5sum -c m2.json.md5 ) || { echo "FATAL: m2.json does not match its md5 sidecar - re-copy from machine 1"; exit 1; }
fi

# ---------- stage 3 (m1 only): LOCK decontention, alone at 8 workers ----------
if [ "$ROLE" = m1 ]; then
  stage "Batch L: LOCK decontention (8 workers; nothing else may run on this box now)"
  run_phase experiments/configs/lock.json ""
fi

# ---------- stage 4: main battery (phases in information-density order) ----------
CFG="experiments/configs/$ROLE.json"
for PH in E0,A B G; do
  stage "main battery phase(s) $PH"
  run_phase "$CFG" "$PH"
done

# ---------- stage 5: aggregate ----------
# results/lock exists only on m1 (LOCK batch); results/traj only after B batch. Include each
# only if present, else aggregate's dir-existence guard would FATAL on m2 (review fix).
stage "aggregate + report tables"
AGG=(--results results/main)
[ -d results/lock ] && AGG+=(--results results/lock)
[ -d results/traj ] && AGG+=(--traj results/traj)
$PY analysis/aggregate_aaai.py "${AGG[@]}" --out analysis/out \
    || { echo "FATAL: aggregation failed (pairing conflict or bad inputs) - inspect before trusting any table"; exit 1; }

stage "DONE (role=$ROLE)"
if [ "$ROLE" = m2 ]; then
  echo "Now copy this machine's results/main/ into machine 1's results/main/ and re-run stage-5 aggregation there."
fi
