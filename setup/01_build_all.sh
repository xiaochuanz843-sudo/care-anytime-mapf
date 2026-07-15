#!/bin/bash
# Build all 4 host binaries from source (tackle binary serves both tackle and tackle_alns modes).
# Deps: cmake >= 3.10, g++ >= 9, libboost (program_options, system, filesystem), libeigen3.
#   Ubuntu: apt-get install -y cmake g++ libboost-all-dev libeigen3-dev
set -e
PKG="$(cd "$(dirname "$0")/.." && pwd)"
LOG="$PKG/setup/build.log"; : > "$LOG"
for h in lns2 balance address tackle; do
  echo "[build] $h ..."
  cd "$PKG/src/$h"
  if ! cmake -DCMAKE_BUILD_TYPE=Release . >> "$LOG" 2>&1 || ! make -j"$(nproc)" >> "$LOG" 2>&1; then
    echo "[build] $h FAILED - last 20 lines of $LOG:"; tail -20 "$LOG"
    echo "(deps: apt-get install -y cmake g++ libboost-all-dev libeigen3-dev)"; exit 1
  fi
  echo "[build] $h OK"
done
echo "[build] binaries:"
ls -la "$PKG"/src/lns2/lns "$PKG"/src/balance/balance "$PKG"/src/address/address "$PKG"/src/tackle/tackle
