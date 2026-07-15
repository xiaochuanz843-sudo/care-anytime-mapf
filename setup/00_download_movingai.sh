#!/bin/bash
# Download the MovingAI MAPF benchmark maps + random scenarios needed by this battery.
# Idempotent: skips anything already present in data/maps + data/scen-random.
# Source: https://movingai.com/benchmarks/mapf/ (Stern et al., SoCS 2019)
set -e
PKG="$(cd "$(dirname "$0")/.." && pwd)"
MAPS="$PKG/data/maps"; SCEN="$PKG/data/scen-random"; TMP="$PKG/data/_dl"
mkdir -p "$MAPS" "$SCEN" "$TMP"

# the 33 maps of the battery (see experiments/gen_config_aaai.py NSTAR)
MAPLIST="Berlin_1_256 Boston_0_256 Paris_1_256 brc202d den312d den520d empty-16-16 empty-32-32
empty-48-48 empty-8-8 ht_chantry ht_mansion_n lak303d lt_gallowstemplar_n maze-128-128-1
maze-128-128-10 maze-128-128-2 maze-32-32-2 maze-32-32-4 orz900d ost003d random-32-32-10
random-32-32-20 random-64-64-10 random-64-64-20 room-32-32-4 room-64-64-16 room-64-64-8
w_woundedcoast warehouse-10-20-10-2-1 warehouse-10-20-10-2-2 warehouse-20-40-10-2-1
warehouse-20-40-10-2-2"

need=0
for m in $MAPLIST; do
  [ -f "$MAPS/$m.map" ] || need=1
  for s in $(seq 1 25); do [ -f "$SCEN/$m-random-$s.scen" ] || need=1; done
done
if [ "$need" = 0 ]; then echo "[movingai] all maps+scens present - skip download"; exit 0; fi

echo "[movingai] downloading benchmark archives ..."
cd "$TMP"
# corrupt/partial zips from an interrupted run are deleted and re-downloaded (idempotence);
# NOTE: cp -n returns non-zero on existing targets under coreutils>=9.2, so use test||cp.
for z in mapf-map.zip mapf-scen-random.zip; do
  if [ -f "$z" ] && ! unzip -tq "$z" >/dev/null 2>&1; then
    echo "[movingai] $z corrupt (interrupted download?) - re-downloading"; rm -f "$z"
  fi
  if [ ! -f "$z" ]; then
    wget -q --timeout=60 --tries=3 "https://movingai.com/benchmarks/mapf/$z" \
      || { echo "[movingai] FATAL: download of $z failed (no network? proxy?) - fetch it manually into data/_dl/"; exit 1; }
  fi
  unzip -tq "$z" >/dev/null 2>&1 || { echo "[movingai] FATAL: $z failed integrity check"; exit 1; }
done
unzip -oq mapf-map.zip -d unz_map
unzip -oq mapf-scen-random.zip -d unz_scen

missing=0
for m in $MAPLIST; do
  src=$(find unz_map -name "$m.map" | head -1)
  if [ -n "$src" ]; then [ -f "$MAPS/$m.map" ] || cp "$src" "$MAPS/"; else echo "MISSING map: $m"; missing=1; fi
  for s in $(seq 1 25); do
    sf=$(find unz_scen -name "$m-random-$s.scen" | head -1)
    if [ -n "$sf" ]; then [ -f "$SCEN/$m-random-$s.scen" ] || cp "$sf" "$SCEN/"; else echo "MISSING scen: $m-random-$s"; missing=1; fi
  done
done
[ "$missing" = 1 ] && { echo "[movingai] FATAL: files missing after download"; exit 1; }
echo "[movingai] OK: $(ls "$MAPS" | wc -l) maps, $(ls "$SCEN" | wc -l) scens"
