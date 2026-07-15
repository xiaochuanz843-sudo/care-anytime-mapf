#!/usr/bin/env python3
"""F-block generator v2: procedurally generated MovingAI-format maps + Stern-protocol scenarios.

Lineage: direct descendant of phaseb/gen_random_maps.py ("F-gen v1.0"). The six legacy
families (130 maps, scens 1-4) are BYTE-FROZEN: published data depends on them, and this
script regenerates them bit-identically (same rng_for(zlib.crc32) per-purpose streams, same
constants). Any drift against the legacy MANIFEST is a hard error during --generate/--check.

Families:
  grandom - i.i.d. Bernoulli obstacles, p ~ U{0.10, 0.20} per map     (Stern'19 random-*)
  gmaze   - recursive-backtracker maze, corridor width w ~ U{1, 2}    (Sturtevant maze-*)
  groom   - room lattice, doors p=0.75 + spanning-tree connectivity   (room-*)
  gware   - parametric warehouse: shelf 10x2, aisle a ~ U{1, 2}       (warehouse-*)
  gpogr / gpogm - POGEMA==1.3.1 external-tool side families (size 64, 5 seeds, exploratory).
      On machines without pogema, files are copied from the frozen legacy products
      (md5-verified); --check still fully regenerates them on machines that have pogema.
  gmazeb  - NEW v2: braided (cyclic) maze. gmaze's recursive-backtracker perfect maze
      (own stream rng_for("map","gmazeb",size,seed)), then per-wall independent
      Bernoulli(beta) knockout of corridor-separating interior walls (walls whose two
      opposite 4-neighbours are both free in the perfect-maze snapshot -> removal always
      creates a cycle, never a dead-end nub). Braid stream rng_for("gmazeb",size,seed,
      "braid") excludes beta, so b15/b30 of the same seed share the base maze AND the
      draw sequence (b15 loops are a subset of b30 loops: coupled comparison).
      2 sizes {64,128} x beta {0.15,0.30} x 6 seeds = 24 maps, gmazeb{size}-b{15|30}-s{seed}.
  gtiles  - NEW v2: official-city-map tiles. Paris_1_256 / Berlin_1_256 / Boston_0_256 are
      cut into 16 disjoint 64x64 tiles (row-major); keep tiles with obstacle density in
      [0.05, 0.45] AND |LCC| >= 0.60 * free; first 8 qualifying tiles per city (<= 24 maps),
      gtiles-{City}-t{idx}. MANIFEST records source-map md5 + slicing rule version.
      If the source city maps are not on this machine the family is skipped with a notice
      (rerun after setup/00_download_movingai.sh; the rule is deterministic).

Scenario protocol (Stern et al. SoCS'19, literal): all points of the largest connected
component randomly paired; the first N_high pairs form one scenario. v2: EIGHT scenarios
per map. Scens 1-4 use the unchanged v1 stream rng_for("scen", name, j) and remain
byte-identical; scens 5-8 are the natural extension j=5..8 of the same key structure.

N design (v2 audit fix): N_high = min(NCAP, max(NFLOOR+1, round(0.15*|LCC|)))  [unchanged],
N_low = max(NFLOOR, min(NCAP, N_high-1, round(0.05*|LCC|)))  [now NCAP-capped and forced
below N_high]. If N_low >= N_high the map keeps a single N tier (N_low written equal to
N_high and flagged "single_N": true; downstream gen_config collapses the contrast).

NO post-hoc exclusions: seed schedules are exhaustive ranges; the gtiles filter is a
preregistered deterministic rule applied in fixed row-major order.

CLI: --generate (default; idempotent - files already on disk with the right md5 are
skipped), --check (full regeneration, byte-diff vs installed files + MANIFEST),
--install (copy maps+scens into data/maps + data/scen-random, never overwriting),
--validate (parse written files back and re-verify LCC / N / scenario invariants).
"""
import argparse, hashlib, json, os, random, shutil, sys, zlib
from collections import deque

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
OUTDIR = os.path.join(PKG, "data", "gen_maps")
DATA_MAPS = os.path.join(PKG, "data", "maps")
DATA_SCEN = os.path.join(PKG, "data", "scen-random")
CITY_SRC_DIRS = (DATA_MAPS, os.path.join(PKG, "movingai", "maps"))
# Frozen reference = the pre-generated products shipped INSIDE this package (data/gen_maps).
# --generate byte-checks regeneration against it; pogema families fall back to copying from it
# when pogema is not installed. Override --legacy-dir only for external provenance audits.
DEFAULT_LEGACY = OUTDIR

SIZES = (64, 128)
SEEDS = list(range(15))
POG_SEEDS = list(range(5))
RHO = (0.05, 0.15)
NCAP = 1000
NFLOOR = 10
SCENS_PER_MAP = 8            # v2 (v1 wrote 1-4; those stay byte-frozen)
LEGACY_SCENS = 4
GEN_VERSION = "F-gen v2.0"
POGEMA_PIN = "1.3.1"

MZB_SIZES = (64, 128)
MZB_BETAS = ((15, 0.15), (30, 0.30))     # (name tag, probability)
MZB_SEEDS = list(range(6))

TILE_CITIES = ("Paris_1_256", "Berlin_1_256", "Boston_0_256")
TILE = 64
TILE_GRID = 4                            # 4x4 = 16 disjoint tiles, row-major
TILE_DENSITY = (0.05, 0.45)              # inclusive obstacle-density band
TILE_LCC_MIN = 0.60                      # |LCC| >= 0.60 * free
TILE_PER_CITY = 8
TILE_RULE_VERSION = "gtiles-v1 (64x64 row-major 4x4, density [0.05,0.45], LCC>=0.60*free, first 8/city)"

CHANGELOG = [
    "v2.0: scenarios per map 4 -> 8; scens 1-4 byte-frozen from F-gen v1.0 "
    "(unchanged stream rng_for('scen', name, j)); scens 5-8 = same key, j=5..8",
    "v2.0: new family gmazeb (24 maps) - braided cyclic mazes: gmaze backtracker base "
    "(rng_for('map','gmazeb',size,seed)) + per-wall independent Bernoulli(beta) knockout of "
    "corridor-separating interior walls via rng_for('gmazeb',size,seed,'braid'), "
    "beta in {0.15,0.30}, sizes {64,128}, seeds 0-5",
    "v2.0: new family gtiles (<=24 maps) - deterministic 64x64 row-major tiles of official "
    "MovingAI city maps (Paris_1_256/Berlin_1_256/Boston_0_256), density in [0.05,0.45], "
    "LCC>=0.60*free, first 8 qualifying per city; source md5s recorded",
    "v2.0 audit fix: N_low now NCAP-capped and forced < N_high "
    "(N_low = max(NFLOOR, min(NCAP, N_high-1, round(0.05*LCC)))); if the tiers collapse the "
    "map keeps a single N (flag single_N, N_low written equal to N_high)",
    "frozen: the 130 F-gen v1.0 maps and their scens 1-4 are byte-identical to the "
    "published legacy MANIFEST (cross-checked at generation time)",
]

# ---------------------------------------------------------------- rng streams
def rng_for(*key):
    return random.Random(zlib.crc32("|".join(map(str, key)).encode()))

# ---------------------------------------------------------------- families
# grid convention: grid[row][col], True = blocked. All movement 4-connected.

def gen_random(size, seed):
    r = rng_for("map", "grandom", size, seed)
    p = r.choice((0.10, 0.20))
    g = [[r.random() < p for _ in range(size)] for _ in range(size)]
    return g, {"p": p}

def _backtracker_maze(r, size):
    """Recursive backtracker on a cell lattice; passages w x w, walls 1 thick.
    Connected by construction (single DFS tree spans all cells). Draw order is
    byte-frozen (w choice first, then DFS neighbour choices) - do not touch."""
    w = r.choice((1, 2))
    ncells = (size - 1) // (w + 1)          # cells fitting into size with 1-thick walls + border
    g = [[True] * size for _ in range(size)]
    def carve_cell(cy, cx):
        y0, x0 = 1 + cy * (w + 1), 1 + cx * (w + 1)
        for dy in range(w):
            for dx in range(w):
                g[y0 + dy][x0 + dx] = False
    def carve_wall(cy, cx, ny, nx):
        y0, x0 = 1 + cy * (w + 1), 1 + cx * (w + 1)
        if ny > cy:      # wall strip below
            for dx in range(w): g[y0 + w][x0 + dx] = False
        elif ny < cy:
            for dx in range(w): g[y0 - 1][x0 + dx] = False
        elif nx > cx:
            for dy in range(w): g[y0 + dy][x0 + w] = False
        else:
            for dy in range(w): g[y0 + dy][x0 - 1] = False
    seen = [[False] * ncells for _ in range(ncells)]
    stack = [(0, 0)]
    seen[0][0] = True
    carve_cell(0, 0)
    while stack:
        cy, cx = stack[-1]
        nbrs = [(cy + dy, cx + dx) for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0))
                if 0 <= cy + dy < ncells and 0 <= cx + dx < ncells and not seen[cy + dy][cx + dx]]
        if not nbrs:
            stack.pop(); continue
        ny, nx = r.choice(nbrs)
        seen[ny][nx] = True
        carve_cell(ny, nx)
        carve_wall(cy, cx, ny, nx)
        stack.append((ny, nx))
    return g, w, ncells

def gen_maze(size, seed):
    r = rng_for("map", "gmaze", size, seed)
    g, w, ncells = _backtracker_maze(r, size)
    return g, {"w": w, "cells": ncells}

def gen_mazeb(size, seed, beta):
    """Braided maze: perfect maze (own map stream, beta-independent so both betas share
    the base), then knock out corridor-separating interior walls with prob beta.
    Candidates are evaluated on a SNAPSHOT of the perfect maze in row-major order, one
    independent Bernoulli draw each; the braid stream excludes beta, so the b15 loop set
    of a seed is a subset of its b30 loop set (coupled comparison)."""
    r = rng_for("map", "gmazeb", size, seed)
    g, w, ncells = _backtracker_maze(r, size)
    snap = [row[:] for row in g]
    rb = rng_for("gmazeb", size, seed, "braid")
    cand = removed = 0
    for y in range(1, size - 1):
        for x in range(1, size - 1):
            if not snap[y][x]:
                continue
            if ((not snap[y - 1][x] and not snap[y + 1][x]) or
                    (not snap[y][x - 1] and not snap[y][x + 1])):
                cand += 1
                if rb.random() < beta:
                    g[y][x] = False
                    removed += 1
    return g, {"w": w, "cells": ncells, "beta": beta,
               "wall_candidates": cand, "walls_removed": removed}

def gen_room(size, seed):
    """Room lattice r x r, walls 1 thick; each adjacent room pair gets a random door with
    p=0.75; a random spanning tree over the room lattice then guarantees connectivity
    (doors added on tree edges that lack one). Connected by construction."""
    r_ = rng_for("map", "groom", size, seed)
    rs = r_.choice((8, 16))
    nrooms = (size - 1) // (rs + 1)
    g = [[True] * size for _ in range(size)]
    for i in range(nrooms):
        for j in range(nrooms):
            y0, x0 = 1 + i * (rs + 1), 1 + j * (rs + 1)
            for dy in range(rs):
                for dx in range(rs):
                    g[y0 + dy][x0 + dx] = False
    doors = set()
    def open_door(i, j, di, dj):
        y0, x0 = 1 + i * (rs + 1), 1 + j * (rs + 1)
        off = r_.randrange(rs)
        if dj:   # door in vertical wall right of room (i,j)
            g[y0 + off][x0 + rs] = False
        else:    # door in horizontal wall below room (i,j)
            g[y0 + rs][x0 + off] = False
        doors.add(((i, j), (i + di, j + dj)))
    for i in range(nrooms):
        for j in range(nrooms):
            if j + 1 < nrooms and r_.random() < 0.75: open_door(i, j, 0, 1)
            if i + 1 < nrooms and r_.random() < 0.75: open_door(i, j, 1, 0)
    # random spanning tree (randomized DFS over the room lattice)
    seen = {(0, 0)}
    stack = [(0, 0)]
    while stack:
        i, j = stack[-1]
        nbrs = [(i + di, j + dj, di, dj) for di, dj in ((0, 1), (0, -1), (1, 0), (-1, 0))
                if 0 <= i + di < nrooms and 0 <= j + dj < nrooms and (i + di, j + dj) not in seen]
        if not nbrs:
            stack.pop(); continue
        ni, nj, di, dj = nbrs[r_.randrange(len(nbrs))]
        seen.add((ni, nj))
        edge = ((i, j), (ni, nj)) if (di, dj) in ((0, 1), (1, 0)) else ((ni, nj), (i, j))
        if edge not in doors:
            if edge[0][0] == edge[1][0]:  # horizontal neighbors -> vertical wall
                open_door(edge[0][0], edge[0][1], 0, 1)
            else:
                open_door(edge[0][0], edge[0][1], 1, 0)
        stack.append((ni, nj))
    return g, {"r": rs, "rooms": nrooms}

def gen_ware(size, seed):
    """Parametric warehouse: 10x2 shelf blocks tiled with aisle width a ~ U{1,2} in both
    directions, open margin 3 around the perimeter. Free space (aisles + margin) is
    connected by construction (shelves are isolated rectangles)."""
    r = rng_for("map", "gware", size, seed)
    a = r.choice((1, 2))
    sw, sh, margin = 10, 2, 3
    g = [[False] * size for _ in range(size)]
    y = margin
    while y + sh <= size - margin:
        x = margin
        while x + sw <= size - margin:
            for dy in range(sh):
                for dx in range(sw):
                    g[y + dy][x + dx] = True
            x += sw + a
        y += sh + a
    return g, {"aisle": a, "shelf": [sw, sh], "margin": margin}

def gen_pogema_random(size, seed):
    import warnings; warnings.filterwarnings("ignore")
    from pogema import GridConfig
    from pogema.generator import generate_obstacles
    p = rng_for("map", "gpogr", size, seed).choice((0.10, 0.20))
    ob = generate_obstacles(GridConfig(size=size, density=p, seed=seed, num_agents=1))
    g = [[bool(ob[y][x]) for x in range(len(ob[y]))] for y in range(len(ob))]
    return g, {"p": p, "tool": "pogema.generator.generate_obstacles"}

def gen_pogema_maze(size, seed):
    import warnings; warnings.filterwarnings("ignore")
    from pogema_toolbox.generators.maze_generator import MazeGenerator, MazeRangeSettings
    settings = MazeRangeSettings(width_min=size, width_max=size,
                                 height_min=size, height_max=size).sample(seed)
    maze = MazeGenerator.generate_maze(**settings)
    rows = maze.split("\n") if isinstance(maze, str) else list(maze)
    rows = [row for row in rows if row]
    g = [[c == "#" for c in row] for row in rows]
    pub = {k: (v.item() if hasattr(v, "item") else v)
           for k, v in settings.items() if k != "seed"}
    return g, {"tool": "pogema_toolbox.maze_generator (validation protocol, size pinned)",
               **pub}

LEGACY_FAMILIES = {   # name -> (fn, sizes, seeds, confirmatory)   [byte-frozen, v1.0 order]
    "grandom": (gen_random, SIZES, SEEDS, True),
    "gmaze":   (gen_maze,   SIZES, SEEDS, True),
    "groom":   (gen_room,   SIZES, SEEDS, True),
    "gware":   (gen_ware,   SIZES, SEEDS, True),
    "gpogr":   (gen_pogema_random, (64,), POG_SEEDS, False),
    "gpogm":   (gen_pogema_maze,   (64,), POG_SEEDS, False),
}
CONNECTED_FAMILIES = ("gmaze", "gmazeb", "groom", "gware")   # single component by construction

# ---------------------------------------------------------------- scen machinery
def components(g):
    h, w = len(g), len(g[0])
    seen = [[False] * w for _ in range(h)]
    comps = []
    for sy in range(h):
        for sx in range(w):
            if g[sy][sx] or seen[sy][sx]: continue
            comp = []
            dq = deque([(sy, sx)]); seen[sy][sx] = True
            while dq:
                y, x = dq.popleft()
                comp.append((y, x))
                for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and not g[ny][nx] and not seen[ny][nx]:
                        seen[ny][nx] = True; dq.append((ny, nx))
            comps.append(comp)
    return comps

def stern_pairs(lcc, name, scen_j, n_pairs):
    """Stern-literal: shuffle the LCC, pair consecutive points; first n_pairs pairs.
    Starts and goals are disjoint cell sets by construction. Stream keyed by
    ("scen", map name, scenario index) - unchanged from v1.0, so j=1..4 are frozen."""
    r = rng_for("scen", name, scen_j)
    cells = sorted(lcc)          # canonical order first -> shuffle is seed-deterministic
    r.shuffle(cells)
    return [(cells[2 * i], cells[2 * i + 1]) for i in range(n_pairs)]

def map_text(g):
    h, w = len(g), len(g[0])
    rows = ["".join("@" if c else "." for c in row) for row in g]
    return f"type octile\nheight {h}\nwidth {w}\nmap\n" + "\n".join(rows) + "\n"

def parse_map_text(txt, ctx="map"):
    lines = txt.split("\n")
    assert lines[0] == "type octile" and lines[3] == "map", f"{ctx}: bad header"
    h = int(lines[1].split()[1]); w = int(lines[2].split()[1])
    rows = lines[4:4 + h]
    assert len(rows) == h and all(len(r) == w for r in rows), f"{ctx}: bad dims"
    bad = set("".join(rows)) - set(".@TOSW")
    assert not bad, f"{ctx}: unexpected chars {bad}"
    return [[c != "." for c in r] for r in rows]     # '.' free, everything else blocked

def scen_text(name, g, pairs):
    h, w = len(g), len(g[0])
    lines = ["version 1"]
    for (sy, sx), (gy, gx) in pairs:
        # MovingAI column order: bucket, map, width, height, sx(col), sy(row), gx(col), gy(row), dist
        lines.append(f"0\t{name}.map\t{w}\t{h}\t{sx}\t{sy}\t{gx}\t{gy}\t0")
    return "\n".join(lines) + "\n"

def md5_of(data):
    return hashlib.md5(data.encode() if isinstance(data, str) else data).hexdigest()

def md5_file(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

# ---------------------------------------------------------------- N design (v2)
def design_N(lcc_size):
    n_high = min(NCAP, max(NFLOOR + 1, round(RHO[1] * lcc_size)))
    n_low = max(NFLOOR, min(NCAP, n_high - 1, round(RHO[0] * lcc_size)))
    single = n_low >= n_high
    return n_low, n_high, single

# ---------------------------------------------------------------- sources
def find_city_map(city):
    for d in CITY_SRC_DIRS:
        p = os.path.join(d, city + ".map")
        if os.path.isfile(p):
            return p
    return None

def load_legacy(legacy_dir):
    mp = os.path.join(legacy_dir, "MANIFEST.json")
    if not os.path.isfile(mp):
        return None
    with open(mp) as f:
        man = json.load(f)
    return {"dir": legacy_dir, "man": man}

def resolve_pogema_mode(requested, legacy):
    if requested != "auto":
        return requested
    try:
        import pogema                                      # noqa: F401
        import pogema_toolbox.generators.maze_generator    # noqa: F401
        return "generate"
    except ImportError:
        return "copy" if legacy else "missing"

def pogema_version():
    try:
        import pogema
        return getattr(pogema, "__version__", "unknown")
    except ImportError:
        return None

# ---------------------------------------------------------------- generation
def build_all(pogema_mode, legacy, quiet=False):
    """Return ({relpath: text}, manifest, notes). Pure function of the frozen constants
    (+ legacy files in pogema 'copy' mode, + official city maps for gtiles)."""
    files, maps_meta, notes = {}, {}, []

    def say(msg):
        notes.append(msg)
        if not quiet:
            print(msg)

    def add_map(name, fam, seed, confirm, g, params, size=None):
        assert "." not in name, f"{name}: map names must not contain dots"
        h, w = len(g), len(g[0])
        comps = sorted(components(g), key=len, reverse=True)
        free = sum(len(c) for c in comps)
        lcc = comps[0]
        lcc_frac = len(lcc) / free if free else 0.0
        assert lcc_frac > 0.5, f"{name}: pathological LCC {lcc_frac:.2f} (generator bug)"
        if fam in CONNECTED_FAMILIES:
            assert len(comps) == 1, f"{name}: {fam} must be connected by construction"
        n_low, n_high, single = design_N(len(lcc))
        assert 2 * n_high <= len(lcc), f"{name}: LCC too small for pairing"
        files[f"maps/{name}.map"] = map_text(g)
        for j in range(1, SCENS_PER_MAP + 1):
            pairs = stern_pairs(lcc, name, j, n_high)
            files[f"scen-random/{name}-random-{j}.scen"] = scen_text(name, g, pairs)
        meta = {"family": fam, "seed": seed, "confirmatory": confirm, "params": params,
                "h": h, "w": w, "free": free, "lcc": len(lcc),
                "lcc_frac": round(lcc_frac, 4),
                "N_low": n_high if single else n_low, "N_high": n_high}
        if size is not None:
            meta["size"] = size
        if single:
            meta["single_N"] = True
            say(f"NOTE {name}: N tiers collapsed -> single N = {n_high}")
        maps_meta[name] = meta

    # ---- legacy six families (byte-frozen)
    for fam, (fn, sizes, seeds, confirm) in LEGACY_FAMILIES.items():
        for size in sizes:
            for seed in seeds:
                name = f"{fam}-{size}-s{seed}"
                if fam.startswith("gpog") and pogema_mode != "generate":
                    if pogema_mode != "copy":
                        raise SystemExit(
                            f"FATAL: {name}: pogema not importable and no legacy products "
                            f"to copy from (--legacy-dir). Install pogema=={POGEMA_PIN} or "
                            f"point --legacy-dir at the frozen F-gen v1.0 output.")
                    if not legacy:
                        raise SystemExit(
                            f"FATAL: {name}: pogema not importable and no frozen products "
                            f"available to copy from (data/gen_maps/MANIFEST.json missing). "
                            f"Install pogema=={POGEMA_PIN} or restore the shipped data/gen_maps/.")
                    src = os.path.join(legacy["dir"], "maps", name + ".map")
                    want = legacy["man"]["md5"].get(f"maps/{name}.map")
                    got = md5_file(src)
                    assert got == want, f"{name}: legacy copy md5 mismatch {got} != {want}"
                    with open(src, newline="") as f:
                        txt = f.read()
                    g = parse_map_text(txt, name)
                    params = legacy["man"]["maps"][name]["params"]
                    add_map(name, fam, seed, confirm, g, params, size=size)
                    files[f"maps/{name}.map"] = txt   # exact legacy bytes, not a re-render
                else:
                    g, params = fn(size, seed)
                    add_map(name, fam, seed, confirm, g, params, size=size)
    if pogema_mode == "copy":
        say(f"pogema families copied from legacy products (md5-verified); "
            f"provenance pin pogema=={POGEMA_PIN}")

    # ---- gmazeb (braided mazes)
    for size in MZB_SIZES:
        for btag, beta in MZB_BETAS:
            for seed in MZB_SEEDS:
                name = f"gmazeb{size}-b{btag}-s{seed}"
                g, params = gen_mazeb(size, seed, beta)
                add_map(name, "gmazeb", seed, True, g, params, size=size)

    # ---- gtiles (official city-map tiles)
    tile_sources = {}
    gtiles_status = "generated"
    missing = [c for c in TILE_CITIES if not find_city_map(c)]
    if missing:
        gtiles_status = (f"skipped: city source maps not found locally ({', '.join(missing)}) "
                         f"- fetch them first (run setup/00_download_movingai.sh, then rerun "
                         f"setup/02_gen_maps.py)")
        say(f"SKIP gtiles: {gtiles_status}")
    else:
        for city in TILE_CITIES:
            path = find_city_map(city)
            with open(path, "rb") as f:
                raw = f.read()
            tile_sources[city] = hashlib.md5(raw).hexdigest()
            cg = parse_map_text(raw.decode().replace("\r\n", "\n"), city)
            assert len(cg) == TILE_GRID * TILE and len(cg[0]) == TILE_GRID * TILE, \
                f"{city}: expected {TILE_GRID * TILE}x{TILE_GRID * TILE}"
            kept = 0
            for idx in range(TILE_GRID * TILE_GRID):          # row-major, deterministic
                if kept >= TILE_PER_CITY:
                    break
                ty, tx = divmod(idx, TILE_GRID)
                sub = [row[tx * TILE:(tx + 1) * TILE]
                       for row in cg[ty * TILE:(ty + 1) * TILE]]
                blocked = sum(sum(row) for row in sub)
                density = blocked / (TILE * TILE)
                free = TILE * TILE - blocked
                if not (TILE_DENSITY[0] <= density <= TILE_DENSITY[1]):
                    continue
                comps = sorted(components(sub), key=len, reverse=True)
                if not comps or len(comps[0]) < TILE_LCC_MIN * free:
                    continue
                kept += 1
                name = f"gtiles-{city}-t{idx}"
                add_map(name, "gtiles", None, True, sub,
                        {"city": city, "tile_idx": idx, "tile_row": ty, "tile_col": tx,
                         "density": round(density, 4), "source_md5": tile_sources[city],
                         "rule": TILE_RULE_VERSION})
            say(f"gtiles {city}: kept {kept}/{TILE_GRID * TILE_GRID} tiles "
                f"(rule: density {TILE_DENSITY}, LCC>={TILE_LCC_MIN}*free, first {TILE_PER_CITY})")

    # ---- hard freeze guard: every regenerated file that exists in the legacy MANIFEST
    #      must be byte-identical (published data depends on these bytes).
    if legacy:
        lman = legacy["man"]
        drift = [rel for rel in files
                 if rel in lman["md5"] and md5_of(files[rel]) != lman["md5"][rel]]
        assert not drift, f"FROZEN-BYTES VIOLATION vs legacy v1.0: {drift[:10]}"
        covered = sum(1 for rel in lman["md5"] if rel in files)
        say(f"legacy freeze check: {covered}/{len(lman['md5'])} v1.0 files regenerated "
            f"byte-identical")
        skipped_meta = []
        for name, old in lman["maps"].items():
            if name not in maps_meta:   # not regenerated this run (e.g. gtiles without city src maps)
                skipped_meta.append(name); continue
            new = maps_meta[name]
            for k in ("family", "free", "lcc", "N_low", "N_high", "confirmatory"):
                assert new[k] == old[k], f"{name}: meta drift {k}: {new[k]} != {old[k]}"
        if skipped_meta:
            say(f"NOTE: {len(skipped_meta)} frozen maps not regenerated this run "
                f"(source assets absent, e.g. {skipped_meta[:3]}) - meta-drift check skipped for them")

    manifest = {
        "version": GEN_VERSION,
        "changelog": CHANGELOG,
        "protocol": {
            "scen": "Stern SoCS'19 largest-component random pairing",
            "rho": RHO, "ncap": NCAP, "nfloor": NFLOOR,
            "scens_per_map": SCENS_PER_MAP,
            "legacy_frozen": {"version": "F-gen v1.0", "maps": 130,
                              "scens_per_map": LEGACY_SCENS,
                              "note": "maps + scens 1-4 byte-identical to published v1.0"},
            "seeds": {"main": SEEDS, "pogema": POG_SEEDS, "gmazeb": MZB_SEEDS},
            "n_rule": "N_high=min(1000,max(11,round(0.15*LCC))); "
                      "N_low=max(10,min(1000,N_high-1,round(0.05*LCC))); "
                      "single_N if tiers collapse",
            "gmazeb": {"betas": [b for _, b in MZB_BETAS], "sizes": list(MZB_SIZES),
                       "seeds": MZB_SEEDS,
                       "base": "gmaze recursive backtracker, stream rng_for('map','gmazeb',size,seed)",
                       "braid": "per-wall independent Bernoulli(beta) on interior walls with two "
                                "opposite free 4-neighbours (perfect-maze snapshot, row-major), "
                                "stream rng_for('gmazeb',size,seed,'braid') (beta-independent -> "
                                "b15 loops subset of b30 loops per seed)"},
            "gtiles": {"rule_version": TILE_RULE_VERSION, "cities": list(TILE_CITIES),
                       "tile": TILE, "grid": f"{TILE_GRID}x{TILE_GRID} row-major disjoint",
                       "density_range": list(TILE_DENSITY), "lcc_min_frac": TILE_LCC_MIN,
                       "per_city": TILE_PER_CITY, "source_md5": tile_sources,
                       "status": gtiles_status},
            "env": {"pogema_pin": POGEMA_PIN, "pogema_mode": pogema_mode,
                    "pogema_local": pogema_version()},
            "exclusions": "NONE - no post-hoc exclusions; seed ranges exhaustive; the gtiles "
                          "filter is a preregistered deterministic rule",
            "dist_column": "0 (unused by MAPF-LNS-lineage loaders)"},
        "maps": maps_meta,
        "md5": {rel: md5_of(txt) for rel, txt in sorted(files.items())}}
    return files, manifest, notes

# ---------------------------------------------------------------- write / generate
def write_all(files, manifest):
    written = skipped = 0
    for rel, txt in sorted(files.items()):
        path = os.path.join(OUTDIR, rel)
        want = md5_of(txt)
        if os.path.isfile(path) and md5_file(path) == want:
            skipped += 1
            continue
        if os.path.isfile(path):
            print(f"WARN: {rel} exists with different bytes -> overwriting (md5 "
                  f"{md5_file(path)} -> {want})")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", newline="\n") as f:
            f.write(txt)
        written += 1
    mpath = os.path.join(OUTDIR, "MANIFEST.json")
    os.makedirs(OUTDIR, exist_ok=True)
    with open(mpath, "w", newline="\n") as f:
        json.dump(manifest, f, indent=1, sort_keys=True)
    digest = md5_file(mpath)
    with open(mpath + ".md5", "w", newline="\n") as f:
        f.write(f"{digest}  MANIFEST.json\n")
    return written, skipped

def cmd_generate(pogema_mode, legacy):
    files, manifest, _ = build_all(pogema_mode, legacy)
    written, skipped = write_all(files, manifest)
    fams = {}
    for m in manifest["maps"].values():
        k = f"{m['family']}-{m.get('size', 64)}"
        fams.setdefault(k, []).append(m)
    for k in sorted(fams):
        ms = fams[k]
        print(f"{k}: {len(ms)} maps, LCC frac med={sorted(m['lcc_frac'] for m in ms)[len(ms)//2]:.3f}, "
              f"N_high med={sorted(m['N_high'] for m in ms)[len(ms)//2]}")
    print(f"TOTAL maps={len(manifest['maps'])} files={len(files)} "
          f"(written {written}, idempotent-skip {skipped}) -> {OUTDIR}")

# ---------------------------------------------------------------- check
def cmd_check(pogema_mode, legacy):
    """Full regeneration, byte-diff against installed data/gen_maps + MANIFEST."""
    mpath = os.path.join(OUTDIR, "MANIFEST.json")
    if not os.path.isfile(mpath):
        sys.exit("FATAL --check: no MANIFEST.json in data/gen_maps (run --generate first)")
    with open(mpath) as f:
        man = json.load(f)
    files, _, _ = build_all(pogema_mode, legacy, quiet=True)
    drift, weak, missing = [], [], []
    for rel, want in man["md5"].items():
        path = os.path.join(OUTDIR, rel)
        if not os.path.isfile(path):
            missing.append(rel); continue
        disk = open(path, "rb").read()
        if rel in files:                       # strong: regenerated bytes vs disk bytes
            if files[rel].encode() != disk or md5_of(files[rel]) != want:
                drift.append(rel)
        else:                                  # not regenerable here: disk md5 vs manifest
            weak.append(rel)
            if md5_of(disk) != want:
                drift.append(rel)
    extra = [rel for rel in files if rel not in man["md5"]]
    if drift or missing or extra:
        print(f"CHECK FAILED: drift={drift[:10]} missing={missing[:10]} "
              f"unlisted-regenerated={extra[:10]}")
        sys.exit(1)
    msg = f"check OK: {len(files)} files regenerated byte-identical vs installed tree"
    if weak:
        msg += (f"; {len(weak)} files not regenerable on this machine "
                f"(md5-verified vs MANIFEST only)")
    print(msg)

# ---------------------------------------------------------------- install
def cmd_install():
    mpath = os.path.join(OUTDIR, "MANIFEST.json")
    if not os.path.isfile(mpath):
        sys.exit("FATAL --install: run --generate first")
    with open(mpath) as f:
        man = json.load(f)
    os.makedirs(DATA_MAPS, exist_ok=True)
    os.makedirs(DATA_SCEN, exist_ok=True)
    copied = skipped = 0
    for rel, want in sorted(man["md5"].items()):
        src = os.path.join(OUTDIR, rel)
        base = os.path.basename(rel)
        assert base.startswith("g"), f"{base}: generated files must be g-prefixed"
        dst = os.path.join(DATA_MAPS if rel.startswith("maps/") else DATA_SCEN, base)
        assert md5_file(src) == want, f"{rel}: gen_maps copy drifted from MANIFEST"
        if os.path.isfile(dst):
            if md5_file(dst) == want:
                skipped += 1
                continue
            sys.exit(f"FATAL --install: {dst} exists with DIFFERENT content - refusing to "
                     f"overwrite (official files are never touched; delete manually if this "
                     f"is a stale generated file)")
        shutil.copyfile(src, dst)
        copied += 1
    print(f"install OK: {copied} copied, {skipped} already present -> "
          f"{DATA_MAPS} + {DATA_SCEN}")

# ---------------------------------------------------------------- validate
def cmd_validate():
    """Independent re-checks on the WRITTEN files (parse back from disk)."""
    mpath = os.path.join(OUTDIR, "MANIFEST.json")
    with open(mpath) as f:
        man = json.load(f)
    n_scen = man["protocol"]["scens_per_map"]
    singles = []
    for name, meta in man["maps"].items():
        mp = os.path.join(OUTDIR, "maps", name + ".map")
        g = parse_map_text(open(mp, newline="").read(), name)
        h, w = len(g), len(g[0])
        assert (h, w) == (meta["h"], meta["w"]), f"{name}: dims"
        comps = sorted(components(g), key=len, reverse=True)
        lcc = set(comps[0])
        assert len(comps[0]) == meta["lcc"], f"{name}: LCC mismatch"
        assert sum(len(c) for c in comps) == meta["free"], f"{name}: free mismatch"
        if meta["family"] in CONNECTED_FAMILIES:
            assert len(comps) == 1, f"{name}: disconnected {meta['family']}"
        n_low, n_high, single = design_N(len(lcc))
        assert n_high == meta["N_high"], f"{name}: N_high rule mismatch"
        assert (n_high if single else n_low) == meta["N_low"], f"{name}: N_low rule mismatch"
        assert meta["N_low"] <= NCAP and meta["N_high"] <= NCAP, f"{name}: NCAP violated"
        assert single == bool(meta.get("single_N")), f"{name}: single_N flag mismatch"
        if single:
            singles.append(name)
        else:
            assert meta["N_low"] < meta["N_high"], f"{name}: N_low >= N_high"
        if meta["family"] == "gtiles":
            density = (h * w - meta["free"]) / (h * w)
            assert TILE_DENSITY[0] <= density <= TILE_DENSITY[1], f"{name}: tile density"
            assert len(lcc) >= TILE_LCC_MIN * meta["free"], f"{name}: tile LCC rule"
        for j in range(1, n_scen + 1):
            sp = os.path.join(OUTDIR, "scen-random", f"{name}-random-{j}.scen")
            slines = [l for l in open(sp, newline="").read().split("\n") if l]
            assert slines[0] == "version 1"
            rows_ = [l.split("\t") for l in slines[1:]]
            assert len(rows_) >= meta["N_high"], f"{name} scen{j}: rows {len(rows_)} < N_high"
            starts, goals = set(), set()
            for f9 in rows_:
                assert len(f9) == 9 and f9[1] == name + ".map"
                sx, sy, gx, gy = int(f9[4]), int(f9[5]), int(f9[6]), int(f9[7])
                assert 0 <= sx < w and 0 <= sy < h and 0 <= gx < w and 0 <= gy < h
                assert (sy, sx) in lcc and (gy, gx) in lcc, f"{name} scen{j}: cell not in LCC"
                starts.add((sy, sx)); goals.add((gy, gx))
            assert len(starts) == len(rows_) and len(goals) == len(rows_), "dup start/goal"
            assert not (starts & goals), "start/goal sets overlap"
    # md5 re-check (files on disk == manifest)
    for rel, want in man["md5"].items():
        assert md5_file(os.path.join(OUTDIR, rel)) == want, f"{rel}: md5 drift"
    msg = (f"validate OK: {len(man['maps'])} maps, {len(man['md5'])} files "
           f"({n_scen} scens/map), 0 violations")
    if singles:
        msg += f"; single-N maps: {singles}"
    print(msg)

# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--generate", action="store_true",
                    help="build all families + MANIFEST v2 (default; idempotent)")
    ap.add_argument("--check", action="store_true",
                    help="fully regenerate and byte-diff vs installed data/gen_maps")
    ap.add_argument("--install", action="store_true",
                    help="copy maps+scens into data/maps + data/scen-random (no overwrite)")
    ap.add_argument("--validate", action="store_true",
                    help="parse written files back, re-verify LCC/N/scen invariants + md5s")
    ap.add_argument("--legacy-dir", default=DEFAULT_LEGACY,
                    help="frozen F-gen v1.0 output (byte-freeze guard + pogema copy source)")
    ap.add_argument("--pogema-mode", choices=("auto", "generate", "copy"), default="auto",
                    help="pogema families: regenerate via pogema, or copy from --legacy-dir")
    args = ap.parse_args()

    legacy = load_legacy(args.legacy_dir)
    if legacy is None:
        print(f"NOTE: legacy v1.0 products not found at {args.legacy_dir} - freeze guard "
              f"limited to MANIFEST md5s; pogema families need a local pogema=={POGEMA_PIN}")
    mode = resolve_pogema_mode(args.pogema_mode, legacy)
    if mode == "missing":
        mode = "copy"   # will fail with a clear message inside build_all if truly needed
    pv = pogema_version()
    if mode == "generate" and pv != POGEMA_PIN:
        print(f"WARN: local pogema=={pv} != pinned {POGEMA_PIN}; byte-freeze guard will "
              f"catch any drift" if pv else "")

    if args.check:
        cmd_check(mode, legacy)
    elif args.validate:
        cmd_validate()
    elif args.install:
        cmd_install()
    else:
        cmd_generate(mode, legacy)

if __name__ == "__main__":
    main()
