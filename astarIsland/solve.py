#!/usr/bin/env python3
"""
Astar Island Solver v2 — NM i AI Competition
Strategy: Per-round calibration via nearest-round matching.
  1. Study past rounds → build per-round transition tables + activity signatures
  2. Fixed viewport grid, cycling through seeds (same viewport for all seeds)
  3. Estimate this round's activity from observations → match to closest past round(s)
  4. Predict using matched round's transition tables + Dirichlet blending with observations
"""

import os
import sys
import time
import json
import argparse
import numpy as np
import requests
from pathlib import Path
from collections import defaultdict
from sklearn.ensemble import HistGradientBoostingRegressor, ExtraTreesRegressor

BASE_URL = "https://api.ainm.no"
CACHE_DIR = Path(__file__).parent / ".cache"
PROB_FLOOR = 0.0001

# --- Terrain codes ---
OCEAN, PLAINS, EMPTY = 10, 11, 0
SETTLEMENT, PORT, RUIN, FOREST, MOUNTAIN = 1, 2, 3, 4, 5

N_CLASSES = 6
TERRAIN_TO_CLASS = {
    OCEAN: 0, PLAINS: 0, EMPTY: 0,
    SETTLEMENT: 1, PORT: 2, RUIN: 3, FOREST: 4, MOUNTAIN: 5,
}
STATIC_TERRAINS = {OCEAN, MOUNTAIN}


# ─────────────────────────────────────────────
# API Client
# ─────────────────────────────────────────────

def load_token():
    token = os.environ.get("ASTAR_TOKEN")
    if token:
        return token
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line.startswith("ASTAR_TOKEN="):
                return line.split("=", 1)[1].strip().strip("'\"")
    print("Error: Set ASTAR_TOKEN in environment or in .env file")
    sys.exit(1)


def make_session(token):
    session = requests.Session()
    session.headers["Authorization"] = f"Bearer {token}"
    return session


def api_get(session, path):
    resp = session.get(f"{BASE_URL}{path}")
    resp.raise_for_status()
    return resp.json()


def api_post(session, path, data):
    resp = session.post(f"{BASE_URL}{path}", json=data)
    if resp.status_code == 429:
        print("  Rate limited, waiting 1.5s...")
        time.sleep(1.5)
        return api_post(session, path, data)
    resp.raise_for_status()
    return resp.json()


# ─────────────────────────────────────────────
# Caching
# ─────────────────────────────────────────────

def cache_path(name):
    CACHE_DIR.mkdir(exist_ok=True)
    return CACHE_DIR / f"{name}.json"

def cache_get(name):
    p = cache_path(name)
    return json.loads(p.read_text()) if p.exists() else None

def cache_set(name, data):
    cache_path(name).write_text(json.dumps(data))


# ─────────────────────────────────────────────
# Map Analysis Utilities
# ─────────────────────────────────────────────

def grid_to_np(grid):
    return np.array(grid, dtype=np.int8)


def settlement_positions(initial_state):
    return [(s["x"], s["y"], s.get("has_port", False))
            for s in initial_state.get("settlements", [])
            if s.get("alive", True)]


def distance_to_nearest_settlement(height, width, settlements):
    """Compute Chebyshev (chessboard) distance to nearest settlement."""
    dist = np.full((height, width), 999, dtype=np.int16)
    for sx, sy, _ in settlements:
        for y in range(height):
            for x in range(width):
                d = max(abs(x - sx), abs(y - sy))
                if d < dist[y, x]:
                    dist[y, x] = d
    return dist


def is_coastal_map(grid_np):
    """Compute boolean map of cells adjacent to ocean."""
    h, w = grid_np.shape
    coastal = np.zeros((h, w), dtype=bool)
    ocean_mask = (grid_np == OCEAN)
    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        shifted = np.roll(np.roll(ocean_mask, dy, axis=0), dx, axis=1)
        coastal |= shifted
    # Don't count ocean cells themselves as coastal
    coastal &= ~ocean_mask
    return coastal


def adj_forest_map(grid_np):
    """Count adjacent forest cells for each cell."""
    h, w = grid_np.shape
    forest_mask = (grid_np == FOREST).astype(np.int8)
    count = np.zeros((h, w), dtype=np.int8)
    for dy in [-1, 0, 1]:
        for dx in [-1, 0, 1]:
            if dy == 0 and dx == 0:
                continue
            shifted = np.roll(np.roll(forest_mask, dy, axis=0), dx, axis=1)
            count += shifted
    return count


# ─────────────────────────────────────────────
# Fixed Viewport Grid
# ─────────────────────────────────────────────

def make_viewport_grid(width, height, vp_size=15):
    """
    Generate fixed viewports that cover the map. Always 15x15 (max size)
    to maximize observations per query. Overlaps at edges are fine —
    they give extra samples on those cells.
    For 40x40: positions at x/y ∈ {0, 13, 25} → 9 viewports + 1 center.
    """
    # Compute start positions: spread evenly, always allowing full vp_size
    def make_starts(size):
        starts = [0]
        pos = 0
        while pos + vp_size < size:
            pos = min(pos + vp_size, size - vp_size)
            if pos not in starts:
                starts.append(pos)
        # If gap between last two is too large, add intermediate
        if len(starts) >= 2 and starts[-1] - starts[-2] > vp_size:
            starts.insert(-1, (starts[-2] + starts[-1]) // 2)
        return starts

    starts_x = make_starts(width)
    starts_y = make_starts(height)

    viewports = []
    for vy in starts_y:
        for vx in starts_x:
            vw = min(vp_size, width - vx)
            vh = min(vp_size, height - vy)
            viewports.append((vx, vy, vw, vh))

    # Deduplicate
    viewports = list(dict.fromkeys(viewports))

    # Add center viewport for extra overlap observations
    if len(viewports) < 10:
        cx = max(0, (width - vp_size) // 2)
        cy = max(0, (height - vp_size) // 2)
        center = (cx, cy, min(vp_size, width - cx), min(vp_size, height - cy))
        if center not in viewports:
            viewports.append(center)

    return viewports


# ─────────────────────────────────────────────
# Study Phase — Per-Round Transition Tables
# ─────────────────────────────────────────────

def dist_bucket(d):
    """Bucket distances for grouping."""
    if d <= 4:
        return d
    elif d <= 7:
        return 5
    else:
        return 8


def study_round(session, round_info):
    """Analyze a completed round, building per-round transition table."""
    round_id = round_info["id"]
    rn = round_info["round_number"]

    cached = cache_get(f"study_v2_r{rn}")
    if cached:
        print(f"  Round {rn}: loaded from cache")
        return cached

    detail = api_get(session, f"/astar-island/rounds/{round_id}")
    seeds_count = detail.get("seeds_count", 5)
    initial_states = detail.get("initial_states", [])
    width = detail["map_width"]
    height = detail["map_height"]

    # Collect transitions grouped by (terrain, dist_bucket, coastal)
    groups = defaultdict(lambda: {"probs": [], "count": 0})
    # Also collect raw stats for activity signature
    settl_alive = 0
    settl_total = 0
    expansion_cells = 0
    expansion_total = 0
    forest_stayed = 0
    forest_total = 0
    # For decay rate estimation
    expansion_by_dist = defaultdict(lambda: {"expanded": 0, "total": 0})

    for seed_idx in range(seeds_count):
        if seed_idx >= len(initial_states):
            continue
        try:
            analysis = api_get(session, f"/astar-island/analysis/{round_id}/{seed_idx}")
        except requests.HTTPError:
            continue

        gt = np.array(analysis["ground_truth"])
        init_grid = grid_to_np(initial_states[seed_idx]["grid"])
        settlements = settlement_positions(initial_states[seed_idx])
        dist_map = distance_to_nearest_settlement(height, width, settlements)
        coastal_map = is_coastal_map(init_grid)

        for y in range(height):
            for x in range(width):
                terrain = int(init_grid[y, x])
                if terrain in STATIC_TERRAINS:
                    continue

                d = int(dist_map[y, x])
                db = dist_bucket(d)
                c = bool(coastal_map[y, x])
                key = f"{terrain}_{db}_{c}"

                gt_probs = gt[y, x].tolist()
                groups[key]["probs"].append(gt_probs)
                groups[key]["count"] += 1

                # Activity stats
                if terrain == SETTLEMENT or terrain == PORT:
                    settl_total += 1
                    settl_alive += gt_probs[1] + gt_probs[2]  # P(settlement) + P(port)
                elif terrain in (EMPTY, PLAINS) and d <= 8:
                    expansion_total += 1
                    expansion_cells += gt_probs[1] + gt_probs[2]
                    expansion_by_dist[d]["total"] += 1
                    expansion_by_dist[d]["expanded"] += gt_probs[1] + gt_probs[2]
                elif terrain == FOREST:
                    forest_total += 1
                    forest_stayed += gt_probs[4]

    # Build transition table
    table = {}
    for key, data in groups.items():
        table[key] = np.mean(data["probs"], axis=0).tolist()

    # Compute decay profile (expansion rate at each distance)
    decay = {}
    for d in sorted(expansion_by_dist.keys()):
        data = expansion_by_dist[d]
        if data["total"] > 0:
            decay[str(d)] = data["expanded"] / data["total"]

    # Compute activity signature (including decay steepness)
    d1_val = decay.get("1", 0.0)
    d3_val = decay.get("3", 0.001)
    decay_steepness = d1_val / (d3_val + 0.001)  # high = sharp decay

    sig = {
        "settl_survival": settl_alive / max(settl_total, 1),
        "expansion_rate": expansion_cells / max(expansion_total, 1),
        "forest_survival": forest_stayed / max(forest_total, 1),
        "decay_steepness": decay_steepness,
    }

    result = {
        "round_number": rn,
        "table": table,
        "signature": sig,
        "decay_profile": decay,
        "n_cells": sum(g["count"] for g in groups.values()),
    }
    cache_set(f"study_v2_r{rn}", result)
    print(f"  Round {rn}: {result['n_cells']} cells, "
          f"settl_surv={sig['settl_survival']:.3f} "
          f"expansion={sig['expansion_rate']:.3f} "
          f"forest_surv={sig['forest_survival']:.3f} "
          f"decay_steep={decay_steepness:.1f}")
    return result


def cmd_study(session):
    """Study all completed rounds."""
    rounds = api_get(session, "/astar-island/rounds")
    completed = [r for r in rounds if r["status"] in ("completed", "scoring")]
    if not completed:
        print("No completed rounds")
        return None

    print(f"Studying {len(completed)} round(s)...")
    round_data = []
    for r in completed:
        rd = study_round(session, r)
        if rd and rd["n_cells"] > 0:
            round_data.append(rd)

    # Cache the full model
    cache_set("round_data_v2", round_data)
    print(f"\n{len(round_data)} rounds studied and cached")
    return round_data


# ─────────────────────────────────────────────
# Activity Estimation & Round Matching
# ─────────────────────────────────────────────

def estimate_activity_from_observations(observations, initial_states, seeds_count, height, width):
    """
    Estimate this round's activity signature from simulation observations.
    Compare observed final state with initial state.
    Also estimates per-distance expansion decay profile.
    """
    settl_alive = 0
    settl_total = 0
    expansion = 0
    expansion_total = 0
    forest_stayed = 0
    forest_total = 0
    expansion_by_dist = defaultdict(lambda: {"expanded": 0, "total": 0})

    for seed_idx, obs_list in observations.items():
        if seed_idx >= len(initial_states):
            continue
        init_grid = grid_to_np(initial_states[seed_idx]["grid"])
        settlements = settlement_positions(initial_states[seed_idx])
        dist_map = distance_to_nearest_settlement(height, width, settlements)

        for obs_grid, vx, vy, vw, vh in obs_list:
            for dy in range(vh):
                for dx in range(vw):
                    y, x = vy + dy, vx + dx
                    if y >= height or x >= width:
                        continue
                    init_terrain = int(init_grid[y, x])
                    obs_terrain = obs_grid[dy][dx]
                    obs_cls = TERRAIN_TO_CLASS.get(obs_terrain, 0)
                    d = int(dist_map[y, x])

                    if init_terrain in (SETTLEMENT, PORT):
                        settl_total += 1
                        if obs_cls in (1, 2):  # still settlement or port
                            settl_alive += 1
                    elif init_terrain in (EMPTY, PLAINS) and d <= 8:
                        expansion_total += 1
                        expansion_by_dist[d]["total"] += 1
                        if obs_cls in (1, 2):
                            expansion += 1
                            expansion_by_dist[d]["expanded"] += 1
                    elif init_terrain == FOREST:
                        forest_total += 1
                        if obs_cls == 4:
                            forest_stayed += 1

    # Per-distance decay profile from observations
    decay_profile = {}
    for d in range(10):
        data = expansion_by_dist.get(d, {"expanded": 0, "total": 0})
        if data["total"] > 0:
            decay_profile[d] = data["expanded"] / data["total"]
        else:
            decay_profile[d] = 0

    d1_val = decay_profile.get(1, 0.0)
    d3_val = decay_profile.get(3, 0.001)
    decay_steepness = d1_val / (d3_val + 0.001)

    sig = {
        "settl_survival": settl_alive / max(settl_total, 1),
        "expansion_rate": expansion / max(expansion_total, 1),
        "forest_survival": forest_stayed / max(forest_total, 1),
        "decay_steepness": decay_steepness,
    }
    sig["observed_decay"] = decay_profile

    return sig


def match_rounds(observed_sig, round_data, bandwidth=0.20):
    """
    Gaussian kernel-weighted matching: all rounds contribute,
    weighted by similarity to observed activity signature.
    bandwidth=0.20 tested optimal with decay scaling active.
    Now includes decay_steepness as a 4th matching dimension.
    """
    # Normalize decay_steepness to comparable scale (log scale, capped)
    obs_steep = np.log1p(min(observed_sig.get("decay_steepness", 5.0), 100.0))

    weights = []
    for rd in round_data:
        sig = rd["signature"]
        rd_steep = np.log1p(min(sig.get("decay_steepness", 5.0), 100.0))
        d2 = (
            3.0 * (sig["settl_survival"] - observed_sig["settl_survival"]) ** 2
            + 2.0 * (sig["expansion_rate"] - observed_sig["expansion_rate"]) ** 2
            + 1.0 * (sig["forest_survival"] - observed_sig["forest_survival"]) ** 2
            + 1.5 * ((rd_steep - obs_steep) / 3.0) ** 2
        )
        w = np.exp(-d2 / (2 * bandwidth ** 2))
        weights.append((rd, w))

    total = sum(w for _, w in weights)
    if total < 1e-10:
        return [(rd, 1.0 / len(round_data)) for rd, _ in weights]
    return [(rd, w / total) for rd, w in weights]


def get_matched_prior(matched_rounds, terrain, dist, coastal):
    """Get blended prior from matched rounds' transition tables."""
    db = dist_bucket(dist)
    key = f"{terrain}_{db}_{coastal}"

    blended = np.zeros(N_CLASSES)
    total_weight = 0

    for rd, weight in matched_rounds:
        table = rd["table"]
        if key in table:
            blended += weight * np.array(table[key])
            total_weight += weight
        else:
            # Try without coastal
            for c in [True, False]:
                fallback_key = f"{terrain}_{db}_{c}"
                if fallback_key in table:
                    blended += weight * np.array(table[fallback_key])
                    total_weight += weight
                    break

    if total_weight > 0:
        return blended / total_weight
    # Ultimate fallback
    return np.ones(N_CLASSES) / N_CLASSES


def get_blended_decay(matched_rounds):
    """Compute blended decay profile from matched rounds."""
    blended = {}
    for d in range(10):
        val = 0
        for rd, w in matched_rounds:
            dp = rd.get("decay_profile", {})
            val += w * dp.get(str(d), 0)
        blended[d] = val
    return blended


# ─────────────────────────────────────────────
# ML Ensemble (HGBR)
# ─────────────────────────────────────────────

def compute_features_for_cell(terrain, dist, coastal, adj_forest, adj_mountain,
                               adj_ocean, adj_settl, n_settl_d2, n_settl_d4, n_settl_d6,
                               sig, decay_profile,
                               x_norm=0.0, y_norm=0.0, n_total_settl=0.0,
                               ocean_ratio=0.0, forest_ratio=0.0,
                               dist_to_port=15.0, adj_ruin=0):
    """Compute feature vector for a single cell."""
    # Terrain one-hot (5 dims)
    is_settlement = 1.0 if terrain in (SETTLEMENT, PORT) else 0.0
    is_port = 1.0 if terrain == PORT else 0.0
    is_forest = 1.0 if terrain == FOREST else 0.0
    is_empty = 1.0 if terrain in (EMPTY, PLAINS) else 0.0
    is_ruin = 1.0 if terrain == RUIN else 0.0

    # Spatial (5 dims)
    d_norm = min(dist, 15) / 15.0
    coastal_f = 1.0 if coastal else 0.0
    dist_inv = 1.0 / (1.0 + dist)
    port_d_norm = min(dist_to_port, 15) / 15.0

    # Round-level (7 dims)
    settl_surv = sig.get("settl_survival", 0.3)
    exp_rate = sig.get("expansion_rate", 0.05)
    forest_surv = sig.get("forest_survival", 0.8)
    d1 = decay_profile.get(1, decay_profile.get("1", 0.0))
    d2 = decay_profile.get(2, decay_profile.get("2", 0.0))
    d3 = decay_profile.get(3, decay_profile.get("3", 0.0))
    d4 = decay_profile.get(4, decay_profile.get("4", 0.0))
    d5 = decay_profile.get(5, decay_profile.get("5", 0.0))

    # Derived features (3 dims)
    decay_ratio = d1 / (d2 + 0.001)  # steepness of expansion decay
    # Expected expansion probability at this cell's distance
    dd = min(int(dist), 8)
    expansion_at_d = decay_profile.get(dd, decay_profile.get(str(dd), 0.0))

    return [
        is_settlement, is_port, is_forest, is_empty, is_ruin,
        d_norm, coastal_f, dist_inv, port_d_norm,
        x_norm, y_norm,
        n_settl_d2, n_settl_d4, n_settl_d6,
        adj_forest, adj_mountain, adj_ocean, adj_settl, adj_ruin,
        n_total_settl, ocean_ratio, forest_ratio,
        settl_surv, exp_rate, forest_surv,
        d1, d2, d3, d4, d5,
        decay_ratio, expansion_at_d,
    ]


def compute_cell_maps(init_grid_np, settlements, height, width):
    """Precompute per-cell spatial maps for feature extraction."""
    dist_map = distance_to_nearest_settlement(height, width, settlements)
    coastal_map = is_coastal_map(init_grid_np)
    forest_adj = adj_forest_map(init_grid_np)

    # Adjacent terrain counts
    mountain_mask = (init_grid_np == MOUNTAIN).astype(np.int8)
    ocean_mask = (init_grid_np == OCEAN).astype(np.int8)
    settl_mask = ((init_grid_np == SETTLEMENT) | (init_grid_np == PORT)).astype(np.int8)
    ruin_mask = (init_grid_np == RUIN).astype(np.int8)

    adj_mtn = np.zeros((height, width), dtype=np.int8)
    adj_ocn = np.zeros((height, width), dtype=np.int8)
    adj_stl = np.zeros((height, width), dtype=np.int8)
    adj_ruin = np.zeros((height, width), dtype=np.int8)
    for dy in [-1, 0, 1]:
        for dx in [-1, 0, 1]:
            if dy == 0 and dx == 0:
                continue
            adj_mtn += np.roll(np.roll(mountain_mask, dy, axis=0), dx, axis=1)
            adj_ocn += np.roll(np.roll(ocean_mask, dy, axis=0), dx, axis=1)
            adj_stl += np.roll(np.roll(settl_mask, dy, axis=0), dx, axis=1)
            adj_ruin += np.roll(np.roll(ruin_mask, dy, axis=0), dx, axis=1)

    # Settlement counts within radius 2, 4, 6
    n_d2 = np.zeros((height, width), dtype=np.float32)
    n_d4 = np.zeros((height, width), dtype=np.float32)
    n_d6 = np.zeros((height, width), dtype=np.float32)
    for sx, sy, _ in settlements:
        for y in range(height):
            for x in range(width):
                d = max(abs(x - sx), abs(y - sy))
                if d <= 2:
                    n_d2[y, x] += 1
                if d <= 4:
                    n_d4[y, x] += 1
                if d <= 6:
                    n_d6[y, x] += 1

    # Distance to nearest port
    ports = [(sx, sy) for sx, sy, hp in settlements if hp]
    dist_to_port = np.full((height, width), 999, dtype=np.int16)
    for px, py in ports:
        for y in range(height):
            for x in range(width):
                d = max(abs(x - px), abs(y - py))
                if d < dist_to_port[y, x]:
                    dist_to_port[y, x] = d

    # Map-level stats
    total_cells = height * width
    n_total_settl = float(len(settlements))
    ocean_ratio = float(np.sum(init_grid_np == OCEAN)) / total_cells
    forest_ratio = float(np.sum(init_grid_np == FOREST)) / total_cells

    return (dist_map, coastal_map, forest_adj, adj_mtn, adj_ocn, adj_stl, adj_ruin,
            n_d2, n_d4, n_d6, dist_to_port, n_total_settl, ocean_ratio, forest_ratio)


def build_features_grid(init_grid_np, settlements, height, width, sig, decay_profile):
    """Build feature matrix for all non-static cells."""
    maps = compute_cell_maps(init_grid_np, settlements, height, width)
    (dist_map, coastal_map, forest_adj, adj_mtn, adj_ocn, adj_stl, adj_ruin,
     n_d2, n_d4, n_d6, dist_to_port, n_total_settl, ocean_ratio, forest_ratio) = maps

    features = []
    coords = []
    for y in range(height):
        for x in range(width):
            terrain = int(init_grid_np[y, x])
            if terrain in STATIC_TERRAINS:
                continue
            feat = compute_features_for_cell(
                terrain, int(dist_map[y, x]), bool(coastal_map[y, x]),
                int(forest_adj[y, x]), int(adj_mtn[y, x]),
                int(adj_ocn[y, x]), int(adj_stl[y, x]),
                float(n_d2[y, x]), float(n_d4[y, x]), float(n_d6[y, x]),
                sig, decay_profile,
                x_norm=x / width, y_norm=y / height,
                n_total_settl=n_total_settl, ocean_ratio=ocean_ratio,
                forest_ratio=forest_ratio,
                dist_to_port=int(dist_to_port[y, x]),
                adj_ruin=int(adj_ruin[y, x]),
            )
            features.append(feat)
            coords.append((y, x))

    return np.array(features, dtype=np.float32), coords


def cache_analysis(session, round_id, seed_idx):
    """Cache ground truth analysis data locally to avoid re-fetching."""
    cache_name = f"analysis_{round_id}_{seed_idx}"
    cached = cache_get(cache_name)
    if cached is not None:
        return cached
    try:
        analysis = api_get(session, f"/astar-island/analysis/{round_id}/{seed_idx}")
        cache_set(cache_name, analysis)
        return analysis
    except Exception:
        return None


def cache_round_detail(session, round_id):
    """Cache round detail data locally."""
    cache_name = f"detail_{round_id}"
    cached = cache_get(cache_name)
    if cached is not None:
        return cached
    detail = api_get(session, f"/astar-island/rounds/{round_id}")
    cache_set(cache_name, detail)
    return detail


def build_training_data(session, round_data):
    """Build training features + targets from all studied rounds."""
    cache_file = CACHE_DIR / "ensemble_train.npz"
    # Check if we have a cached version with the right number of rounds
    if cache_file.exists():
        data = np.load(cache_file)
        cached_n_rounds = int(data.get("n_rounds", 0))
        if cached_n_rounds == len(round_data):
            print(f"  Loaded cached training data: {data['X'].shape[0]} samples from {cached_n_rounds} rounds")
            return data["X"], data["y"]

    rounds = api_get(session, "/astar-island/rounds")
    completed = {r["round_number"]: r for r in rounds if r["status"] in ("completed", "scoring")}

    all_X = []
    all_y = []

    for rd in round_data:
        rn = rd["round_number"]
        round_info = completed.get(rn)
        if not round_info:
            continue

        round_id = round_info["id"]
        detail = cache_round_detail(session, round_id)
        initial_states = detail.get("initial_states", [])
        width = detail["map_width"]
        height = detail["map_height"]
        sig = rd["signature"]
        decay = {int(k): v for k, v in rd.get("decay_profile", {}).items()}

        for seed_idx in range(min(detail.get("seeds_count", 5), len(initial_states))):
            analysis = cache_analysis(session, round_id, seed_idx)
            if analysis is None:
                continue

            gt = np.array(analysis["ground_truth"])
            init_grid = grid_to_np(initial_states[seed_idx]["grid"])
            settlements = settlement_positions(initial_states[seed_idx])

            X, coords = build_features_grid(init_grid, settlements, height, width, sig, decay)

            targets = np.array([gt[y, x] for y, x in coords], dtype=np.float32)
            all_X.append(X)
            all_y.append(targets)

        print(f"  R{rn}: processed")

    X = np.vstack(all_X)
    y = np.vstack(all_y)
    CACHE_DIR.mkdir(exist_ok=True)
    np.savez(cache_file, X=X, y=y, n_rounds=len(round_data))
    print(f"  Training data: {X.shape[0]} samples, {X.shape[1]} features")
    return X, y


def train_hgbr_models(X, y):
    """Train one HGBR per class. Returns list of 6 models."""
    models = []
    for c in range(N_CLASSES):
        model = HistGradientBoostingRegressor(
            max_iter=300, max_depth=5, learning_rate=0.05,
            min_samples_leaf=20, random_state=42,
        )
        model.fit(X, y[:, c])
        models.append(model)
    return models


def train_rf_models(X, y):
    """Train one ExtraTrees model per class. Returns list of 6 models."""
    models = []
    for c in range(N_CLASSES):
        model = ExtraTreesRegressor(
            n_estimators=200, max_depth=12, min_samples_leaf=10,
            random_state=42, n_jobs=-1,
        )
        model.fit(X, y[:, c])
        models.append(model)
    return models


def predict_ml(models, X):
    """Predict probabilities using ML models. Returns (n_samples, 6) array."""
    preds = np.column_stack([m.predict(X) for m in models])
    preds = np.maximum(preds, PROB_FLOOR)
    preds /= preds.sum(axis=1, keepdims=True)
    return preds


# ─────────────────────────────────────────────
# Prediction Building
# ─────────────────────────────────────────────

def build_prediction(initial_grid_np, settlements, obs_list, height, width,
                     matched_rounds, prior_strength=15.0, target_decay=None,
                     observed_sig=None, hgbr_models=None, rf_models=None,
                     ensemble_alpha=0.5):
    """
    Build H×W×6 prediction using matched round priors + observations.
    If target_decay is provided, rescale settlement probabilities per distance
    to match the observed expansion decay profile.
    If observed_sig is provided, also rescale settlement survival for initial settlements.
    """
    dist_map = distance_to_nearest_settlement(height, width, settlements)
    coastal_map = is_coastal_map(initial_grid_np)

    # Precompute blended decay for scaling
    blended_decay_cache = get_blended_decay(matched_rounds) if target_decay else {}

    # Precompute settlement survival scale factor
    surv_scale = 1.0
    if observed_sig:
        blended_surv = sum(w * rd["signature"]["settl_survival"]
                           for rd, w in matched_rounds)
        obs_surv = observed_sig.get("settl_survival", blended_surv)
        if blended_surv > 0.01:
            surv_scale = max(0.2, min(obs_surv / blended_surv, 3.0))

        # Forest survival scaling tested but net neutral — omitted

    # Count observations per cell
    obs_counts = np.zeros((height, width, N_CLASSES), dtype=np.float64)
    obs_total = np.zeros((height, width), dtype=np.float64)

    for obs_grid, vx, vy, vw, vh in obs_list:
        for dy in range(vh):
            for dx in range(vw):
                y, x = vy + dy, vx + dx
                if y >= height or x >= width:
                    continue
                cell = obs_grid[dy][dx]
                cls = TERRAIN_TO_CLASS.get(cell, 0)
                obs_counts[y, x, cls] += 1.0
                obs_total[y, x] += 1.0

    prediction = np.zeros((height, width, N_CLASSES), dtype=np.float64)

    for y in range(height):
        for x in range(width):
            terrain = int(initial_grid_np[y, x])

            if terrain == OCEAN:
                prediction[y, x, 0] = 1.0
            elif terrain == MOUNTAIN:
                prediction[y, x, 5] = 1.0
            else:
                d = int(dist_map[y, x])
                c = bool(coastal_map[y, x])
                prior = get_matched_prior(matched_rounds, terrain, d, c)

                # Rescale settlement survival for existing settlements
                if terrain in (SETTLEMENT, PORT) and abs(surv_scale - 1.0) > 0.05:
                    prior = prior.copy()
                    settl_mass = prior[1] + prior[2]
                    if settl_mass > 0.01:
                        new_mass = min(settl_mass * surv_scale, 0.99)
                        factor = new_mass / settl_mass
                        prior[1] *= factor
                        prior[2] *= factor
                        prior = np.maximum(prior, 0)
                        prior /= prior.sum()

                # Rescale expansion probability based on decay profile
                if target_decay and d >= 1:
                    bd = blended_decay_cache.get(d, 0) if blended_decay_cache else 0
                    td = target_decay.get(d, target_decay.get(str(d), bd))
                    if bd > 0.005:
                        scale = np.clip(td / bd, 0.1, 5.0)
                    else:
                        scale = 1.0
                    settl_mass = prior[1] + prior[2]
                    if settl_mass > 0.001 and abs(scale - 1.0) > 0.05:
                        prior = prior.copy()
                        new_mass = min(settl_mass * scale, 0.9)
                        factor = new_mass / settl_mass
                        prior[1] *= factor
                        prior[2] *= factor
                        prior = np.maximum(prior, 0)
                        prior /= prior.sum()

                # Store lookup prior (before Dirichlet), apply Dirichlet later
                prediction[y, x] = prior

    # Ensemble: blend lookup prediction with ML models
    if (hgbr_models is not None or rf_models is not None) and observed_sig is not None:
        decay_for_feat = target_decay if target_decay else {}
        sig_for_feat = observed_sig if observed_sig else {"settl_survival": 0.3, "expansion_rate": 0.05, "forest_survival": 0.8}
        X, coords = build_features_grid(initial_grid_np, settlements, height, width, sig_for_feat, decay_for_feat)
        if len(X) > 0:
            if hgbr_models is not None and rf_models is not None:
                # Triple ensemble: 0.4 lookup + 0.3 HGBR + 0.3 RF
                hgbr_preds = predict_ml(hgbr_models, X)
                rf_preds = predict_ml(rf_models, X)
                for i, (y, x) in enumerate(coords):
                    prediction[y, x] = (0.4 * prediction[y, x]
                                        + 0.3 * hgbr_preds[i]
                                        + 0.3 * rf_preds[i])
            elif hgbr_models is not None:
                # Dual ensemble: alpha * lookup + (1-alpha) * HGBR
                hgbr_preds = predict_ml(hgbr_models, X)
                for i, (y, x) in enumerate(coords):
                    prediction[y, x] = (ensemble_alpha * prediction[y, x]
                                        + (1 - ensemble_alpha) * hgbr_preds[i])

    # ── Observation-based calibration ──
    # Compare model predictions on observed cells vs observations,
    # then adjust all predictions to correct systematic bias
    has_obs = obs_total > 0
    n_obs_cells = int(has_obs.sum())
    if n_obs_cells >= 20:
        # Compute observed class frequencies on dynamic observed cells
        dynamic_obs_mask = has_obs & ~np.isin(initial_grid_np, list(STATIC_TERRAINS))
        n_dynamic_obs = int(dynamic_obs_mask.sum())
        if n_dynamic_obs >= 20:
            pred_on_obs = prediction[dynamic_obs_mask]  # (n, 6)
            obs_freq = (obs_counts[dynamic_obs_mask] /
                        obs_total[dynamic_obs_mask, np.newaxis])  # (n, 6)

            # Per-class mean: predicted vs observed
            pred_mean = pred_on_obs.mean(axis=0)
            obs_mean = obs_freq.mean(axis=0)

            # Compute calibration factors for classes 1-4 (skip 0=empty, 5=mountain)
            cal_factors = np.ones(N_CLASSES)
            for c in [1, 2, 3, 4]:
                if pred_mean[c] > 0.002:
                    cal_factors[c] = np.clip(obs_mean[c] / pred_mean[c], 0.5, 2.0)

            # Apply calibration to all dynamic cells
            dynamic_mask = ~np.isin(initial_grid_np, list(STATIC_TERRAINS))
            for y in range(height):
                for x in range(width):
                    if not dynamic_mask[y, x]:
                        continue
                    prediction[y, x, 1:5] *= cal_factors[1:5]
                    prediction[y, x] = np.maximum(prediction[y, x], PROB_FLOOR)
                    prediction[y, x] /= prediction[y, x].sum()

    # ── Dirichlet updating with observations ──
    for y in range(height):
        for x in range(width):
            if obs_total[y, x] > 0 and initial_grid_np[y, x] not in STATIC_TERRAINS:
                alpha = prediction[y, x] * prior_strength
                posterior = alpha + obs_counts[y, x]
                prediction[y, x] = posterior / posterior.sum()

    # Apply probability floor and renormalize
    prediction = np.maximum(prediction, PROB_FLOOR)
    prediction /= prediction.sum(axis=-1, keepdims=True)
    return prediction


# ─────────────────────────────────────────────
# Main Commands
# ─────────────────────────────────────────────

def cmd_info(session):
    rounds = api_get(session, "/astar-island/rounds")
    print(f"{'#':<4} {'Status':<12} {'Date':<12} {'Size':<8}")
    print("-" * 40)
    for r in rounds:
        print(f"{r['round_number']:<4} {r['status']:<12} {r.get('event_date', '?'):<12} "
              f"{r['map_width']}x{r['map_height']}")
    try:
        budget = api_get(session, "/astar-island/budget")
        print(f"\nBudget: {budget['queries_used']}/{budget['queries_max']}")
    except Exception:
        print("\nNo active round or not on a team")
    try:
        my_rounds = api_get(session, "/astar-island/my-rounds")
        scored = [r for r in my_rounds if r.get("round_score") is not None]
        if scored:
            print(f"\nScored rounds:")
            for r in scored:
                print(f"  R{r['round_number']}: score={r['round_score']:.2f}, "
                      f"rank={r.get('rank')}/{r.get('total_teams')}")
    except Exception:
        pass


def cmd_solve(session, round_data=None, dry_run=False):
    """Solve the active round with per-round calibration."""
    rounds = api_get(session, "/astar-island/rounds")
    active = next((r for r in rounds if r["status"] == "active"), None)
    if not active:
        print("No active round!")
        return

    round_id = active["id"]
    width = active["map_width"]
    height = active["map_height"]
    print(f"Round {active['round_number']}: {width}x{height}, "
          f"closes {active.get('closes_at', '?')}")

    detail = api_get(session, f"/astar-island/rounds/{round_id}")
    seeds_count = detail.get("seeds_count", 5)
    initial_states = detail.get("initial_states", [])

    budget = api_get(session, "/astar-island/budget")
    queries_remaining = budget["queries_max"] - budget["queries_used"]
    print(f"Budget: {budget['queries_used']}/{budget['queries_max']} ({queries_remaining} remaining)")

    if not round_data:
        round_data = cache_get("round_data_v2")
        if round_data:
            print(f"Loaded {len(round_data)} round profiles from cache")
        else:
            print("WARNING: No round data! Run --study first!")
            return

    # ── Fixed viewport grid ──
    viewports = make_viewport_grid(width, height)
    n_vp = len(viewports)
    queries_available = queries_remaining
    # How many viewport batches can we afford? Each batch = seeds_count queries
    max_batches = queries_available // seeds_count
    n_batches = min(max_batches, n_vp)
    vps_to_use = viewports[:n_batches]
    # Extra queries for re-observation of remaining viewports
    leftover = queries_available - n_batches * seeds_count

    print(f"Viewport grid: {n_vp} viewports, using {n_batches} batches "
          f"({n_batches * seeds_count} queries) + {leftover} extra")
    for i, (vx, vy, vw, vh) in enumerate(vps_to_use):
        print(f"  VP{i}: ({vx},{vy}) {vw}x{vh}")

    if dry_run:
        print("(dry run — not executing)")
        # Show what matching would look like with no observations
        return

    # ── Execute queries: cycle through seeds per viewport ──
    all_observations = {i: [] for i in range(seeds_count)}
    query_count = budget["queries_used"]

    for vp_idx, (vx, vy, vw, vh) in enumerate(vps_to_use):
        print(f"\n── VP{vp_idx}: ({vx},{vy}) {vw}x{vh} ──")
        for seed_idx in range(seeds_count):
            result = api_post(session, "/astar-island/simulate", {
                "round_id": round_id,
                "seed_index": seed_idx,
                "viewport_x": vx, "viewport_y": vy,
                "viewport_w": vw, "viewport_h": vh,
            })
            vp = result["viewport"]
            all_observations[seed_idx].append(
                (result["grid"], vp["x"], vp["y"], vp["w"], vp["h"])
            )
            query_count = result["queries_used"]
            print(f"  seed {seed_idx}: ({vp['x']},{vp['y']}) {vp['w']}x{vp['h']} "
                  f"[{query_count}/{result['queries_max']}]")
            time.sleep(0.22)

        # After each batch, show activity estimate
        if (vp_idx + 1) % 2 == 0 or vp_idx == n_batches - 1:
            sig = estimate_activity_from_observations(
                all_observations, initial_states, seeds_count, height, width)
            matched = match_rounds(sig, round_data)
            match_str = ", ".join(
                f"R{rd['round_number']}({w:.2f})" for rd, w in matched)
            print(f"  Activity: surv={sig['settl_survival']:.3f} "
                  f"exp={sig['expansion_rate']:.3f} "
                  f"forest={sig['forest_survival']:.3f}")
            print(f"  Matched: {match_str}")

    # ── Use leftover queries for extra observations ──
    if leftover > 0:
        print(f"\n── {leftover} extra queries ──")
        # Distribute across seeds, re-observing first viewports (most dynamic)
        extra_vp_idx = 0
        for q in range(leftover):
            seed_idx = q % seeds_count
            vx, vy, vw, vh = vps_to_use[extra_vp_idx % len(vps_to_use)]
            result = api_post(session, "/astar-island/simulate", {
                "round_id": round_id,
                "seed_index": seed_idx,
                "viewport_x": vx, "viewport_y": vy,
                "viewport_w": vw, "viewport_h": vh,
            })
            vp = result["viewport"]
            all_observations[seed_idx].append(
                (result["grid"], vp["x"], vp["y"], vp["w"], vp["h"])
            )
            query_count = result["queries_used"]
            print(f"  extra: seed {seed_idx} VP{extra_vp_idx % len(vps_to_use)} "
                  f"[{query_count}/{result['queries_max']}]")
            if (q + 1) % seeds_count == 0:
                extra_vp_idx += 1
            time.sleep(0.22)

    # ── Cache observations for potential re-submission ──
    obs_cache = {str(k): v for k, v in all_observations.items()}
    cache_set(f"obs_r{active['round_number']}", obs_cache)

    # ── Final activity estimation & round matching ──
    sig = estimate_activity_from_observations(
        all_observations, initial_states, seeds_count, height, width)
    matched = match_rounds(sig, round_data)
    observed_decay = sig.get("observed_decay", None)
    print(f"\nFinal activity: surv={sig['settl_survival']:.3f} "
          f"exp={sig['expansion_rate']:.3f} "
          f"forest={sig['forest_survival']:.3f}")
    if observed_decay:
        decay_str = " ".join(f"d{d}={v:.3f}" for d, v in sorted(observed_decay.items()) if d <= 6)
        print(f"Observed decay: {decay_str}")
    # Show top contributing rounds
    top_matched = sorted(matched, key=lambda x: -x[1])[:5]
    print(f"Top matches: "
          + ", ".join(f"R{rd['round_number']}({w:.2f})" for rd, w in top_matched))

    # ── Train ML ensemble ──
    print("\nTraining ML ensemble...")
    X_train, y_train = build_training_data(session, round_data)
    hgbr_models = train_hgbr_models(X_train, y_train)
    print(f"  Trained {len(hgbr_models)} HGBR models on {X_train.shape[0]} samples")
    rf_models = train_rf_models(X_train, y_train)
    print(f"  Trained {len(rf_models)} RF models")

    # ── Build & submit predictions ──
    for seed_idx in range(seeds_count):
        init_grid = grid_to_np(initial_states[seed_idx]["grid"])
        settlements = settlement_positions(initial_states[seed_idx])

        pred = build_prediction(
            init_grid, settlements,
            all_observations[seed_idx],
            height, width, matched,
            target_decay=observed_decay,
            observed_sig=sig,
            hgbr_models=hgbr_models,
            rf_models=rf_models,
        )

        # Verify
        assert pred.shape == (height, width, N_CLASSES)
        sums = pred.sum(axis=-1)
        assert np.allclose(sums, 1.0, atol=0.02)
        assert (pred >= 0).all()

        print(f"\nSeed {seed_idx}: submitting... ", end="")
        result = api_post(session, "/astar-island/submit", {
            "round_id": round_id,
            "seed_index": seed_idx,
            "prediction": pred.tolist(),
        })
        print(f"{result.get('status', result)}")
        time.sleep(0.5)

    print("\nDone!")
    budget = api_get(session, "/astar-island/budget")
    print(f"Final budget: {budget['queries_used']}/{budget['queries_max']}")


def compute_score(prediction, ground_truth):
    """Compute the entropy-weighted KL divergence score (same as competition)."""
    H, W, C = ground_truth.shape
    pred = np.maximum(prediction, 1e-10)
    gt = np.maximum(ground_truth, 1e-10)

    # Entropy per cell
    entropy = -np.sum(gt * np.log(gt + 1e-15), axis=-1)
    # KL divergence per cell
    kl = np.sum(gt * np.log(gt / pred + 1e-15), axis=-1)

    # Weighted average (only cells with entropy > 0)
    mask = entropy > 0.001
    if mask.sum() == 0:
        return 100.0
    weighted_kl = np.sum(entropy[mask] * kl[mask]) / np.sum(entropy[mask])
    score = max(0, min(100, 100 * np.exp(-3 * weighted_kl)))
    return score


def cmd_backtest(session, round_data=None):
    """
    Backtest: for each past round, pretend we don't know it,
    use the other rounds' data to predict, and compute score.
    Uses ground truth signatures (perfect activity estimation) as upper bound.
    """
    if not round_data:
        round_data = cache_get("round_data_v2")
    if not round_data:
        print("No round data. Run --study first.")
        return

    rounds = api_get(session, "/astar-island/rounds")
    completed = {r["round_number"]: r for r in rounds
                 if r["status"] in ("completed", "scoring")}

    # Pre-build full training data for LOO HGBR
    print("Building training data for LOO ensemble backtest...")
    full_X, full_y = build_training_data(session, round_data)
    # Need per-round sample counts to split for LOO
    round_sample_counts = {}
    sample_idx = 0
    for rd in round_data:
        rn = rd["round_number"]
        ri = completed.get(rn)
        if not ri:
            round_sample_counts[rn] = 0
            continue
        rid = ri["id"]
        det = cache_round_detail(session, rid)
        istates = det.get("initial_states", [])
        h, w = det["map_height"], det["map_width"]
        count = 0
        for si in range(min(det.get("seeds_count", 5), len(istates))):
            ig = grid_to_np(istates[si]["grid"])
            n_dynamic = int(np.sum(~np.isin(ig, list(STATIC_TERRAINS))))
            count += n_dynamic
        round_sample_counts[rn] = count

    # Build index ranges per round
    round_ranges = {}
    idx = 0
    for rd in round_data:
        rn = rd["round_number"]
        n = round_sample_counts[rn]
        round_ranges[rn] = (idx, idx + n)
        idx += n

    print(f"Backtesting {len(round_data)} rounds (leave-one-out)...\n")
    print(f"{'Round':<8} {'Score':<8} {'Matched':<30} {'Activity'}")
    print("-" * 80)

    all_scores = []

    for test_rd in round_data:
        rn = test_rd["round_number"]
        # Use all OTHER rounds as training
        train_data = [rd for rd in round_data if rd["round_number"] != rn]

        # Use this round's TRUE signature for matching (best case)
        sig = test_rd["signature"]
        matched = match_rounds(sig, train_data)

        # Train ML models on all rounds except this one
        start, end = round_ranges[rn]
        mask = np.ones(len(full_X), dtype=bool)
        mask[start:end] = False
        X_train_loo = full_X[mask]
        y_train_loo = full_y[mask]
        hgbr_models = train_hgbr_models(X_train_loo, y_train_loo)
        rf_models_loo = train_rf_models(X_train_loo, y_train_loo)

        # Get ground truth and initial states
        round_info = completed.get(rn)
        if not round_info:
            continue
        round_id = round_info["id"]
        detail = cache_round_detail(session, round_id)
        initial_states = detail.get("initial_states", [])
        width = detail["map_width"]
        height = detail["map_height"]

        # Use this round's true decay profile for rescaling
        target_decay = {int(k): v for k, v in test_rd.get("decay_profile", {}).items()}

        seed_scores = []
        for seed_idx in range(min(detail.get("seeds_count", 5), len(initial_states))):
            analysis = cache_analysis(session, round_id, seed_idx)
            if analysis is None:
                continue

            gt = np.array(analysis["ground_truth"])
            init_grid = grid_to_np(initial_states[seed_idx]["grid"])
            settlements = settlement_positions(initial_states[seed_idx])

            # Build prediction using triple ensemble (lookup + HGBR + RF)
            pred = build_prediction(
                init_grid, settlements, [],
                height, width, matched,
                target_decay=target_decay,
                observed_sig=sig,
                hgbr_models=hgbr_models,
                rf_models=rf_models_loo,
            )
            score = compute_score(pred, gt)
            seed_scores.append(score)

        if seed_scores:
            avg = np.mean(seed_scores)
            all_scores.append(avg)
            match_str = ", ".join(f"R{rd['round_number']}" for rd, _ in matched)
            print(f"R{rn:<6} {avg:<8.2f} {match_str:<30} "
                  f"surv={sig['settl_survival']:.2f} "
                  f"exp={sig['expansion_rate']:.2f} "
                  f"for={sig['forest_survival']:.2f}")

    if all_scores:
        print(f"\nMean backtest score: {np.mean(all_scores):.2f} "
              f"(min={np.min(all_scores):.2f}, max={np.max(all_scores):.2f})")


def cmd_scores(session):
    my_rounds = api_get(session, "/astar-island/my-rounds")
    for r in my_rounds:
        score = r.get("round_score")
        if score is not None:
            seed_scores = r.get("seed_scores", [])
            ss = ", ".join(f"{s:.1f}" for s in seed_scores) if seed_scores else "?"
            print(f"R{r['round_number']}: score={score:.2f}, "
                  f"rank={r.get('rank')}/{r.get('total_teams')}, seeds=[{ss}]")
        else:
            submitted = r.get("seeds_submitted", 0)
            if submitted > 0:
                print(f"R{r['round_number']}: submitted {submitted}/5, awaiting score")


# ─────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Astar Island Solver v2")
    parser.add_argument("--info", action="store_true", help="Show rounds and scores")
    parser.add_argument("--study", action="store_true", help="Study past rounds")
    parser.add_argument("--solve", action="store_true", help="Solve active round")
    parser.add_argument("--dry-run", action="store_true", help="Show plan without executing")
    parser.add_argument("--scores", action="store_true", help="Show scores")
    parser.add_argument("--all", action="store_true", help="Study + solve")
    parser.add_argument("--backtest", action="store_true", help="Backtest on past rounds")
    args = parser.parse_args()

    if not any([args.info, args.study, args.solve, args.dry_run, args.scores, args.all, args.backtest]):
        parser.print_help()
        return

    token = load_token()
    session = make_session(token)

    if args.info:
        cmd_info(session)
        return
    if args.scores:
        cmd_scores(session)
        return
    if args.backtest:
        cmd_backtest(session)
        return

    round_data = None
    if args.study or args.all:
        round_data = cmd_study(session)

    if args.solve or args.all:
        cmd_solve(session, round_data=round_data)
    elif args.dry_run:
        cmd_solve(session, round_data=round_data, dry_run=True)


if __name__ == "__main__":
    main()
