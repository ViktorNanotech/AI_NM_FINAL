"""
Fetch completed rounds and compute real transition rates from ground truth data.
Run this once to calibrate the prior probabilities in astar_solution.py.
"""
import requests
import numpy as np
from collections import defaultdict

BASE_URL = "https://api.ainm.no"
TOKEN = open(r"c:\aicode\NmiAi\token.txt").read().strip()

# Terrain codes
OCEAN = 10; PLAINS = 11; EMPTY = 0
SETTLEMENT = 1; PORT = 2; RUIN = 3; FOREST = 4; MOUNTAIN = 5

# Class names for printing
CLASS_NAMES = ["Empty", "Settlement", "Port", "Ruin", "Forest", "Mountain"]

session = requests.Session()
session.headers["Authorization"] = f"Bearer {TOKEN}"


def get_completed_rounds():
    rounds = session.get(f"{BASE_URL}/astar-island/rounds").json()
    return [r for r in rounds if r["status"] in ("completed", "scoring")]


def get_analysis(round_id, seed_index):
    resp = session.get(f"{BASE_URL}/astar-island/analysis/{round_id}/{seed_index}")
    if resp.status_code != 200:
        return None
    return resp.json()


def get_round_details(round_id):
    return session.get(f"{BASE_URL}/astar-island/rounds/{round_id}").json()


def compute_hop_distance(initial_grid, settlements, width, height, influence_radius=2):
    """Same wavefront BFS as in astar_solution.py."""
    from collections import deque
    hop = {}
    queue = deque()
    for s in settlements:
        pos = (s["x"], s["y"])
        hop[pos] = 0
        queue.append((pos, 0, s.get("has_port", False)))

    while queue:
        (x, y), h, is_port = queue.popleft()
        for dy in range(-influence_radius, influence_radius + 1):
            for dx in range(-influence_radius, influence_radius + 1):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if not (0 <= nx < width and 0 <= ny < height):
                    continue
                t = initial_grid[ny][nx]
                if t in (OCEAN, MOUNTAIN):
                    continue
                if (nx, ny) not in hop:
                    coastal = any(
                        0 <= nx+ddx < width and 0 <= ny+ddy < height
                        and initial_grid[ny+ddy][nx+ddx] == OCEAN
                        for ddx, ddy in [(-1,0),(1,0),(0,-1),(0,1)]
                    )
                    hop[(nx, ny)] = h + 1
                    queue.append(((nx, ny), h + 1, coastal))

        if is_port:
            ocean_seen = set()
            oq = deque()
            for ddx, ddy in [(-1,0),(1,0),(0,-1),(0,1)]:
                nx, ny = x + ddx, y + ddy
                if 0 <= nx < width and 0 <= ny < height and initial_grid[ny][nx] == OCEAN:
                    if (nx, ny) not in ocean_seen:
                        ocean_seen.add((nx, ny))
                        oq.append((nx, ny))
            while oq:
                ox, oy = oq.popleft()
                for ddx, ddy in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nx, ny = ox + ddx, oy + ddy
                    if not (0 <= nx < width and 0 <= ny < height):
                        continue
                    t = initial_grid[ny][nx]
                    if t == OCEAN and (nx, ny) not in ocean_seen:
                        ocean_seen.add((nx, ny))
                        oq.append((nx, ny))
                    elif t not in (OCEAN, MOUNTAIN) and (nx, ny) not in hop:
                        hop[(nx, ny)] = h + 1
                        queue.append(((nx, ny), h + 1, True))
    return hop


def is_coastal(x, y, grid, width, height):
    for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
        nx, ny = x+dx, y+dy
        if 0 <= nx < width and 0 <= ny < height and grid[ny][nx] == OCEAN:
            return True
    return False


def analyse():
    rounds = get_completed_rounds()
    if not rounds:
        print("No completed rounds found.")
        return

    print(f"Found {len(rounds)} completed round(s).\n")

    # Accumulators: initial_terrain -> list of ground truth distributions
    by_terrain = defaultdict(list)          # terrain_code -> [6-vec, ...]
    by_hop = defaultdict(list)              # (terrain, hop) -> [6-vec, ...]
    by_hop_coastal = defaultdict(list)      # (terrain, hop, is_coastal) -> [6-vec, ...]

    for r in rounds:
        round_id = r["id"]
        print(f"Round {r['round_number']} ({round_id[:8]}...)")
        detail = get_round_details(round_id)
        W, H = detail["map_width"], detail["map_height"]
        seeds_count = detail["seeds_count"]
        initial_states = detail["initial_states"]

        for seed_idx in range(seeds_count):
            analysis = get_analysis(round_id, seed_idx)
            if analysis is None:
                print(f"  Seed {seed_idx}: no analysis available, skipping.")
                continue

            ground_truth = np.array(analysis["ground_truth"])  # H x W x 6
            initial_grid = analysis.get("initial_grid") or initial_states[seed_idx]["grid"]
            settlements  = initial_states[seed_idx]["settlements"]

            hop_dist = compute_hop_distance(initial_grid, settlements, W, H)

            for y in range(H):
                for x in range(W):
                    terrain = initial_grid[y][x]
                    gt = ground_truth[y, x]  # 6-vec

                    # Skip near-static cells (low entropy — not interesting)
                    entropy = -np.sum(gt * np.log(gt + 1e-9))
                    if entropy < 0.05:
                        continue

                    hop = hop_dist.get((x, y), 99)
                    coast = is_coastal(x, y, initial_grid, W, H)

                    by_terrain[terrain].append(gt)
                    by_hop[(terrain, hop)].append(gt)
                    by_hop_coastal[(terrain, hop, coast)].append(gt)

            print(f"  Seed {seed_idx}: analysed {W}x{H} grid")

    print("\n" + "="*60)
    print("TRANSITION RATES BY INITIAL TERRAIN TYPE")
    print("="*60)
    terrain_names = {
        OCEAN: "Ocean", PLAINS: "Plains", EMPTY: "Empty",
        SETTLEMENT: "Settlement", PORT: "Port",
        RUIN: "Ruin", FOREST: "Forest", MOUNTAIN: "Mountain",
    }
    for terrain, vecs in sorted(by_terrain.items()):
        mean = np.mean(vecs, axis=0)
        name = terrain_names.get(terrain, str(terrain))
        print(f"\n{name} (n={len(vecs)} dynamic cells):")
        for i, cls in enumerate(CLASS_NAMES):
            print(f"  {cls:<12}: {mean[i]:.3f}")

    print("\n" + "="*60)
    print("TRANSITION RATES BY HOP DISTANCE (Plains/Empty only)")
    print("="*60)
    for hop in range(5):
        for terrain in (PLAINS, EMPTY):
            key = (terrain, hop)
            vecs = by_hop.get(key, [])
            if not vecs:
                continue
            mean = np.mean(vecs, axis=0)
            name = terrain_names.get(terrain, str(terrain))
            print(f"\n{name} hop={hop} (n={len(vecs)}):")
            for i, cls in enumerate(CLASS_NAMES):
                print(f"  {cls:<12}: {mean[i]:.3f}")

    print("\n" + "="*60)
    print("COASTAL vs INLAND (Plains/Empty, hop=1)")
    print("="*60)
    for coast in (True, False):
        for terrain in (PLAINS, EMPTY):
            key = (terrain, 1, coast)
            vecs = by_hop_coastal.get(key, [])
            if not vecs:
                continue
            mean = np.mean(vecs, axis=0)
            label = "Coastal" if coast else "Inland"
            name = terrain_names.get(terrain, str(terrain))
            print(f"\n{name} hop=1 {label} (n={len(vecs)}):")
            for i, cls in enumerate(CLASS_NAMES):
                print(f"  {cls:<12}: {mean[i]:.3f}")


if __name__ == "__main__":
    analyse()
