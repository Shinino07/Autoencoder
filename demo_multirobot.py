import os

import numpy as np
import rerun as rr
import rerun.blueprint as rrb

import config
from gslib import robots as R
from gslib.load import load_gaussians

SPARSE = "data/colmap_truck"
OUT = os.path.join(config.OUT_DIR, "demo_multirobot.rrd")

N_SHOW = 250_000
OPACITY_MIN = 0.05
DEPTH_FACTOR = 6.0
SYNC_EVERY = 20
BYTES_ORIG = 248
BYTES_COMP = 36

ROBOTS = [("red", [230, 57, 70]), ("blue", [29, 120, 230]), ("yellow", [240, 180, 20])]

cam = R.load_cameras(SPARSE, DEPTH_FACTOR)
up = cam["up"]
print(f"{len(cam['intr'])} cameras. Estimated 'up' in COLMAP coordinates: "
      f"({up[0]:+.2f}, {up[1]:+.2f}, {up[2]:+.2f})  -> dominant axis: {'xyz'[np.argmax(np.abs(up))]}")

raw, g = load_gaussians(config.PLY_MAIN)
keep = np.flatnonzero(g["opacity"] >= OPACITY_MIN)
rng = np.random.default_rng(0)
idx = rng.choice(keep, size=min(N_SHOW, len(keep)), replace=False)
pts_world = g["xyz"][idx].astype(np.float64)
pts = R.to_display(cam, pts_world)
rgb = (g["rgb"][idx] * 255).astype(np.uint8)
scale_comp = len(keep) / len(idx)
scale_orig = g["N"] / len(idx)
print(f"{g['N']:,} Gaussians, {len(keep):,} after pruning, {len(idx):,} drawn. "
      f"Max view depth {cam['max_depth']:.1f}")

arcs, T = R.split_paths(len(cam["intr"]), len(ROBOTS))
first_seen = R.first_seen(cam, arcs, T, pts_world)
global_first = first_seen.min(0)
owner = first_seen.argmin(0)
seen_any = global_first < R.NEVER
print("Coverage of the drawn map: " + ", ".join(
    f"{name} {100 * np.mean(first_seen[r] < R.NEVER):.0f}%" for r, (name, _) in enumerate(ROBOTS))
    + f", all together {100 * seen_any.mean():.0f}%")

rr.init("multirobot_gaussian_map")
blueprint = rrb.Blueprint(rrb.Horizontal(
    rrb.Spatial3DView(origin="world", name="Map"),
    rrb.Vertical(
        rrb.TimeSeriesView(origin="stats/mb_sent", name="MB sent (original vs compressed)"),
        rrb.TimeSeriesView(origin="stats/known", name="% of map known by each robot"),
        rrb.TextDocumentView(origin="info", name="About"),
    ), column_shares=[3, 2]))
rr.save(OUT, default_blueprint=blueprint)

rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
rr.log("info", rr.TextDocument(
    "## Three robots sharing a Gaussian map\n"
    "- Each robot walks one third of the camera ring around the truck.\n"
    "- A Gaussian appears when a robot first sees it, colored by that robot.\n"
    f"- Every {SYNC_EVERY} steps robots send their new map parts to each other (arrows).\n"
    f"- Bandwidth: original {BYTES_ORIG} B/Gaussian vs compressed {BYTES_COMP} B/Gaussian "
    f"(pruning + AE k=8 for SH).\n"
    "- Last step shows the merged map in true color.\n\n"
    "*Simplified: no real SLAM, occlusion ignored, bandwidth scaled from the drawn sample.*",
    media_type=rr.MediaType.MARKDOWN), static=True)

know = np.zeros(first_seen.shape, bool)
last_sync = -1
mb_orig = mb_comp = 0.0
total_seen = max(int(seen_any.sum()), 1)

for s in range(T):
    rr.set_time("step", sequence=s)
    for r, (name, color) in enumerate(ROBOTS):
        f = R.frame_at(arcs[r], s)
        w, h, fx, fy = cam["intr"][f]
        rr.log(f"world/{name}/camera", rr.Transform3D(translation=cam["pos"][f], mat3x3=cam["rot"][f]))
        rr.log(f"world/{name}/camera", rr.Pinhole(focal_length=[fx, fy], width=w, height=h,
                                                   camera_xyz=rr.ViewCoordinates.RDF,
                                                   image_plane_distance=2.0, color=color))
        path = cam["pos"][arcs[r][:min(s, len(arcs[r]) - 1) + 1]]
        rr.log(f"world/{name}/path", rr.LineStrips3D([path], colors=[color], radii=-2.0))

        new = (global_first == s) & (owner == r)
        if new.any():
            rr.log(f"world/map/{name}/step_{s:03d}",
                   rr.Points3D(pts[new], colors=np.tile(color, (new.sum(), 1)), radii=-1.5))
        know[r] |= first_seen[r] <= s

    if (s + 1) % SYNC_EVERY == 0 or s == T - 1:
        origins, vectors, labels = [], [], []
        for r, (name, color) in enumerate(ROBOTS):
            delta = (first_seen[r] > last_sync) & (first_seen[r] <= s)
            n = int(delta.sum())
            pos_r = cam["pos"][R.frame_at(arcs[r], s)]
            for q in range(len(ROBOTS)):
                if q == r:
                    continue
                know[q] |= delta
                mb_orig += n * scale_orig * BYTES_ORIG / 1024 ** 2
                mb_comp += n * scale_comp * BYTES_COMP / 1024 ** 2
                pos_q = cam["pos"][R.frame_at(arcs[q], s)]
                origins.append(pos_r)
                vectors.append(pos_q - pos_r)
                labels.append(f"{name}: {n * scale_comp * BYTES_COMP / 1024 ** 2:.1f} MB")
        rr.log("world/sync", rr.Arrows3D(origins=origins, vectors=vectors, labels=labels,
                                         colors=[c for _, c in ROBOTS for _ in range(len(ROBOTS) - 1)]))
        last_sync = s
    else:
        rr.log("world/sync", rr.Clear(recursive=False))

    rr.log("stats/mb_sent/original", rr.Scalars(mb_orig))
    rr.log("stats/mb_sent/compressed", rr.Scalars(mb_comp))
    for r, (name, _) in enumerate(ROBOTS):
        rr.log(f"stats/known/{name}", rr.Scalars(100 * know[r].sum() / total_seen))

rr.set_time("step", sequence=T)
rr.log("world/sync", rr.Clear(recursive=False))
rr.log("world/map", rr.Clear(recursive=True))
rr.log("world/final_map", rr.Points3D(pts[seen_any], colors=rgb[seen_any], radii=-1.5))

rr.disconnect()
print(f"Total sent: original {mb_orig:.0f} MB vs compressed {mb_comp:.0f} MB (x{mb_orig / max(mb_comp, 1e-9):.1f})")
print(f"Saved {OUT} ({os.path.getsize(OUT) / 1024 ** 2:.1f} MB). Open with:  rerun {OUT}")
