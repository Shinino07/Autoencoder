import os

import numpy as np
import rerun as rr
import rerun.blueprint as rrb

import config
from gslib import robots as R
from gslib import sh_codec as C
from gslib.load import load_gaussians

SPARSE = "data/colmap_truck"
MODEL = "models/sh_ae_k8.npz"
OUT = os.path.join(config.OUT_DIR, "demo_sharing.rrd")

OPACITY_MIN = 0.05
DEPTH_FACTOR = 6.0
SYNC_EVERY = 20
VOXELS = [0.1, 0.25, 0.5, 1.0, 2.0]
SHOW_VOXEL = 0.25
VOXEL_ID_BYTES = 4
N_SHOW = 250_000

ROBOTS = [("red", [230, 57, 70]), ("blue", [29, 120, 230]), ("yellow", [240, 180, 20])]
STRATEGY_COLORS = {"send_new": [150, 150, 150], "ask_first": [46, 160, 67], "oracle": [120, 80, 200]}

cam = R.load_cameras(SPARSE, DEPTH_FACTOR)
raw, g = load_gaussians(config.PLY_MAIN)
keep = np.flatnonzero(g["opacity"] >= OPACITY_MIN)
raw_k = raw[keep]
pts_world = g["xyz"][keep].astype(np.float64)
N = len(keep)
lo, hi = g["xyz"].min(0), g["xyz"].max(0)

ae = C.load_ae(MODEL)
latents, zero_sh = C.encode_sh(ae, C.sh_from_raw(raw_k))
B_PER = C.bytes_per_gaussian(ae["k"])

arcs, T = R.split_paths(len(cam["intr"]), len(ROBOTS))
seen = R.first_seen(cam, arcs, T, pts_world)
seen_any = seen.min(0) < R.NEVER
sync_steps = {s for s in range(T) if (s + 1) % SYNC_EVERY == 0 or s == T - 1}
print(f"{N:,} Gaussians after pruning, {seen_any.sum():,} seen by at least one robot; "
      f"{ae['k']}-byte SH latent -> {B_PER} B/Gaussian")


def packet_bytes(n):
    return 0 if n == 0 else C.HEADER.size + (n + 7) // 8 + n * B_PER


def voxel_index(size):
    keys = np.floor((pts_world - lo) / size).astype(np.int64)
    _, vidx = np.unique(keys, axis=0, return_inverse=True)
    return vidx.ravel(), int(vidx.max()) + 1


def simulate(strategy, voxel=None):
    n_r = len(ROBOTS)
    know = np.zeros((n_r, N), bool)
    sent = np.zeros((n_r, n_r, N), bool)
    if strategy == "ask_first":
        vidx, n_vox = voxel_index(voxel)
        told = np.zeros((n_r, n_r, n_vox), bool)
    data_b = summary_b = 0
    mb, complete, transfers, last = [], [], {}, -1
    for s in range(T):
        know |= seen <= s
        if s in sync_steps:
            for r in range(n_r):
                for q in range(n_r):
                    if q == r:
                        continue
                    if strategy == "send_new":
                        cand = (seen[r] > last) & (seen[r] <= s)
                        summ = 0
                    elif strategy == "oracle":
                        cand = know[r] & ~know[q]
                        summ = 0
                    else:
                        has = np.zeros(n_vox, bool)
                        has[vidx[know[q]]] = True
                        new_vox = has & ~told[q, r]
                        summ = VOXEL_ID_BYTES * int(new_vox.sum())
                        told[q, r] |= new_vox
                        cand = know[r] & ~sent[r, q] & ~told[q, r][vidx]
                    b = packet_bytes(int(cand.sum()))
                    data_b += b
                    summary_b += summ
                    sent[r, q] |= cand
                    know[q] |= cand
                    transfers[(s, r, q)] = (b + summ, cand)
            last = s
        mb.append((data_b + summary_b) / 1024 ** 2)
        complete.append(100 * know[:, seen_any].mean())
    return {"mb": mb, "complete": complete, "transfers": transfers, "know": know,
            "data_mb": data_b / 1024 ** 2, "summary_mb": summary_b / 1024 ** 2}


runs = {"send_new": simulate("send_new"), "oracle": simulate("oracle")}
for v in VOXELS:
    runs[f"ask_first@{v}"] = simulate("ask_first", v)

raw_send_new = sum(int(c.sum()) for _, c in runs["send_new"]["transfers"].values()) * 248 / 1024 ** 2
print(f"\n{'strategy':<17}{'data MB':>9}{'summary MB':>12}{'total MB':>10}{'vs send_new':>13}{'final map %':>13}")
base = runs["send_new"]["mb"][-1]
for name, run in runs.items():
    print(f"{name:<17}{run['data_mb']:>9.1f}{run['summary_mb']:>12.2f}{run['mb'][-1]:>10.1f}"
          f"{base / run['mb'][-1]:>12.1f}x{run['complete'][-1]:>12.1f}%")
print(f"(send_new with the uncompressed .ply format would be {raw_send_new:.0f} MB; "
      f"the whole compressed map once is {packet_bytes(N) / 1024 ** 2:.1f} MB)")

(s0, r0, q0), (b0, c0) = next((k, v) for k, v in runs[f"ask_first@{SHOW_VOXEL}"]["transfers"].items() if v[1].any())
sel = np.flatnonzero(c0)
buf = C.pack(raw_k[sel], ae, lo, hi, latents[sel], zero_sh[sel])
got = C.unpack(buf, ae)
sh_in, sh_out = C.sh_from_raw(raw_k[sel]), C.sh_from_raw(got)
print(f"\nCheck: real packet {ROBOTS[r0][0]}->{ROBOTS[q0][0]} at step {s0}: {len(buf):,} B "
      f"(formula {packet_bytes(len(sel)):,} B); receiver decoded {len(got):,} Gaussians, "
      f"SH energy lost {100 * ((sh_out - sh_in) ** 2).sum() / (sh_in ** 2).sum():.1f}%")

show = {"send_new": runs["send_new"], "ask_first": runs[f"ask_first@{SHOW_VOXEL}"], "oracle": runs["oracle"]}
rng = np.random.default_rng(0)
draw = rng.choice(np.flatnonzero(seen_any), size=min(N_SHOW, int(seen_any.sum())), replace=False)
pts = R.to_display(cam, pts_world[draw])
first, owner = seen.min(0)[draw], seen.argmin(0)[draw]

rr.init("multirobot_map_sharing")
blueprint = rrb.Blueprint(rrb.Horizontal(
    rrb.Spatial3DView(origin="world", name="Map"),
    rrb.Vertical(
        rrb.TimeSeriesView(origin="stats/mb_sent", name="MB sent (compressed packets)"),
        rrb.TimeSeriesView(origin="stats/complete", name="% of seen map each robot has (average)"),
        rrb.TextDocumentView(origin="info", name="About"),
    ), column_shares=[3, 2]))
rr.save(OUT, default_blueprint=blueprint)

rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
for name, color in STRATEGY_COLORS.items():
    rr.log(f"stats/mb_sent/{name}", rr.SeriesLines(colors=[color], names=[name]), static=True)
    rr.log(f"stats/complete/{name}", rr.SeriesLines(colors=[color], names=[name]), static=True)
rr.log("info", rr.TextDocument(
    "## Send only what the other robot lacks\n"
    f"- **send_new** (grey): every {SYNC_EVERY} steps each robot sends its new observations to everyone.\n"
    f"- **ask_first** (green): the receiver first lists the {SHOW_VOXEL}-unit voxels it already covers; "
    "the sender skips those.\n"
    "- **oracle** (purple): exactly the missing Gaussians - the best possible, only doable in simulation.\n"
    f"- Packets: pruned map, AE k=8 for SH, {B_PER} B/Gaussian. Arrow labels: ask_first vs send_new.\n\n"
    "*Simplified: no real SLAM, occlusion ignored; all robots share one trained map, "
    "so ids match across robots (real robots would differ).*",
    media_type=rr.MediaType.MARKDOWN), static=True)

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
        new = (first == s) & (owner == r)
        if new.any():
            rr.log(f"world/map/{name}/step_{s:03d}",
                   rr.Points3D(pts[new], colors=np.tile(color, (new.sum(), 1)), radii=-1.5))

    if s in sync_steps:
        origins, vectors, labels, colors = [], [], [], []
        for r, (name, color) in enumerate(ROBOTS):
            for q in range(len(ROBOTS)):
                if q == r:
                    continue
                pos_r, pos_q = cam["pos"][R.frame_at(arcs[r], s)], cam["pos"][R.frame_at(arcs[q], s)]
                smart = show["ask_first"]["transfers"][(s, r, q)][0] / 1024 ** 2
                naive = show["send_new"]["transfers"][(s, r, q)][0] / 1024 ** 2
                origins.append(pos_r)
                vectors.append(pos_q - pos_r)
                labels.append(f"{name}->{ROBOTS[q][0]}: {smart:.1f} MB (send_new {naive:.1f})")
                colors.append(color)
        rr.log("world/sync", rr.Arrows3D(origins=origins, vectors=vectors, labels=labels, colors=colors))
    else:
        rr.log("world/sync", rr.Clear(recursive=False))

    for name, run in show.items():
        rr.log(f"stats/mb_sent/{name}", rr.Scalars(run["mb"][s]))
        rr.log(f"stats/complete/{name}", rr.Scalars(run["complete"][s]))

rr.set_time("step", sequence=T)
rr.log("world/sync", rr.Clear(recursive=False))
rr.log("world/map", rr.Clear(recursive=True))
rgb = (g["rgb"][keep][draw] * 255).astype(np.uint8)
rr.log("world/final_map", rr.Points3D(pts, colors=rgb, radii=-1.5))

rr.disconnect()
print(f"\nSaved {OUT} ({os.path.getsize(OUT) / 1024 ** 2:.1f} MB). Open with:  rerun {OUT}")
