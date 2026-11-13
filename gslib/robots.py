import os
import struct

import numpy as np

COLMAP_PARAMS = {0: 3, 1: 4, 2: 4, 3: 5, 4: 8, 5: 8, 6: 12, 7: 5, 8: 4, 9: 5, 10: 12}
COLMAP_HAS_FY = {1, 4, 5, 6, 7, 10}
NEVER = np.iinfo(np.int32).max


def read_colmap_cameras(path):
    cams = {}
    with open(path, "rb") as f:
        n, = struct.unpack("<Q", f.read(8))
        for _ in range(n):
            cid, model = struct.unpack("<ii", f.read(8))
            w, h = struct.unpack("<QQ", f.read(16))
            p = struct.unpack(f"<{COLMAP_PARAMS[model]}d", f.read(8 * COLMAP_PARAMS[model]))
            cams[cid] = (w, h, p[0], p[1] if model in COLMAP_HAS_FY else p[0])
    return cams


def read_colmap_images(path):
    out = []
    with open(path, "rb") as f:
        n, = struct.unpack("<Q", f.read(8))
        for _ in range(n):
            f.read(4)
            q = np.array(struct.unpack("<4d", f.read(32)))
            t = np.array(struct.unpack("<3d", f.read(24)))
            cid, = struct.unpack("<i", f.read(4))
            name = b""
            while (c := f.read(1)) != b"\0":
                name += c
            n2d, = struct.unpack("<Q", f.read(8))
            f.seek(24 * n2d, 1)
            out.append((name.decode(), qvec_to_rot(q), t, cid))
    return sorted(out, key=lambda x: x[0])


def qvec_to_rot(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def rotation_to_z(v):
    z = np.array([0.0, 0.0, 1.0])
    axis = np.cross(v, z)
    s, c = np.linalg.norm(axis), float(v @ z)
    if s < 1e-9:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    k = axis / s
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + s * K + (1 - c) * K @ K


def look_at_point(centers, forwards):
    A, b = np.zeros((3, 3)), np.zeros(3)
    for c, d in zip(centers, forwards):
        P = np.eye(3) - np.outer(d, d)
        A += P
        b += P @ c
    return np.linalg.solve(A, b)


def load_cameras(sparse_dir, depth_factor):
    intr_by_id = read_colmap_cameras(os.path.join(sparse_dir, "cameras.bin"))
    images = read_colmap_images(os.path.join(sparse_dir, "images.bin"))
    cam = {"R_c2w": np.stack([R.T for _, R, _, _ in images]),
           "centers": np.stack([-R.T @ t for _, R, t, _ in images]),
           "intr": [intr_by_id[cid] for *_, cid in images]}
    up = -cam["R_c2w"][:, :, 1].mean(0)
    cam["up"] = up / np.linalg.norm(up)
    cam["target"] = look_at_point(cam["centers"], cam["R_c2w"][:, :, 2])
    cam["align"] = rotation_to_z(cam["up"])
    cam["pos"] = to_display(cam, cam["centers"])
    cam["rot"] = cam["align"] @ cam["R_c2w"]
    cam["max_depth"] = depth_factor * np.median(np.linalg.norm(cam["pos"], axis=1))
    return cam


def to_display(cam, pts_world):
    return ((pts_world - cam["target"]) @ cam["align"].T).astype(np.float32)


def visible(cam, frame, pts_world):
    w, h, fx, fy = cam["intr"][frame]
    pc = (pts_world - cam["centers"][frame]) @ cam["R_c2w"][frame]
    z = pc[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u, v = fx * pc[:, 0] / z, fy * pc[:, 1] / z
    return (z > 0.05) & (z < cam["max_depth"]) & (np.abs(u) < w / 2) & (np.abs(v) < h / 2)


def split_paths(n_frames, n_robots):
    arcs = np.array_split(np.arange(n_frames), n_robots)
    return arcs, max(len(a) for a in arcs)


def frame_at(arc, step):
    return arc[min(step, len(arc) - 1)]


def first_seen(cam, arcs, steps, pts_world):
    seen = np.full((len(arcs), len(pts_world)), NEVER, np.int32)
    for r, arc in enumerate(arcs):
        for s in range(steps):
            new = visible(cam, frame_at(arc, s), pts_world) & (seen[r] == NEVER)
            seen[r, new] = s
    return seen
