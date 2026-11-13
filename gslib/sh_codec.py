import struct

import numpy as np

N_SH = 45
MAGIC = b"GSPK"
HEADER = struct.Struct("<4sIB6f")


def _silu(x):
    return x * 0.5 * (1.0 + np.tanh(0.5 * x))


def _linear(x, ae, name):
    return x @ ae[name + "_weight"].T + ae[name + "_bias"]


def load_ae(path):
    with np.load(path) as f:
        ae = {name: f[name].astype(np.float32) for name in f.files}
    ae["k"] = int(ae["k"])
    ae["scale"] = float(ae["scale"])
    return ae


def sh_from_raw(raw):
    return np.stack([raw[f"f_rest_{i}"] for i in range(N_SH)], axis=1).astype(np.float32)


def encode_sh(ae, sh, batch=200_000):
    zero = np.all(sh == 0, axis=1)
    lat = np.zeros((len(sh), ae["k"]), np.uint8)
    for i in range(0, len(sh), batch):
        x = sh[i:i + batch] / ae["scale"]
        h = _silu(_linear(x, ae, "enc_0"))
        h = _silu(_linear(h, ae, "enc_2"))
        z = np.tanh(_linear(h, ae, "enc_4"))
        lat[i:i + batch] = np.round((z + 1) / 2 * 255)
    lat[zero] = 0
    return lat, zero


def decode_sh(ae, lat, zero, batch=200_000):
    out = np.zeros((len(lat), N_SH), np.float32)
    for i in range(0, len(lat), batch):
        z = lat[i:i + batch].astype(np.float32) / 255 * 2 - 1
        h = _silu(_linear(z, ae, "dec_0"))
        h = _silu(_linear(h, ae, "dec_2"))
        out[i:i + batch] = _linear(h, ae, "dec_4") * ae["scale"]
    out[zero] = 0
    return out


def record_dtype(k):
    return np.dtype([("pos", "<u2", (3,)), ("f_dc", "<f2", (3,)), ("opacity", "<f2"),
                     ("scale", "<f2", (3,)), ("rot", "<f2", (4,)), ("latent", "u1", (k,))])


def _cols(raw, names):
    return np.stack([raw[n] for n in names], axis=1)


def pack(raw, ae, lo, hi, latents=None, zero=None):
    k = ae["k"]
    if latents is None:
        latents, zero = encode_sh(ae, sh_from_raw(raw))
    lo, hi = np.asarray(lo, np.float64), np.asarray(hi, np.float64)
    rec = np.zeros(len(raw), record_dtype(k))
    xyz = _cols(raw, ["x", "y", "z"]).astype(np.float64)
    rec["pos"] = np.clip(np.round((xyz - lo) / (hi - lo) * 65535), 0, 65535)
    rec["f_dc"] = _cols(raw, ["f_dc_0", "f_dc_1", "f_dc_2"])
    rec["opacity"] = raw["opacity"]
    rec["scale"] = _cols(raw, ["scale_0", "scale_1", "scale_2"])
    rec["rot"] = _cols(raw, ["rot_0", "rot_1", "rot_2", "rot_3"])
    rec["latent"] = latents
    return (HEADER.pack(MAGIC, len(raw), k, *lo, *hi)
            + np.packbits(zero).tobytes() + rec.tobytes())


def unpack(buf, ae):
    magic, n, k, *box = HEADER.unpack_from(buf)
    if magic != MAGIC or k != ae["k"]:
        raise ValueError("Packet does not match this decoder")
    lo, hi = np.array(box[:3]), np.array(box[3:])
    off = HEADER.size
    n_mask = (n + 7) // 8
    zero = np.unpackbits(np.frombuffer(buf, np.uint8, n_mask, off))[:n].astype(bool)
    rec = np.frombuffer(buf, record_dtype(k), n, off + n_mask)

    names = (["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2"]
             + [f"f_rest_{i}" for i in range(N_SH)]
             + ["opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"])
    out = np.zeros(n, np.dtype([(name, "<f4") for name in names]))
    xyz = rec["pos"].astype(np.float64) / 65535 * (hi - lo) + lo
    for i, a in enumerate("xyz"):
        out[a] = xyz[:, i]
        out[f"f_dc_{i}"] = rec["f_dc"][:, i]
        out[f"scale_{i}"] = rec["scale"][:, i]
    sh = decode_sh(ae, rec["latent"], zero)
    for i in range(N_SH):
        out[f"f_rest_{i}"] = sh[:, i]
    out["opacity"] = rec["opacity"]
    for i in range(4):
        out[f"rot_{i}"] = rec["rot"][:, i]
    return out


def bytes_per_gaussian(k):
    return record_dtype(k).itemsize
