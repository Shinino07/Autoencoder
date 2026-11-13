# Multi-robot Gaussian map sharing

Three simulated robots build a 3D Gaussian Splatting map and share it with each other, using an
autoencoder to compress the SH colour coefficients and sending each robot only the parts it lacks.
Split out of `../Neural network model for 3DGS compression and decompression` (the A-H analysis of the 3DGS output).

## Contents

| Path | What |
|---|---|
| `gslib/load.py` | Read a 3DGS `.ply` and decode its attributes |
| `gslib/sh_codec.py` | SH autoencoder in plain numpy + 36-byte packets per Gaussian (`pack` / `unpack`) |
| `gslib/robots.py` | Robot paths from the COLMAP cameras and what each robot sees |
| `demo_multirobot.py` | Robots build the map and broadcast it: original vs compressed bandwidth |
| `demo_sharing.py` | Direction C: `send_new` vs `ask_first` (voxel summary) vs `oracle` |
| `models/sh_ae_k8.npz`, `sh_ae_k16.npz` | Autoencoder weights, trained on Colab (`sh_autoencoder.ipynb`) and exported to numpy |
| `data/truck_30000.ply` | Trained Truck scene (30k iterations) |
| `data/colmap_truck/` | COLMAP camera poses of the Truck scene |

## Run

```
python -m venv .venv
.venv\Scripts\activate
pip install numpy rerun-sdk
python demo_multirobot.py
python demo_sharing.py
rerun outputs\demo_sharing.rrd
```

Browser instead of the desktop viewer: `rerun --serve-web --web-viewer-port 9090 outputs\demo_sharing.rrd`,
then open the link it prints.

## Simplifications

No real SLAM (robots reuse the trained map), occlusion ignored, robot paths are the camera ring split in three,
and all robots share one map so Gaussian ids match across robots (only `ask_first` would work on real robots).
