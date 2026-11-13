import os

SCENE = "truck"

PLY_MAIN = os.environ.get("GS_PLY_MAIN", "data/truck_30000.ply")

OUT_DIR = os.environ.get("GS_OUT", "outputs")

os.makedirs(OUT_DIR, exist_ok=True)
