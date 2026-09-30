import shutil, yaml
from pathlib import Path

SRC = Path("data/ds_v1")
OUT = Path("data/dms_v1")
CLASSES = ["face", "phone", "hold_phone", "cigarette"]

for src_split, dst_split in [("train", "train"), ("valid", "val"), ("test", "test")]:
    for kind in ["images", "labels"]:
        s, d = SRC / src_split / kind, OUT / kind / dst_split
        if s.exists():
            d.mkdir(parents=True, exist_ok=True)
            for f in s.iterdir():
                shutil.copy(f, d / f.name)
            print(f"{kind}/{dst_split}: {len(list(d.iterdir()))} files")

# Ultralytics ke liye bhi data.yaml, taaki baad mein comparison kar sako
yaml.safe_dump({"path": ".", "train": "images/train", "val": "images/val",
                "test": "images/test",
                "names": {i: n for i, n in enumerate(CLASSES)}},
               open(OUT / "data.yaml", "w"), sort_keys=False)