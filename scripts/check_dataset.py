import hashlib
from collections import Counter
from pathlib import Path

ROOT = Path("data/ds_v1")
NAMES = ["face", "phone", "hold_phone", "cigarette"]
hashes = {}

for split in ["train", "valid", "test"]:
    counts, empty, bad, tiny = Counter(), 0, 0, 0
    for lbl in (ROOT / split / "labels").glob("*.txt"):
        lines = [l.split() for l in lbl.read_text().splitlines() if l.strip()]
        if not lines:
            empty += 1
        for p in lines:
            c, x, y, w, h = int(p[0]), *map(float, p[1:5])
            counts[NAMES[c]] += 1
            if not all(0 <= v <= 1 for v in (x, y, w, h)):
                bad += 1
            if w * h < 0.0005:
                tiny += 1
    for img in (ROOT / split / "images").iterdir():
        hashes.setdefault(hashlib.md5(img.read_bytes()).hexdigest(), set()).add(split)
    print(f"\n{split}: empty={empty}  out_of_range={bad}  tiny_boxes={tiny}")
    for n in NAMES:
        print(f"  {n:12s} {counts[n]}")

leak = sum(1 for s in hashes.values() if len(s) > 1)
print(f"\nSame image in multiple splits (leakage): {leak}")