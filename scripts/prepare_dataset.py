import shutil, yaml
from pathlib import Path

SRC = Path("ds3")
OUT = Path("data/ds_v1")
FINAL = ["face", "phone", "hold_phone", "cigarette"]

names = yaml.safe_load(open(SRC / "data.yaml"))["names"]
id_map = {i: FINAL.index(n) for i, n in enumerate(names) if n in FINAL}
print("mapping:", id_map)   # expected: {2: 3, 3: 0, 4: 2, 5: 1}

dropped, face_imgs, eye_imgs = 0, 0, 0
for split in ["train", "valid", "test"]:
    img_dir = SRC / split / "images"
    if not img_dir.exists():
        continue
    (OUT / split / "images").mkdir(parents=True, exist_ok=True)
    (OUT / split / "labels").mkdir(parents=True, exist_ok=True)
    for img in img_dir.iterdir():
        shutil.copy(img, OUT / split / "images" / img.name)
        lbl = SRC / split / "labels" / (img.stem + ".txt")
        keep, has_face, has_eye = [], False, False
        if lbl.exists():
            for line in lbl.read_text().splitlines():
                p = line.split()
                if not p:
                    continue
                c = int(p[0])
                if c in id_map:
                    p[0] = str(id_map[c])
                    keep.append(" ".join(p))
                    has_face |= (id_map[c] == 0)
                else:
                    dropped += 1
                    has_eye = True
        face_imgs += has_face
        eye_imgs += (has_face and has_eye)
        (OUT / split / "labels" / (img.stem + ".txt")).write_text("\n".join(keep))

yaml.safe_dump({"path": ".", "train": "train/images", "val": "valid/images",
                "test": "test/images",
                "names": {i: n for i, n in enumerate(FINAL)}},
               open(OUT / "data.yaml", "w"), sort_keys=False)

print(f"Eye boxes dropped: {dropped}")
print(f"Images with a face: {face_imgs}, of which with any eye label: {eye_imgs}")