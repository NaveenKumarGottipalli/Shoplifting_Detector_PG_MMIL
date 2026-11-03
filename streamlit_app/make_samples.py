"""Copy the curated test-set clips into samples/ so the app (and your GitHub / Hugging Face repo) can bundle them.

Run this where the dataset videos exist (e.g. your Google Drive 'Dataset' folder or a local copy of the
Hugging Face dataset NaveenKumarGottipalli/Shoplifting_DataCollection):

    python make_samples.py --video-root "/path/to/Dataset"

The manifest's hf_path column is relative to that root, e.g. "S3/data/customdataset/shoplifting/S_000024.mp4".
The 20 selected clips are short, balanced (10 shoplifting / 10 normal) and come from the TEST partition only.
Check each source dataset's licence before redistributing clips publicly.
"""
import argparse, shutil
from pathlib import Path
from samples import select_samples

ap = argparse.ArgumentParser()
ap.add_argument("--video-root", required=True, help="folder that contains S3/, DCSASS Dataset/, Shop DataSet/, UCF/")
ap.add_argument("--manifest", default="Data_Manifests/split_manifest.csv")
ap.add_argument("--out", default="samples")
ap.add_argument("--seed", type=int, default=7)
a = ap.parse_args()

root, out = Path(a.video_root), Path(a.out)
out.mkdir(parents=True, exist_ok=True)
df = select_samples(a.manifest, seed=a.seed)
keep, total = [], 0
for _, r in df.iterrows():
    src = root / r["hf_path"]
    if not src.exists():
        hits = list(root.rglob(Path(r["hf_path"]).name))
        src = hits[0] if hits else None
    if src is None:
        print("MISSING:", r["hf_path"]); continue
    shutil.copy2(src, out / r["file"]); total += (out / r["file"]).stat().st_size; keep.append(r)
import pandas as pd
pd.DataFrame(keep).to_csv(out / "samples_index.csv", index=False)
print(f"Copied {len(keep)}/{len(df)} clips ({total/1e6:.1f} MB) to {out}/ and wrote samples_index.csv")
