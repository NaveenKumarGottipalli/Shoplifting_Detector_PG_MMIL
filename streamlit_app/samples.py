"""Test-set sample selection shared by the app and make_samples.py.

Picks a small, balanced, *short* set of videos from the held-out TEST partition of Data_Manifests/split_manifest.csv
(per domain: shoplifting and normal clips) so that people without CCTV footage can try the detector.
"""
from pathlib import Path
import pandas as pd

HF_DATASET_ID = "NaveenKumarGottipalli/Shoplifting_DataCollection"   # source repo used by the notebook (hf_path is relative to it)
MAX_SECONDS = {"UCF-Crime": 120.0, "Shop-Dataset": 30.0, "DCSASS": 10.0, "MNNIT-S3": 10.0}
PER_CLASS = {"UCF-Crime": 1, "Shop-Dataset": 3, "DCSASS": 3, "MNNIT-S3": 3}


def select_samples(manifest_csv, seed=7):
    df = pd.read_csv(manifest_csv)
    df = df[df["split"] == "test"].copy()
    rows = []
    for dom, g in df.groupby("domain_name"):
        g = g[g["duration_sec"] <= MAX_SECONDS.get(dom, 30.0)]
        for lab in (1, 0):
            h = g[g["label"] == lab]
            n = min(PER_CLASS.get(dom, 2), len(h))
            if n:
                rows.append(h.sample(n=n, random_state=seed))
    out = pd.concat(rows).sort_values(["domain_name", "label", "hf_path"]).reset_index(drop=True)
    out["file"] = [f"{r.domain_name.replace('-', '').lower()}_{'shoplifting' if r.label == 1 else 'normal'}_{i:02d}{Path(r.hf_path).suffix.lower()}"
                   for i, r in out.iterrows()]
    return out[["file", "label", "domain_name", "split", "duration_sec", "hf_path"]]


def sample_label(r):
    return f"{r['domain_name']} · {r['duration_sec']:.0f} s · #{r.name + 1}"
