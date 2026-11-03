---
title: Retail Shoplifting Detector (PG-MMIL)
emoji: 🛒
colorFrom: gray
colorTo: green
sdk: streamlit
app_file: app.py
pinned: false
short_description: Pose-guided multimodal MIL research demo for shoplifting detection
---

# Retail Shoplifting Detector — PG-MMIL

Research prototype for weakly supervised retail shoplifting detection: YOLO-pose + frozen CLIP ViT-B/32 + retail text prompts
feed a dual-stream Transformer MIL model (PG-MMIL) trained from video-level labels on four public datasets
(UCF-Crime retail subset, DCSASS, MNNIT-S3, Shop Dataset).

**Alerts are for human review only and are not evidence of theft.** Held-out-domain ROC-AUC is only about 0.55–0.70,
so performance on a new store can be much lower than the pooled benchmark. See the app's *Limits & ethics* tab.

## Repository layout (the app expects this)

```
app.py  model_def.py  features.py  inference.py  viz.py  prompts.py  samples.py  make_samples.py
requirements.txt  packages.txt  .streamlit/config.toml  assets/pipeline.png
Models/        <- notebook checkpoints, e.g. Proposed_PG_MMIL_Full_seed42.pth
Metrics/  Figures/  Data_Manifests/   <- already in the GitHub repo
weights/       <- optional: yolo26n-pose.pt (exact feature parity with training)
samples/       <- samples_index.csv (included) + the 20 short test-set clips (see below)
```

Checkpoints are the files written by the notebook: `<model>_seed<seed>.pth` containing `state_dict` and `threshold`.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Hugging Face Spaces

1. Create a new Space (SDK: **Streamlit**) at huggingface.co/new-space.
2. Push this repository to it:
   ```bash
   git remote add space https://huggingface.co/spaces/<user>/<space-name>
   git push space main
   ```
   (Use a Hugging Face access token with write permission when prompted. Never commit tokens.)
3. Do not push raw videos or the 2.4 MB notebook outputs if you want a lean Space; checkpoints (a few MB each) are fine.
4. First start downloads the CLIP weights (LAION-2B ViT-B/32) from the Hub; CPU Basic hardware works but each video takes tens of seconds.

## Notes on fidelity

* `model_def.py` is copied from the notebook so checkpoints load unchanged.
* The Sultani-, RTFM-, Shopformer- and VadCLIP-named models are compact *inspired* baselines, not the published systems.
* On CPU, features are computed in FP32 (the notebook used FP16 on GPU), so scores can differ slightly.

## Troubleshooting

**`ImportError: libGL.so.1` (OpenCV) on Streamlit Community Cloud.** Community Cloud reads `packages.txt` and `requirements.txt`
only from the **repository root**, not from a sub-folder. If the app lives in `streamlit_app/`, copy `packages.txt`
(`libgl1`, `libglib2.0-0`) and `requirements.txt` to the repo root, then reboot the app. The UI code itself no longer imports OpenCV;
only video decoding and ultralytics do, and those load when you analyse a video.

**Automatic fallback.** If `libGL.so.1` is still missing at run time, `bootstrap.py` swaps `opencv-python` for `opencv-python-headless` once (pip, or uv) and retries. Fixing `packages.txt` at the repo root and rebooting remains the cleanest solution, and the first analysis after a reboot may take longer while this runs.

**Python version.** If a package has no wheel for the default Python (3.14), pick Python 3.12 under Advanced settings when deploying.

## Links in the UI

Edit the two constants at the top of `app.py`:

```python
GITHUB_URL = "https://github.com/NaveenKumarGottipalli/Shoplifting_Detector_PG_MMIL"
LINKEDIN_URL = ""   # paste your LinkedIn profile URL; leave empty and the link is hidden everywhere
```

## Test-set sample clips

`samples/samples_index.csv` lists 20 short clips (10 shoplifting, 10 normal) drawn only from the **held-out test partition**.
The app shows them in a "Test-set sample" picker and reveals the true label only after the model answers.

To make the clips available to everyone, run this once where the dataset videos exist (Drive folder or a local copy of the dataset):

```bash
python make_samples.py --video-root "/path/to/Dataset"
```

It copies the clips into `samples/`; commit them (about a few MB). If a clip is not in `samples/`, the app tries to download it
from the Hugging Face dataset `NaveenKumarGottipalli/Shoplifting_DataCollection` using the manifest's `hf_path`, and shows a friendly
message if that fails. Check each source dataset's licence before redistributing clips publicly.
