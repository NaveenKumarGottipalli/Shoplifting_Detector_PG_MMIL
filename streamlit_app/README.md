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
app.py  model_def.py  features.py  inference.py  viz.py  prompts.py
requirements.txt  packages.txt  .streamlit/config.toml  assets/pipeline.png
Models/        <- notebook checkpoints, e.g. Proposed_PG_MMIL_Full_seed42.pth
Metrics/  Figures/  Data_Manifests/   <- already in the GitHub repo
weights/       <- optional: yolo26n-pose.pt (exact feature parity with training)
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
