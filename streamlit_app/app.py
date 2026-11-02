"""Shoplifting Detector - PG-MMIL research demo (Streamlit).

Run locally:  streamlit run app.py
Layout expected next to this file (same as the GitHub repo):  Models/  Metrics/  Figures/  Data_Manifests/  (+ weights/ optional)
"""
import hashlib, json, tempfile, time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

import viz
from inference import find_checkpoints

ROOT = Path(__file__).resolve().parent
NAMES = {
    "Proposed_PG_MMIL_Full": "PG-MMIL (full)", "Proposed_PG_MMIL_NoGRL": "PG-MMIL (no GRL)",
    "VadCLIP_VisualPromptMIL": "VadCLIP-inspired", "RTFM_Net": "RTFM-inspired",
    "Shopformer_PoseMIL": "Pose-only Transformer", "Sultani_MIL": "Sultani-inspired",
}
ORDER = list(NAMES)

st.set_page_config(page_title="Retail Shoplifting Detector | PG-MMIL", page_icon="🛒", layout="wide",
                   initial_sidebar_state="expanded")

CSS = """
<style>
.block-container {padding-top: 1.2rem; max-width: 1250px;}
.hero {background: linear-gradient(120deg,#14213D 0%,#0B6E6E 100%); border-radius: 18px; padding: 28px 34px; color:#fff; margin-bottom: 14px;}
.hero h1 {margin:0 0 6px 0; font-size: 2.0rem; color:#fff;}
.hero p {margin:0; font-size: 1.02rem; color:#DCEBEA; max-width: 900px;}
.chips span {display:inline-block; background: rgba(255,255,255,.14); border-radius: 999px; padding: 3px 12px; margin: 10px 8px 0 0; font-size:.82rem; color:#fff;}
.card {background:#FFFFFF; border:1px solid #E1E8EE; border-radius:14px; padding:16px 18px; color:#14213D; height:100%;}
.card h4 {margin:0 0 6px 0; color:#0B6E6E; font-size:1rem;}
.card p, .card li {color:#33415C; font-size:.93rem;}
.verdict {border-radius:16px; padding:18px 22px; color:#fff; margin: 4px 0 10px 0;}
.verdict.alert {background:#E4572E;} .verdict.ok {background:#0B6E6E;}
.verdict h2 {margin:0; color:#fff; font-size:1.5rem;} .verdict p {margin:4px 0 0 0; color:#FFF3EE; font-size:.95rem;}
.note {background:#FFF6E5; border-left:5px solid #F2A33A; border-radius:8px; padding:10px 14px; color:#4A3A12; font-size:.92rem; margin:8px 0;}
.small {color:#5B6B82; font-size:.85rem;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ------------------------------------------------------------------ helpers
def find_dir(name):
    for base in (ROOT, ROOT.parent):
        p = base / name
        if p.exists():
            return p
    return None


@st.cache_data(show_spinner=False)
def read_csv(rel):
    d = find_dir(rel.split("/")[0])
    p = (d.parent / rel) if d else None
    return pd.read_csv(p) if p and p.exists() else None


def show_figure(name, caption):
    d = find_dir("Figures")
    p = d / name if d else None
    if p and p.exists():
        st.image(str(p))
        st.caption(caption)
    else:
        st.info(f"Figure {name} not found in Figures/.")


def card(title, body):
    st.markdown(f'<div class="card"><h4>{title}</h4>{body}</div>', unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading pose and CLIP models (first run downloads weights)...")
def get_extractor():
    from features import Extractor
    wd = find_dir("weights")
    return Extractor(weights_dir=wd)


@st.cache_resource(show_spinner=False)
def get_model(name, path):
    from inference import load_model
    return load_model(name, path)


CKPTS = find_checkpoints(find_dir("Models") or ROOT / "Models")

# ------------------------------------------------------------------ hero
st.markdown("""
<div class="hero">
  <h1>🛒 Retail Shoplifting Detector</h1>
  <p>Pose-Guided Multimodal Multiple-Instance Learning (PG-MMIL): a research prototype that scores a video for
  suspicious concealment behaviour using body pose, CLIP visual features and retail text prompts, trained from
  video-level labels only.</p>
  <div class="chips"><span>YOLO-pose</span><span>CLIP ViT-B/32</span><span>Dual-stream Transformer</span>
  <span>Cross-attention + gate</span><span>Gradient-reversal domain adaptation</span><span>4 public datasets</span></div>
</div>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("Detector settings")
    avail_models = [m for m in ORDER if any(k[0] == m for k in CKPTS)]
    if avail_models:
        model_name = st.selectbox("Model", avail_models, format_func=lambda m: NAMES[m])
        seeds = sorted({s for (m, s) in CKPTS if m == model_name})
        seed = st.selectbox("Training seed", seeds, help="The benchmark trained each model with seeds 42, 7 and 21.")
        ckpt_path = CKPTS[(model_name, seed)]
    else:
        model_name = seed = ckpt_path = None
        st.warning("No checkpoints found in Models/.")
    sens = st.slider("Alert sensitivity adjustment", -0.30, 0.30, 0.0, 0.01,
                     help="Shifts the decision threshold from the validation-selected value. Negative = more alerts (more false alarms).")
    compare_all = st.toggle("Compare all available models", value=False)
    st.divider()
    st.markdown("**Research prototype.** Alerts are for human review only and are not evidence of theft. "
                "See the *Limits & ethics* tab.")

tabs = st.tabs(["🎥 Analyze video", "📊 Benchmark results", "🧪 Data & protocol", "🧠 How it works", "⚖️ Limits & ethics"])

# ================================================================== TAB 1
with tabs[0]:
    if not CKPTS:
        st.markdown("""<div class="note"><b>Trained checkpoints are needed for live scoring.</b> Place the notebook's
        checkpoint files in <code>Models/</code> using the notebook's naming, e.g.
        <code>Proposed_PG_MMIL_Full_seed42.pth</code>. Each file must contain
        <code>state_dict</code> and <code>threshold</code>. The other tabs work without them.</div>""", unsafe_allow_html=True)
    up = st.file_uploader("Upload a short retail video (mp4, avi, mov, mkv, webm)", type=["mp4", "avi", "mov", "mkv", "webm"])
    st.markdown('<p class="small">Videos are processed in memory on this server, not stored, and the app does not identify people. '
                'Use footage you are authorised to analyse.</p>', unsafe_allow_html=True)

    if up is not None and CKPTS:
        data = up.getvalue()
        key = hashlib.md5(data).hexdigest()
        if st.session_state.get("analysis_key") != key:
            t0 = time.time()
            with st.status("Extracting pose and CLIP features (32 snippets)...", expanded=False) as status:
                try:
                    with tempfile.NamedTemporaryFile(suffix=Path(up.name).suffix, delete=False) as tmp:
                        tmp.write(data)
                        tmp_path = tmp.name
                    res = get_extractor().extract(tmp_path)
                    Path(tmp_path).unlink(missing_ok=True)
                    st.session_state["analysis"], st.session_state["analysis_key"] = res, key
                    status.update(label=f"Features ready in {time.time() - t0:.1f}s", state="complete")
                except Exception as e:  # show a readable error instead of a stack trace
                    status.update(label="Feature extraction failed", state="error")
                    st.error(f"Could not process this video: {e}")
                    st.stop()
        res = st.session_state["analysis"]
        meta, feats, frames, boxes = res["meta"], res["features"], res["frames"], res["boxes"]
        from inference import score_bag
        model, minfo = get_model(model_name, ckpt_path)
        snip, bag = score_bag(model, feats, minfo["k"])
        thr = float(np.clip(minfo["threshold"] + sens, 0.0, 1.0))
        flagged = bag >= thr

        c1, c2 = st.columns([3, 2])
        with c1:
            cls = "alert" if flagged else "ok"
            head = "Review recommended: possible concealment behaviour" if flagged else "No alert: behaviour looks like normal shopping"
            st.markdown(f'<div class="verdict {cls}"><h2>{head}</h2><p>Bag score {bag:.3f} vs threshold {thr:.3f} '
                        f'(margin {bag - thr:+.3f}) · {NAMES[model_name]} · seed {seed}</p></div>', unsafe_allow_html=True)
            st.pyplot(viz.score_bar(bag, thr))
        with c2:
            detected = int(sum(b is not None for b in boxes))
            st.metric("Video length", f"{meta['duration_s']:.1f} s", f"{meta['total_frames']} frames @ {meta['fps']:.0f} fps")
            st.metric("Person found in snippets", f"{detected} / 32")
            st.metric("Bag score (mean of top-%d snippets)" % minfo["k"], f"{bag:.3f}")
        if meta.get("pose_model") != "yolo26n-pose.pt":
            st.markdown(f'<div class="note">Pose model <b>{meta.get("pose_model")}</b> was used instead of the '
                        'yolo26n-pose.pt used for training, so features may differ slightly from the training distribution. '
                        'Put yolo26n-pose.pt in <code>weights/</code> for exact parity.</div>', unsafe_allow_html=True)
        if detected < 8:
            st.markdown('<div class="note">A person was detected in few snippets, so the pose stream had little to work with '
                        'and the result is less reliable.</div>', unsafe_allow_html=True)

        t_time, t_evid, t_prompt, t_pose, t_cmp, t_export = st.tabs(
            ["Timeline", "Evidence frames", "Semantic prompts", "Pose motion", "Compare models", "Export"])
        with t_time:
            st.pyplot(viz.snippet_timeline(snip, thr, minfo["k"], meta["duration_s"]))
            st.markdown('<div class="note">Snippet scores are <b>qualitative</b>. The model is trained with video-level labels only and was '
                        'never evaluated for temporal localisation, so a high score at a moment does not prove that the act happened then.</div>',
                        unsafe_allow_html=True)
        with t_evid:
            top = np.argsort(snip)[::-1][:6]
            cols = st.columns(3)
            for j, i in enumerate(top):
                t_s = i * meta["duration_s"] / 31
                with cols[j % 3]:
                    st.image(viz.annotate_frame(frames[i], feats[i, :51], boxes[i], f"t={t_s:.1f}s  score {snip[i]:.2f}"))
            st.caption("Highest-scoring snippets with the detected person box and pose skeleton.")
        with t_prompt:
            from prompts import PROMPT_LABELS
            st.pyplot(viz.prompt_heatmap(feats[:, 563:567], PROMPT_LABELS))
            st.caption("Cosine similarity between each person crop and four zero-shot CLIP prompts. These are fed to the model as semantic priors.")
        with t_pose:
            fig, _ = viz.wrist_speed(feats[:, :51], meta["duration_s"])
            st.pyplot(fig)
            st.caption("Hip-centred wrist speed between snippets: a simple view of the kinematic signal that the pose stream models.")
        with t_cmp:
            if compare_all:
                rows = []
                for m in ORDER:
                    if (m, seed) in CKPTS:
                        mdl, mi = get_model(m, CKPTS[(m, seed)])
                        _, b = score_bag(mdl, feats, mi["k"])
                        rows.append({"Model": NAMES[m], "Bag score": round(b, 3), "Threshold": round(mi["threshold"], 3),
                                     "Alert": "yes" if b >= mi["threshold"] else "no"})
                if rows:
                    st.dataframe(pd.DataFrame(rows))
                    agree = sum(r["Alert"] == "yes" for r in rows)
                    st.caption(f"{agree} of {len(rows)} models raise an alert. Disagreement between models is itself a signal that a human should look.")
                else:
                    st.info(f"No checkpoints for seed {seed}.")
            else:
                st.info("Turn on 'Compare all available models' in the sidebar.")
        with t_export:
            report = {"file": up.name, "model": model_name, "seed": seed, "bag_score": bag, "threshold": thr,
                      "alert": bool(flagged), "snippet_scores": [float(s) for s in snip], "video": {k: meta[k] for k in
                      ("total_frames", "fps", "duration_s", "pose_model")}, "note": "Research prototype; for human review only."}
            st.download_button("Download JSON report", json.dumps(report, indent=2), file_name="shoplifting_report.json", mime="application/json")
            st.json(report)
    elif up is None:
        c = st.columns(3)
        with c[0]: card("1 · Upload", "<p>Drop a short store-camera clip. The app samples 32 evenly spaced snippets.</p>")
        with c[1]: card("2 · Analyse", "<p>YOLO-pose and CLIP extract motion and semantic features; PG-MMIL scores every snippet.</p>")
        with c[2]: card("3 · Review", "<p>See the bag score against a validation-selected threshold, the evidence frames and per-snippet timeline.</p>")

# ================================================================== TAB 2
with tabs[1]:
    bench = read_csv("Metrics/independent_test_benchmark.csv")
    byseed = read_csv("Metrics/test_metrics_by_seed.csv")
    st.subheader("Independent test-set benchmark")
    st.markdown('<div class="note"><b>How to read this.</b> All models use the same cached features, partitions and protocol. '
                'Baselines are compact <i>inspired</i> versions, not the published systems. Differences between PG-MMIL and the '
                'VadCLIP-inspired model are within sampling noise (see significance below).</div>', unsafe_allow_html=True)
    if byseed is not None:
        agg = byseed.groupby("Model")[["ROC-AUC", "Average Precision", "F1", "FAR (%)"]].agg(["mean", "std"])
        tbl = pd.DataFrame({m: {c: f"{agg.loc[m, (c, 'mean')]:.3f} ± {agg.loc[m, (c, 'std')]:.3f}" for c in ["ROC-AUC", "Average Precision", "F1", "FAR (%)"]}
                            for m in ORDER if m in agg.index}).T
        tbl.index = [NAMES[m] for m in tbl.index]
        st.markdown("**Mean ± SD over 3 seeds (one fixed split)**")
        st.dataframe(tbl)
        means = byseed.groupby("Model")[["ROC-AUC", "Average Precision", "F1", "FAR (%)"]].mean().reindex([m for m in ORDER if m in agg.index])
        st.pyplot(viz.metric_bars(means, [("Average Precision", "Average precision", False), ("F1", "F1", False), ("FAR (%)", "False-alarm rate %", True)],
                                  NAMES, "Proposed_PG_MMIL_Full"))
    else:
        st.info("Metrics/test_metrics_by_seed.csv not found.")
    a, b = st.columns(2)
    with a:
        show_figure("Fig2_ROC_PR_Curves.png", "ROC and precision-recall curves, primary seed (42).")
    with b:
        show_figure("Fig6_Confusion_FAR_Analysis.png", "Row-normalised confusion matrices and F1 vs false-alarm rate.")
    st.subheader("Is PG-MMIL really better? Paired bootstrap")
    pb = read_csv("Metrics/paired_bootstrap_model_differences.csv")
    if pb is not None:
        show = pb.copy()
        show["Comparison"] = show["Comparison"].str.replace("PG-MMIL minus ", "PG-MMIL − ").str.replace("_", " ")
        show["AP difference (95% CI)"] = show.apply(lambda r: f"{r['Average Precision difference']:+.3f} [{r['Average Precision difference 95% CI lower']:+.3f}, {r['Average Precision difference 95% CI upper']:+.3f}]", axis=1)
        show["F1 difference (95% CI)"] = show.apply(lambda r: f"{r['F1 difference']:+.3f} [{r['F1 difference 95% CI lower']:+.3f}, {r['F1 difference 95% CI upper']:+.3f}]", axis=1)
        st.dataframe(show[["Comparison", "AP difference (95% CI)", "F1 difference (95% CI)"]])
        st.caption("A difference is significant only if its interval excludes zero. PG-MMIL clearly beats the Sultani-style baseline but is statistically tied with the others.")
    st.subheader("Cross-domain generalisation (leave-one-domain-out)")
    lodo = read_csv("Metrics/leave_one_domain_out_results.csv")
    if lodo is not None:
        piv = lodo.pivot(index="Held-out domain", columns="Model", values="ROC-AUC").round(3)
        piv.columns = [NAMES.get(c, c) for c in piv.columns]
        c1, c2 = st.columns([2, 3])
        with c1:
            st.dataframe(piv)
            st.markdown('<div class="note">Held-out-domain ROC-AUC is only about 0.55 to 0.70, far below the in-domain values. '
                        'Expect a new store or camera to be much harder than the pooled test set.</div>', unsafe_allow_html=True)
        with c2:
            show_figure("Fig4_Cross_Dataset_Heatmaps.png", "ROC-AUC on a domain withheld from training and source-validation.")
    with st.expander("Training curves and per-domain AUC"):
        show_figure("Fig3_Training_Convergence_Ablation.png", "Validation ROC-AUC over 45 epochs and per-domain test ROC-AUC.")
    with st.expander("Inference cost"):
        ip = read_csv("Metrics/inference_profile.csv")
        if ip is not None:
            ip = ip.copy(); ip["Model"] = ip["Model"].map(lambda m: NAMES.get(m, m))
            st.dataframe(ip[["Model", "milliseconds_per_video", "videos_per_second", "peak_gpu_memory_mb"]].round(3))
            st.caption("Model-only forward pass on cached features, batch of 32. Excludes video decoding, pose and CLIP extraction, which dominate real latency.")
    with st.expander("Feature-space and interpretability figures (qualitative, with caveats)"):
        show_figure("Fig5_tSNE_Feature_Space.png", "t-SNE of test features coloured by class only; this does not demonstrate domain invariance.")
        show_figure("Fig8_Kinematic_Prompt_Interpretability.png", "Single-example view; the score is near-saturated across snippets, so no localisation is claimed.")

# ================================================================== TAB 3
with tabs[2]:
    sm = read_csv("Data_Manifests/split_manifest.csv")
    st.subheader("Four public sources, one unified manifest")
    if sm is not None:
        sm = sm.copy(); sm["Class"] = sm["label"].map({0: "Normal", 1: "Shoplifting"})
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Videos", f"{len(sm):,}"); k2.metric("Shoplifting", f"{int(sm['label'].sum()):,}")
        k3.metric("Normal", f"{int((1 - sm['label']).sum()):,}"); k4.metric("Domains", sm["domain_name"].nunique())
        ct = pd.crosstab([sm["domain_name"], sm["Class"]], sm["split"])[["train", "validation", "test"]]
        ct["total"] = ct.sum(axis=1)
        c1, c2 = st.columns([2, 3])
        with c1:
            st.markdown("**Videos per domain, class and split**")
            st.dataframe(ct)
            dur = sm.groupby("domain_name")["duration_sec"].agg(["mean", "min", "max"]).round(1)
            dur.columns = ["mean (s)", "min (s)", "max (s)"]
            st.markdown("**Clip duration by domain**")
            st.dataframe(dur)
        with c2:
            show_figure("Fig1_Dataset_Distribution.png", "Class balance and duration spread (log scale) across domains.")
    else:
        st.info("Data_Manifests/split_manifest.csv not found.")
    c = st.columns(4)
    srcs = [("UCF-Crime (retail)", "Long untrimmed CCTV. All 50 shoplifting videos plus normal videos filtered to retail scenes by annotation keywords."),
            ("DCSASS", "Short surveillance clips from the Shoplifting folder, labelled with a CSV file."),
            ("MNNIT-S3", "MNNIT Allahabad shoplifting dataset (Mendeley), processed version; clips under 3 s."),
            ("Shop Dataset", "Store videos of shoplifters vs non-shoplifters, 3 to 74 s.")]
    for col, (t, d) in zip(c, srcs):
        with col: card(t, f"<p>{d}</p>")
    st.markdown("**Protocol.** Group-aware stratified split (domain × class; SHA-256 feature hashes as groups) into ~70/15/15; checkpoint and threshold chosen on validation only; "
                "test scored once; leave-one-domain-out removes the target domain before any splitting.")

# ================================================================== TAB 4
with tabs[3]:
    st.subheader("From video to alert")
    pic = ROOT / "assets" / "pipeline.png"
    if pic.exists():
        st.image(str(pic))
    c = st.columns(3)
    with c[0]: card("1 · Snippets and features", "<ul><li>32 evenly spaced frames per video</li><li>YOLO-pose: 17 keypoints (x, y, conf) = 51-D</li><li>Frozen CLIP ViT-B/32 on the person crop: 512-D</li><li>4 prompt similarities: 4-D</li><li>Result: 32 × 567 bag</li></ul>")
    with c[1]: card("2 · PG-MMIL", "<ul><li>Pose stream: hip-centred position, velocity, acceleration (153-D) → 2-layer Transformer</li><li>Visual-prompt stream (516-D) → 2-layer Transformer</li><li>Bidirectional cross-attention + adaptive gate</li><li>One anomaly score per snippet</li></ul>")
    with c[2]: card("3 · Decision", "<ul><li>Bag score = mean of top-k snippet scores (k = 3)</li><li>Threshold chosen on validation data</li><li>Gradient-reversal domain head used only in training</li><li>About 0.85 M trainable parameters</li></ul>")
    st.markdown("**Zero-shot prompts used as semantic priors**")
    from prompts import PROMPTS
    for p in PROMPTS:
        st.markdown(f"- “{p}”")
    st.markdown("**Training objective.** Weighted top-k BCE + 0.5 × ranking hinge + small magnitude, smoothness and sparsity terms + 0.12 × domain cross-entropy through the gradient-reversal layer. AdamW, cosine schedule, 45 epochs, three seeds.")

# ================================================================== TAB 5
with tabs[4]:
    st.subheader("What this demo can and cannot do")
    c = st.columns(2)
    with c[0]:
        card("Known limitations", """<ul>
        <li>In-domain accuracy is high, but <b>held-out-domain ROC-AUC is only about 0.55 to 0.70</b>; a new store may behave very differently.</li>
        <li>The test set is small (102 positive videos; 8 in UCF-Crime), so confidence intervals are wide.</li>
        <li>PG-MMIL is statistically tied with a CLIP prompt-conditioned baseline; the contribution of pose is not yet isolated by ablation.</li>
        <li>Splits remove exact duplicate bags only; near-duplicate clips could still inflate scores.</li>
        <li>Snippet scores are not validated for temporal localisation.</li>
        <li>Baselines are inspired re-implementations, not the published systems.</li></ul>""")
    with c[1]:
        card("Responsible use", """<ul>
        <li>An alert means <b>review</b>, never accusation. A false alarm can wrongly implicate an innocent customer.</li>
        <li>Keep a human in the loop and log decisions.</li>
        <li>No identity recognition is performed. Pose and CLIP features reduce but do not remove privacy risk.</li>
        <li>Only analyse footage you are authorised to use, and follow local data-protection law and signage rules.</li>
        <li>Demographic fairness has not been audited.</li></ul>""")
    st.caption("Research prototype for MAAI901 (University of Wollongong in Dubai). Not a commercial or enforcement product.")
