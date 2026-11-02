"""Matplotlib helpers for the Streamlit UI (no torch/streamlit imports so they are easy to test)."""
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TEAL, AMBER, CORAL, INK, GRAY, LIGHT = "#0B6E6E", "#F2A33A", "#E4572E", "#14213D", "#8A94A6", "#E8EEF2"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": GRAY, "axes.labelcolor": INK,
                     "xtick.color": INK, "ytick.color": INK, "axes.spines.top": False, "axes.spines.right": False})

# COCO-17 skeleton
EDGES = [(0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12),
         (11, 13), (13, 15), (12, 14), (14, 16)]


def score_bar(score, threshold):
    """Horizontal 0-1 scale showing the bag score against the decision threshold."""
    fig, ax = plt.subplots(figsize=(7, 1.25))
    ax.barh([0], [1], color=LIGHT, height=0.5)
    ax.barh([0], [threshold], color="#CFE5E3", height=0.5)
    flagged = score >= threshold
    ax.barh([0], [score], color=CORAL if flagged else TEAL, height=0.5)
    ax.axvline(threshold, color=INK, lw=2, ymin=0.1, ymax=0.9)
    ax.text(threshold, 0.42, f"threshold {threshold:.3f}", ha="center", va="bottom", fontsize=8.5, color=INK)
    ax.text(min(score, 0.97), -0.42, f"score {score:.3f}", ha="center", va="top", fontsize=9, color=INK, fontweight="bold")
    ax.set_xlim(0, 1); ax.set_ylim(-0.7, 0.75); ax.set_yticks([])
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1]); ax.spines["left"].set_visible(False)
    fig.tight_layout()
    return fig


def snippet_timeline(scores, threshold, k, duration_s=None):
    """Per-snippet anomaly scores; the top-k snippets used for the bag score are highlighted."""
    scores = np.asarray(scores, dtype=float)
    t = np.arange(len(scores))
    xs = t if not duration_s else t * duration_s / max(len(scores) - 1, 1)
    top = np.argsort(scores)[::-1][:k]
    fig, ax = plt.subplots(figsize=(7.5, 2.8))
    ax.fill_between(xs, scores, color=TEAL, alpha=0.15)
    ax.plot(xs, scores, color=TEAL, lw=2, marker="o", ms=3.5)
    ax.scatter(xs[top], scores[top], s=70, color=CORAL, zorder=5, label=f"top-{k} snippets (bag score)")
    ax.axhline(threshold, color=INK, ls="--", lw=1.2, label=f"threshold {threshold:.3f}")
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel("time (s)" if duration_s else "snippet index"); ax.set_ylabel("snippet score")
    ax.grid(alpha=0.25); ax.legend(fontsize=8, frameon=False, loc="upper right", ncol=2)
    fig.tight_layout()
    return fig


def prompt_heatmap(sims, labels):
    sims = np.asarray(sims, dtype=float)
    fig, ax = plt.subplots(figsize=(7.5, 2.3))
    im = ax.imshow(sims.T, aspect="auto", cmap="magma")
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=8.5)
    ax.set_xlabel("snippet index")
    for s in ax.spines.values():
        s.set_visible(False)
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01).set_label("CLIP cosine", fontsize=8)
    fig.tight_layout()
    return fig


def wrist_speed(pose_raw, duration_s=None):
    """Normalised wrist speed per snippet from the (x, y, conf) keypoints, hip-centred. Returns (fig, speed)."""
    p = np.asarray(pose_raw, dtype=float).reshape(len(pose_raw), 17, 3)
    xy, conf = p[..., :2], p[..., 2]
    hip = 0.5 * (xy[:, 11] + xy[:, 12])
    rel = xy - hip[:, None, :]
    w = rel[:, [9, 10], :]
    ok = (conf[:, [9, 10]] > 0.1)
    speed = np.zeros(len(p))
    for i in range(1, len(p)):
        d = np.linalg.norm(w[i] - w[i - 1], axis=-1)
        m = ok[i] & ok[i - 1]
        speed[i] = d[m].max() if m.any() else 0.0
    if speed.max() > 0:
        speed = speed / speed.max()
    xs = np.arange(len(p)) if not duration_s else np.arange(len(p)) * duration_s / max(len(p) - 1, 1)
    fig, ax = plt.subplots(figsize=(7.5, 2.2))
    ax.fill_between(xs, speed, color=AMBER, alpha=0.25)
    ax.plot(xs, speed, color=AMBER, lw=2, marker="s", ms=3)
    ax.set_ylim(0, 1.05); ax.set_ylabel("wrist speed (norm.)"); ax.set_xlabel("time (s)" if duration_s else "snippet index")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig, speed


def annotate_frame(frame_bgr, pose_row, box, label=None, max_w=360):
    """Draw person box + skeleton on a frame; returns RGB uint8 resized to max_w."""
    img = frame_bgr.copy()
    h, w = img.shape[:2]
    if box is not None:
        x1, y1, x2, y2 = box
        cv2.rectangle(img, (x1, y1), (x2, y2), (60, 160, 242), max(2, w // 300))
    kp = np.asarray(pose_row, dtype=float).reshape(17, 3)
    if kp[:, 2].max() > 0:
        pts = [(int(x * w), int(y * h), c) for x, y, c in kp]
        for a, b in EDGES:
            if pts[a][2] > 0.3 and pts[b][2] > 0.3:
                cv2.line(img, pts[a][:2], pts[b][:2], (110, 110, 11), max(2, w // 250))
        for x, y, c in pts:
            if c > 0.3:
                cv2.circle(img, (x, y), max(3, w // 160), (46, 87, 228), -1)
    if label:
        cv2.rectangle(img, (0, 0), (min(w, 12 + 11 * len(label)), 30), (0, 0, 0), -1)
        cv2.putText(img, label, (6, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    scale = max_w / w
    img = cv2.resize(img, (max_w, int(h * scale)))
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def metric_bars(df, metric_cols, names, highlight):
    """Grouped small-multiples bar chart from a dataframe indexed by model."""
    fig, axs = plt.subplots(1, len(metric_cols), figsize=(3.6 * len(metric_cols), 3.2))
    axs = np.atleast_1d(axs)
    for ax, (col, title, lower_better) in zip(axs, metric_cols):
        vals = df[col].values
        colors = [CORAL if m == highlight else TEAL for m in df.index]
        ax.bar(range(len(vals)), vals, color=colors, edgecolor=INK, lw=0.5)
        ax.set_xticks(range(len(vals))); ax.set_xticklabels([names.get(m, m).replace(" ", "\n", 1) for m in df.index], fontsize=6.5)
        ax.set_title(title + (" (lower is better)" if lower_better else ""), fontsize=9, fontweight="bold")
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    return fig
