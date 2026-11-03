"""Checkpoint discovery, loading and scoring. Torch is imported lazily."""
import re
from pathlib import Path
import numpy as np

CKPT_RE = re.compile(r"^(?P<name>.+)_seed(?P<seed>\d+)\.pth$")


def find_checkpoints(models_dir):
    """Return {(model_name, seed): Path} for files saved by the notebook as '<model>_seed<seed>.pth'."""
    out = {}
    d = Path(models_dir)
    if d.exists():
        for p in sorted(d.rglob("*.pth")):
            m = CKPT_RE.match(p.name)
            if m and not p.name.startswith("LODO_"):
                out[(m.group("name"), int(m.group("seed")))] = p
    return out


def load_model(name, path, device="cpu"):
    """Load a notebook checkpoint: {'state_dict','seed','threshold','checkpoint_metric','history'}."""
    import torch
    from model_def import MODEL_REGISTRY
    factory, k, label = MODEL_REGISTRY[name]
    ckpt = torch.load(str(path), map_location=device, weights_only=True)
    model = factory().to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, {"threshold": float(ckpt.get("threshold", 0.5)), "k": k, "label": label,
                   "seed": ckpt.get("seed"), "checkpoint_metric": ckpt.get("checkpoint_metric")}


def score_bag(model, feats, k, device="cpu"):
    """feats: (32, 567) float array -> (snippet_scores (32,), bag_score)."""
    import torch
    x = torch.from_numpy(np.asarray(feats, dtype=np.float32)).unsqueeze(0).to(device)
    with torch.no_grad():
        scores, _, _ = model(x, alpha=0.0)
        topk, _ = torch.topk(scores, k=min(k, scores.shape[1]), dim=1)
        bag = float(topk.mean(dim=1).item())
    return scores.squeeze(0).cpu().numpy(), bag
