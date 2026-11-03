"""Feature extraction identical in design to the research notebook (Phase 3).

Per video: 32 uniformly spaced frames -> YOLO-pose (largest person, 17 keypoints x (x, y, conf) = 51-D)
-> person crop (or full frame) resized to 224x224 -> frozen CLIP ViT-B/32 (512-D, L2-normalised)
-> cosine similarity to 4 text prompts (4-D)  =>  32 x 567 feature bag.

Differences from the notebook, all inference-only: CPU is supported (FP32 instead of FP16), and the
function also returns boxes/keypoints for visualisation. The feature values are computed the same way.
"""
from pathlib import Path
import numpy as np
import cv2
import torch
from prompts import PROMPTS, PROMPT_LABELS  # noqa: F401

NUM_SNIPPETS = 32
FEATURE_DIM = 567
POSE_MODEL_PRIMARY = "yolo26n-pose.pt"      # recorded in feature_extraction_signature.json
POSE_MODEL_FALLBACK = "yolov8n-pose.pt"     # the notebook's own fallback
CLIP_NAME, CLIP_PRETRAINED = "ViT-B-32", "laion2b_s34b_b79k"
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


class Extractor:
    """Loads YOLO-pose and CLIP once; call .extract(video_path)."""

    def __init__(self, weights_dir=None, device=None):
        from ultralytics import YOLO
        import open_clip
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        wdir = Path(weights_dir) if weights_dir else None
        self.pose_model_name = None
        for name in (POSE_MODEL_PRIMARY, POSE_MODEL_FALLBACK):
            try:
                local = (wdir / name) if wdir and (wdir / name).exists() else name
                self.pose = YOLO(str(local))
                self.pose_model_name = name
                break
            except Exception:
                continue
        if self.pose_model_name is None:
            raise RuntimeError("Could not load a YOLO-pose model. Put yolo26n-pose.pt (or yolov8n-pose.pt) in weights/.")
        self.clip, _, _ = open_clip.create_model_and_transforms(CLIP_NAME, pretrained=CLIP_PRETRAINED, device=self.device)
        self.clip.eval()
        self.use_half = self.device == "cuda"
        if self.use_half:
            self.clip = self.clip.half()
        tok = open_clip.get_tokenizer(CLIP_NAME)
        with torch.no_grad():
            t = self.clip.encode_text(tok(PROMPTS).to(self.device))
            self.text = (t / t.norm(dim=-1, keepdim=True)).float()
        self.mean = torch.tensor(CLIP_MEAN, device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor(CLIP_STD, device=self.device).view(1, 3, 1, 1)

    # ---- decoding -------------------------------------------------------------------------------
    @staticmethod
    def decode_frames(video_path, num_snippets=NUM_SNIPPETS):
        cap = cv2.VideoCapture(str(video_path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
        if total <= 0:
            cap.release()
            raise ValueError("Video has no readable frames.")
        idx = np.linspace(0, max(total - 1, 0), num_snippets, dtype=int)
        wanted, last_idx = set(idx.tolist()), int(idx[-1])
        got, last_ok = {}, None
        for cur in range(last_idx + 1):
            if cur in wanted:
                ok, fr = cap.read()
                if ok and fr is not None:
                    last_ok = fr
                    got[cur] = fr
                elif last_ok is not None:
                    got[cur] = last_ok
            elif not cap.grab():
                break
        cap.release()
        if not got:
            raise ValueError("Could not decode any frame.")
        fallback = next(iter(got.values()))
        frames = [got.get(int(i), fallback) for i in idx]
        return frames, {"total_frames": total, "fps": fps, "duration_s": total / fps, "indices": idx.tolist()}

    # ---- features -------------------------------------------------------------------------------
    @torch.no_grad()
    def extract_bag(self, frames):
        n = len(frames)
        res = self.pose(frames, batch=n, imgsz=384, half=self.use_half, verbose=False)
        pose = np.zeros((n, 51), dtype=np.float32)
        boxes, crops = [None] * n, []
        for i, r in enumerate(res):
            fr = frames[i]
            crop = fr
            if r.boxes is not None and len(r.boxes) > 0 and r.keypoints is not None:
                xyxy = r.boxes.xyxy
                area = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
                b = int(torch.argmax(area).item())
                k = r.keypoints.xyn[b].detach().cpu().numpy()
                conf = (r.keypoints.conf[b].detach().cpu().numpy()[:, None]
                        if r.keypoints.conf is not None else np.ones((17, 1), dtype=np.float32))
                pose[i] = np.concatenate([k, conf], axis=-1).flatten()
                x1, y1, x2, y2 = map(int, xyxy[b].detach().cpu().numpy())
                boxes[i] = (x1, y1, x2, y2)
                c = fr[max(0, y1):max(1, y2), max(0, x1):max(1, x2)]
                crop = c if c.size else fr
            crops.append(cv2.cvtColor(cv2.resize(crop, (224, 224), interpolation=cv2.INTER_LINEAR), cv2.COLOR_BGR2RGB))
        batch = torch.from_numpy(np.stack(crops)).permute(0, 3, 1, 2).to(self.device)
        batch = (batch.half() if self.use_half else batch.float()) / 255.0
        batch = (batch - self.mean.to(batch.dtype)) / self.std.to(batch.dtype)
        v = self.clip.encode_image(batch)
        v = v / v.norm(dim=-1, keepdim=True)
        sims = (v.float() @ self.text.T).cpu().numpy()
        vis = v.float().cpu().numpy()
        feat = np.concatenate([pose, vis, sims], axis=-1).astype(np.float32)
        feat = np.nan_to_num(feat, nan=0.0, posinf=1.0, neginf=-1.0)
        assert feat.shape == (n, FEATURE_DIM), feat.shape
        return feat, boxes

    def extract(self, video_path):
        frames, meta = self.decode_frames(video_path)
        feat, boxes = self.extract_bag(frames)
        meta["pose_model"] = self.pose_model_name
        return {"frames": frames, "features": feat, "boxes": boxes, "meta": meta}
