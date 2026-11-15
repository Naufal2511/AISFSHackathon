"""Option B predictor for src/behavior/appearance.py.

    from predict import predict
    probs = predict(crops)   # (N, 10), order = taxonomy.BEHAVIORS, rows sum to 1

crops: list of BGR uint8 arrays (one crop per box)  -> frame head
       or list of lists of BGR arrays (2-16 crops of the same cow over ~2-5 s) -> clip head (better
       for walking / rumination / grooming / social, which are motion behaviors)
Crops should be the padded box (~10-60% context). IR night frames can be grey; they are expanded to 3 channels.
"""
import os
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_STATE = {}


def _load(path=None):
    if "model" not in _STATE:
        import joblib, sys as _sys
        from train_behavior import Backbone, Head
        _sys.modules["__main__"].Head = Head  # model was pickled from train_behavior run as a script
        m = joblib.load(path or os.path.join(_HERE, "beh_model.joblib"))
        _STATE.update(model=m, bb=Backbone(m["backbone"]))
    return _STATE["model"], _STATE["bb"]


def _rgb(a):
    a = np.asarray(a)
    if a.ndim == 2:
        a = np.repeat(a[..., None], 3, 2)
    return a[..., ::-1].copy()  # BGR -> RGB


def _softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def _load_ft():
    if "ft" not in _STATE:
        import torch, timm
        ck = torch.load(os.path.join(_HERE, "beh_finetuned.pt"), map_location="cpu")
        dev = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
        m = timm.create_model(ck["backbone"], pretrained=False, img_size=224, num_classes=10)
        m.load_state_dict(ck["state_dict"]); m.eval().to(dev)
        _STATE["ft"] = (m, ck, dev, torch)
    return _STATE["ft"]


def _use_ft():
    f = os.path.join(_HERE, "chosen.txt")
    return os.path.exists(f) and open(f).read().strip() == "finetuned" and os.path.exists(os.path.join(_HERE, "beh_finetuned.pt"))


def _predict_ft(crops):
    from PIL import Image
    m, ck, dev, torch = _load_ft()
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1); std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    def batch(imgs):
        x = np.stack([np.asarray(Image.fromarray(_rgb(c)).resize((224, 224), Image.BILINEAR)) for c in imgs]).astype(np.float32) / 255
        x = (torch.from_numpy(x).permute(0, 3, 1, 2) - mean) / std
        with torch.no_grad():
            return m(x.to(dev)).float().cpu().numpy()
    if isinstance(crops[0], (list, tuple)):
        z = np.stack([batch(s).mean(0) for s in crops])
    else:
        z = batch(crops)
    return _softmax(z / ck["T"]).astype(np.float32)


def predict(crops, model_path=None):
    """Uses the fine-tuned model if chosen.txt says so, else the frozen-probe model."""
    from train_behavior import clip_feature
    if len(crops) and model_path is None and _use_ft():
        return _predict_ft(crops)
    m, bb = _load(model_path)
    if len(crops) == 0:
        return np.zeros((0, 10), np.float32)
    if isinstance(crops[0], (list, tuple)):
        X = np.stack([clip_feature(bb.embed([_rgb(c) for c in stack])) for stack in crops])
        return _softmax(m["clip_head"].logits(X) / m["T_clip"]).astype(np.float32)
    X = bb.embed([_rgb(c) for c in crops])
    return _softmax(m["frame_head"].logits(X) / m["T_frame"]).astype(np.float32)


VERSION = "beh-dinov2s-probe-v1"
