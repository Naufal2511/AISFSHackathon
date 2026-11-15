#!/usr/bin/env python3
"""Train the per-box behavior classifier (beh-dinov2s-probe-v1). Runs LOCALLY on event data.

Design (see report.md written at the end):
  backbone : DINOv2 ViT-S/14 (timm vit_small_patch14_dinov2.lvd142m, Apache-2.0), frozen, 224 px
  feature  : per crop  = [CLS | mean patch]                      (768)
             per clip  = [mean | std | mean |frame-to-frame diff|] over the 5 crops (2304)
  head     : multinomial logistic regression, balanced class weights, C picked by grouped CV
  calib    : temperature fitted on out-of-fold probabilities (grouped by source clip)
  outputs  : 10 probabilities in taxonomy order (BEHAVIORS below), summing to 1

Inputs  : newest exports/pen6_behavior_export_*.json (+ any extra export given with --export),
          data/behavior_annotation/crops/<task>/f*.jpg, manifest.csv (night/ir)
          optional --extra-dir <ImageFolder of public crops, e.g. CVB> capped per class
Outputs : behavior_model/{beh_model.joblib, report.md, metrics.json, oof_predictions.csv}
"""
import argparse, glob, hashlib, json, os, sys, time
import numpy as np

BEHAVIORS = ["lying", "standing", "walking", "feeding", "drinking", "ruminating (lying)",
             "ruminating (standing)", "grooming", "social interaction", "other / not visible"]
SHORT = ["p_lying", "p_standing", "p_walking", "p_feeding", "p_drinking", "p_rum_lying",
         "p_rum_standing", "p_grooming", "p_social", "p_other"]
FOLDER = {"lying": 0, "standing": 1, "walking": 2, "feeding": 3, "drinking": 4, "ruminating_lying": 5,
          "ruminating_standing": 6, "grooming": 7, "social_interaction": 8, "other_not_visible": 9}
VERSION = "beh-dinov2s-probe-v1"
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)  # pen6_annotation


# ---------------------------------------------------------------- data
def load_labels(paths):
    """task_id -> dict(label, camera, clip, split, flags). Latest annotation wins; later files win."""
    tasks = {}
    for p in paths:
        for t in json.load(open(p)):
            anns = [a for a in t.get("annotations", []) if not a.get("was_cancelled")]
            if not anns:
                continue
            a = max(anns, key=lambda a: a.get("updated_at") or "")
            beh, flags = None, []
            for r in a["result"]:
                if r.get("from_name") == "behavior":
                    beh = r["value"]["choices"][0]
                elif r.get("from_name") == "quality":
                    flags += r["value"]["choices"]
            if beh is None:
                continue
            d = t["data"]
            tasks[d["task_id"]] = dict(label=BEHAVIORS.index(beh), camera=d.get("camera", "?"),
                                       clip=d.get("clip", d["task_id"]), split=d.get("split", "train"),
                                       flags=flags)
    keep = {k: v for k, v in tasks.items()
            if not any(f.startswith("bad track") or f == "unsure" for f in v["flags"])}
    return keep, len(tasks) - len(keep)


def load_night(root):
    night = {}
    for m in glob.glob(os.path.join(root, "data", "behavior_annotation", "manifest*.csv")):
        import csv
        for row in csv.DictReader(open(m)):
            night[row["task_id"]] = str(row.get("ir", row.get("night", ""))).lower() == "true"
    return night


# ---------------------------------------------------------------- backbone
class Backbone:
    def __init__(self, name):
        import torch, timm
        self.torch = torch
        self.dev = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
        if name == "stub":  # pipeline test without downloading weights
            self.m = timm.create_model("vit_small_patch14_dinov2", pretrained=False, img_size=224, num_classes=0)
        else:
            self.m = timm.create_model(name, pretrained=True, img_size=224, num_classes=0)
        self.m.eval().to(self.dev)
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.dev)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.dev)

    def embed(self, rgb_list, bs=32):
        """rgb_list: HxWx3 uint8 RGB arrays (any size). Returns (N, 768) = [CLS | mean patch]."""
        from PIL import Image
        torch = self.torch
        out = []
        for i in range(0, len(rgb_list), bs):
            x = np.stack([np.asarray(Image.fromarray(a).convert("RGB").resize((224, 224), Image.BICUBIC))
                          for a in rgb_list[i:i + bs]]).astype(np.float32) / 255.0
            x = torch.from_numpy(x).permute(0, 3, 1, 2).to(self.dev)
            x = (x - self.mean) / self.std
            with torch.no_grad():
                tok = self.m.forward_features(x)  # (B, 1+reg+patches, 384)
                cls = tok[:, 0]
                patches = tok[:, 1 + getattr(self.m, "num_reg_tokens", 0):].mean(1)
                out.append(torch.cat([cls, patches], 1).float().cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, 768), np.float32)


def clip_feature(frame_feats):
    """(T, 768) per-frame features -> (2304,) clip feature: mean, std, mean abs consecutive diff."""
    f = np.asarray(frame_feats, np.float32)
    diff = np.abs(np.diff(f, axis=0)).mean(0) if len(f) > 1 else np.zeros(f.shape[1], np.float32)
    return np.concatenate([f.mean(0), f.std(0), diff])


def read_rgb(path):
    from PIL import Image
    return np.asarray(Image.open(path).convert("RGB"))


# ---------------------------------------------------------------- calibration / metrics
def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def fit_temperature(logits, y):
    best_t, best = 1.0, 1e9
    for t in np.exp(np.linspace(np.log(0.3), np.log(5.0), 120)):
        p = softmax(logits / t)
        nll = -np.mean(np.log(p[np.arange(len(y)), y] + 1e-12))
        if nll < best:
            best_t, best = t, nll
    return float(best_t)


def ece(p, y, bins=10):
    conf, pred = p.max(1), p.argmax(1)
    e = 0.0
    for lo in np.linspace(0, 1, bins + 1)[:-1]:
        m = (conf > lo) & (conf <= lo + 1 / bins)
        if m.any():
            e += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return float(e)


def metrics(p, y, names=BEHAVIORS):
    from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
    pred = p.argmax(1)
    present = sorted(set(y.tolist()))
    pr, rc, f1, sup = precision_recall_fscore_support(y, pred, labels=list(range(len(names))), zero_division=0)
    return dict(n=int(len(y)), accuracy=float((pred == y).mean()),
                macro_f1=float(np.mean([f1[i] for i in present])) if present else 0.0,
                per_class={names[i]: dict(precision=round(float(pr[i]), 3), recall=round(float(rc[i]), 3),
                                          f1=round(float(f1[i]), 3), support=int(sup[i])) for i in range(len(names))},
                confusion=confusion_matrix(y, pred, labels=list(range(len(names)))).tolist(),
                ece=ece(p, y), nll=float(-np.mean(np.log(p[np.arange(len(y)), y] + 1e-12))))


# ---------------------------------------------------------------- head
class Head:
    """Standardise -> logistic regression; predict_logits works for any class subset seen in training."""

    def __init__(self, C):
        self.C = C

    def fit(self, X, y, w=None):
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        self.sc = StandardScaler().fit(X)
        self.lr = LogisticRegression(C=self.C, max_iter=3000, class_weight="balanced")
        self.lr.fit(self.sc.transform(X), y, sample_weight=w)
        return self

    def logits(self, X):
        z = np.full((len(X), len(BEHAVIORS)), -30.0, np.float32)  # unseen classes ~0 prob
        z[:, self.lr.classes_] = self.lr.decision_function(self.sc.transform(X)).reshape(len(X), -1) \
            if len(self.lr.classes_) > 2 else np.stack([-self.lr.decision_function(self.sc.transform(X)),
                                                       self.lr.decision_function(self.sc.transform(X))], 1)
        return z


def grouped_oof(X, y, groups, C, folds=5, extra=None, aug=None):
    """Out-of-fold logits, grouped by source clip so windows of one clip never straddle folds."""
    from sklearn.model_selection import StratifiedGroupKFold
    k = max(2, min(folds, len(set(groups))))
    oof = np.zeros((len(y), len(BEHAVIORS)), np.float32)
    for tr, te in StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=0).split(X, y, groups):
        Xtr, ytr = X[tr], y[tr]
        if aug is not None:
            Xtr, ytr = np.concatenate([Xtr, aug[tr]]), np.concatenate([ytr, y[tr]])
        if extra is not None:
            Xtr, ytr = np.concatenate([Xtr, extra[0]]), np.concatenate([ytr, extra[1]])
        oof[te] = Head(C).fit(Xtr, ytr).logits(X[te])
    return oof


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", nargs="*", default=None, help="behavior export(s); default: all in exports/")
    ap.add_argument("--backbone", default="vit_small_patch14_dinov2.lvd142m")
    ap.add_argument("--extra-dir", default=None, help="ImageFolder of public crops (e.g. CVB) by class folder")
    ap.add_argument("--extra-cap", type=int, default=400)
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--out", default=HERE, help="where model/report go")
    ap.add_argument("--hflip", action="store_true", help="add horizontally flipped crops as training augmentation")
    a = ap.parse_args()
    t0 = time.time()
    out_dir = a.out
    os.makedirs(out_dir, exist_ok=True)
    exports = a.export or sorted(glob.glob(os.path.join(a.root, "exports", "pen6_behavior_export_*.json")))
    if not exports:
        sys.exit("No behavior export found in exports/. Run export_labels.command first.")
    tasks, dropped = load_labels(exports)
    night = load_night(a.root)
    crops_root = os.path.join(a.root, "data", "behavior_annotation", "crops")
    ids = sorted(t for t in tasks if glob.glob(os.path.join(crops_root, t, "f*.jpg")))
    print(f"{len(ids)} labelled tasks with crops ({dropped} dropped as bad track/unsure)")

    bb = Backbone(a.backbone)
    print("backbone on", bb.dev)
    cache = os.path.join(HERE, f"features_{a.backbone.split('.')[0]}.npz")
    cached = dict(np.load(cache, allow_pickle=True)) if os.path.exists(cache) else {}
    frame_feats = {}
    t_emb = time.time(); n_crops = 0
    for i, tid in enumerate(ids):
        if tid in cached and tid + "__flip" in cached:
            frame_feats[tid] = cached[tid]; frame_feats[tid + "__flip"] = cached[tid + "__flip"]; continue
        files = sorted(glob.glob(os.path.join(crops_root, tid, "f*.jpg")))
        imgs = [read_rgb(f) for f in files]
        frame_feats[tid] = bb.embed(imgs); n_crops += len(files)
        frame_feats[tid + "__flip"] = bb.embed([im[:, ::-1].copy() for im in imgs])
        if i % 25 == 0:
            print(f"  embedded {i}/{len(ids)}")
    if n_crops:
        ms_per_crop = 1000 * (time.time() - t_emb) / n_crops
    else:
        ms_per_crop = float("nan")
    np.savez(cache, **frame_feats)

    y = np.array([tasks[t]["label"] for t in ids])
    groups = np.array([tasks[t]["clip"] for t in ids])
    split = np.array([tasks[t]["split"] for t in ids])
    cams = np.array([tasks[t]["camera"] for t in ids])
    nights = np.array(["night/IR" if night.get(t) else "day" for t in ids])
    Xclip = np.stack([clip_feature(frame_feats[t]) for t in ids])
    # frame-level view: each crop is a sample carrying its task label (single-frame head)
    Xfr = np.concatenate([frame_feats[t] for t in ids]); yfr = np.repeat(y, [len(frame_feats[t]) for t in ids])
    gfr = np.repeat(groups, [len(frame_feats[t]) for t in ids])
    aug_clip = np.stack([clip_feature(frame_feats[t + "__flip"]) for t in ids]) if a.hflip else None
    aug_fr = np.concatenate([frame_feats[t + "__flip"] for t in ids]) if a.hflip else None

    extra_clip = extra_fr = None
    if a.extra_dir and os.path.isdir(a.extra_dir):
        ex, ey = [], []
        for folder, ci in FOLDER.items():
            fs = sorted(glob.glob(os.path.join(a.extra_dir, folder, "*.jpg")))[: a.extra_cap]
            if fs:
                ex.append(bb.embed([read_rgb(f) for f in fs])); ey += [ci] * len(fs)
        if ex:
            exf = np.concatenate(ex); ey = np.array(ey)
            extra_fr = (exf, ey)
            extra_clip = (np.stack([clip_feature(np.stack([f, f])) for f in exf]), ey)  # still image: no motion
            print(f"extra public crops: {len(ey)}")

    global VERSION
    VERSION = {"vit_small": "beh-dinov2s-probe-v1", "vit_base": "beh-dinov2b-probe-v1", "vit_large": "beh-dinov2l-probe-v1"}.get(
        a.backbone.split("_patch")[0], "beh-probe-v1")
    report = {"version": VERSION, "backbone": a.backbone, "hflip": bool(a.hflip), "n_tasks": int(len(ids)),
              "label_counts": {BEHAVIORS[i]: int((y == i).sum()) for i in range(10)},
              "ms_per_crop_embed": round(ms_per_crop, 1), "device": bb.dev}

    # pick C by grouped CV macro-F1 (clip head), then compare heads
    Cs = [0.01, 0.03, 0.1, 0.3, 1.0]
    best = None
    for C in Cs:
        oof = grouped_oof(Xclip, y, groups, C, extra=extra_clip, aug=aug_clip)
        m = metrics(softmax(oof), y)
        print(f"  clip head C={C}: OOF macro-F1 {m['macro_f1']:.3f} acc {m['accuracy']:.3f}")
        if best is None or m["macro_f1"] > best[1]:
            best = (C, m["macro_f1"], oof)
    C, _, oof_clip = best
    T = fit_temperature(oof_clip, y)
    oof_fr = grouped_oof(Xfr, yfr, gfr, C, extra=extra_fr, aug=aug_fr)
    T_fr = fit_temperature(oof_fr, yfr)
    # frame head evaluated per task: average its per-frame probabilities
    idx = np.cumsum([0] + [len(frame_feats[t]) for t in ids])
    p_fr_task = np.stack([softmax(oof_fr[idx[i]:idx[i + 1]] / T_fr).mean(0) for i in range(len(ids))])

    report["C"] = C
    report["temperature_clip"] = T
    report["temperature_frame"] = T_fr
    report["oof_clip_raw"] = metrics(softmax(oof_clip), y)
    report["oof_clip_calibrated"] = metrics(softmax(oof_clip / T), y)
    report["oof_frame_head_task_avg"] = metrics(p_fr_task, y)
    report["oof_single_crop"] = metrics(softmax(oof_fr / T_fr), yfr)
    maj = np.bincount(y, minlength=10).argmax()
    report["baseline_majority"] = metrics(np.eye(10)[np.full(len(y), maj)] * 0.999 + 0.0001, y)
    for name, key in [("camera", cams), ("light", nights)]:
        report[f"oof_clip_by_{name}"] = {v: metrics(softmax(oof_clip[key == v] / T), y[key == v])
                                        for v in sorted(set(key))}

    # official val split: train on train split only
    tr, va = split == "train", split == "val"
    if va.any() and len(set(y[tr])) > 1:
        Xtr, ytr = Xclip[tr], y[tr]
        if aug_clip is not None:
            Xtr, ytr = np.concatenate([Xtr, aug_clip[tr]]), np.concatenate([ytr, y[tr]])
        if extra_clip is not None:
            Xtr, ytr = np.concatenate([Xtr, extra_clip[0]]), np.concatenate([ytr, extra_clip[1]])
        lv = Head(C).fit(Xtr, ytr).logits(Xclip[va])
        report["val_split_clip_calibrated"] = metrics(softmax(lv / T), y[va])

    # deploy: fit on everything
    Xall, yall = Xclip, y
    Xfa, yfa = Xfr, yfr
    if aug_clip is not None:
        Xall, yall = np.concatenate([Xall, aug_clip]), np.concatenate([yall, y])
        Xfa, yfa = np.concatenate([Xfa, aug_fr]), np.concatenate([yfa, yfr])
    if extra_clip is not None:
        Xall, yall = np.concatenate([Xall, extra_clip[0]]), np.concatenate([yall, extra_clip[1]])
        Xfa, yfa = np.concatenate([Xfa, extra_fr[0]]), np.concatenate([yfa, extra_fr[1]])
    import joblib
    joblib.dump({"version": VERSION, "backbone": a.backbone, "classes": BEHAVIORS, "short": SHORT,
                 "clip_head": Head(C).fit(Xall, yall), "frame_head": Head(C).fit(Xfa, yfa),
                 "T_clip": T, "T_frame": T_fr, "input": "RGB crop, padded box, resized 224x224, ImageNet norm; IR grey -> 3 channels"},
                os.path.join(out_dir, "beh_model.joblib"))
    import csv
    with open(os.path.join(out_dir, "oof_predictions.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["task_id", "camera", "light", "label", "pred", "conf"] + SHORT)
        pc = softmax(oof_clip / T)
        for i, tid in enumerate(ids):
            w.writerow([tid, cams[i], nights[i], BEHAVIORS[y[i]], BEHAVIORS[pc[i].argmax()], round(float(pc[i].max()), 3)]
                       + [round(float(v), 4) for v in pc[i]])
    json.dump(report, open(os.path.join(out_dir, "metrics.json"), "w"), indent=1)
    write_report(report, os.path.join(out_dir, "report.md"))
    print(f"\nDone in {time.time() - t0:.0f}s. See behavior_model/report.md")


def write_report(r, path):
    L = [f"# Behavior classifier {r['version']}", "",
         f"Backbone `{r['backbone']}` (frozen) on `{r['device']}`, {r['ms_per_crop_embed']} ms per crop to embed. "
         f"Head: logistic regression, C={r['C']}, temperature {r['temperature_clip']:.2f} (clip) / {r['temperature_frame']:.2f} (frame).",
         f"Tasks: {r['n_tasks']}. Labels: " + ", ".join(f"{k} {v}" for k, v in r["label_counts"].items() if v), "",
         "Numbers are out-of-fold (5-fold, grouped by source clip), so no clip is ever scored by a model that saw it.", "",
         "| Model | n | macro-F1 | accuracy | ECE | NLL |", "|---|---|---|---|---|---|"]
    for k, lab in [("baseline_majority", "majority class"), ("oof_single_crop", "single crop (frame head)"),
                   ("oof_frame_head_task_avg", "frame head, avg of 5 crops"), ("oof_clip_raw", "clip head, raw"),
                   ("oof_clip_calibrated", "clip head, temperature-scaled"), ("val_split_clip_calibrated", "clip head on official val split")]:
        if k in r:
            m = r[k]; L.append(f"| {lab} | {m['n']} | {m['macro_f1']:.3f} | {m['accuracy']:.3f} | {m['ece']:.3f} | {m['nll']:.3f} |")
    m = r["oof_clip_calibrated"]
    L += ["", "## Per class (clip head, calibrated, OOF)", "", "| class | support | precision | recall | F1 |", "|---|---|---|---|---|"]
    for c, v in m["per_class"].items():
        if v["support"]:
            L.append(f"| {c} | {v['support']} | {v['precision']} | {v['recall']} | {v['f1']} |")
    L += ["", "## Confusion (rows = truth, cols = predicted, order = class list)", "", "```"]
    L += [" ".join(f"{x:3d}" for x in row) for row in m["confusion"]] + ["```", ""]
    for name in ["camera", "light"]:
        L += [f"## By {name}", "", "| group | n | macro-F1 | accuracy | ECE |", "|---|---|---|---|---|"]
        for g, v in r[f"oof_clip_by_{name}"].items():
            L.append(f"| {g} | {v['n']} | {v['macro_f1']:.3f} | {v['accuracy']:.3f} | {v['ece']:.3f} |")
        L.append("")
    open(path, "w").write("\n".join(L))


if __name__ == "__main__":
    main()
