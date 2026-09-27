"""Train-only model selection and one-time held-out test for MSTP."""
import argparse
import hashlib
import json
import platform
from pathlib import Path

import joblib
import librosa
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

LABELS = ["Prolongation", "Block", "SoundRep", "WordRep", "Interjection"]
REPRESENTATIONS = ("stats", "pattern", "combined")


def load_manifest(path):
    df = pd.read_csv(path, dtype={"audio_path": str, "group_id": str, "split": str})
    required = ["audio_path", "group_id", "split", *LABELS]
    if any(c not in df for c in required) or df[required].isna().any().any():
        raise ValueError(f"Manifest needs complete columns: {required}")
    if not set(df.split).issubset({"train", "dev", "test"}) or set(df.split) != {"train", "dev", "test"}:
        raise ValueError("All train/dev/test splits must occur, with no other split names")
    if df.groupby("group_id").split.nunique().max() != 1:
        raise ValueError("A group_id occurs in more than one split")
    counts = df[LABELS].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(counts.to_numpy()).all() or ((counts < 0) | (counts > 3) | (counts % 1 != 0)).any().any():
        raise ValueError("Label counts must be integers in [0,3]")
    paths = [(path.parent / p).resolve() for p in df.audio_path]
    if len(set(paths)) != len(paths) or not all(p.is_file() for p in paths):
        raise ValueError("Audio paths must exist and occur once only")
    hashes = []
    for p in paths:
        h = hashlib.sha256()
        with p.open("rb") as f:
            for block in iter(lambda: f.read(1024 * 1024), b""):
                h.update(block)
        hashes.append(h.hexdigest())
    for _, rows in pd.DataFrame({"hash": hashes, "split": df.split}).groupby("hash"):
        if rows.split.nunique() > 1:
            raise ValueError("Identical audio bytes occur in different splits")
    df["resolved_path"] = list(map(str, paths))
    return df, (counts.to_numpy() > 0).astype(int), hashes


def hist_codes(logmel, axis, offset):
    # Coordinates compare four offsets around a center; counts normalized to 1.
    if axis == "time":
        step = offset
        if logmel.shape[1] <= 4 * step:
            return np.zeros(81)
        center = logmel[:, 2*step:-2*step]
        neighbors = [logmel[:, (2+d)*step:logmel.shape[1]-(2-d)*step or None] for d in (-2,-1,1,2)]
    else:
        step = max(1, offset // 4)
        center = logmel[2*step:-2*step, :]
        neighbors = [logmel[(2+d)*step:logmel.shape[0]-(2-d)*step or None, :] for d in (-2,-1,1,2)]
    diffs = np.stack([n-center for n in neighbors], axis=0)
    threshold = 0.5 * float(np.std(diffs))
    ternary = (diffs > threshold).astype(np.int8) + (diffs >= -threshold).astype(np.int8)
    code = np.sum(ternary.astype(np.int32) * (3 ** np.arange(4))[:, None, None], axis=0)
    hist = np.bincount(code.ravel(), minlength=81).astype(np.float64)
    return hist / max(hist.sum(), 1)


def extract(path):
    y, sr = librosa.load(path, sr=16000, mono=True)
    if len(y) < 1600 or not np.isfinite(y).all():
        raise ValueError(f"Audio too short or non-finite: {path}")
    power = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=400, hop_length=160,
                                             win_length=400, n_mels=80, power=2.0)
    mel = np.log1p(power)
    stats = np.concatenate([fn(mel, axis=1) for fn in
                            (np.mean, np.std, lambda x, axis: np.quantile(x, .1, axis=axis),
                             lambda x, axis: np.quantile(x, .9, axis=axis))])
    pattern = np.concatenate([hist_codes(mel, axis, offset)
                              for offset in (4, 8, 16) for axis in ("time", "freq")])
    return stats.astype(np.float32), pattern.astype(np.float32)


def train_one(x, y, k):
    if len(np.unique(y)) < 2:
        return {"constant": float(y[0])}
    actual_k = min(k, x.shape[1])
    model = Pipeline([
        ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scale", StandardScaler()),
        ("select", SelectKBest(f_classif, k=actual_k)),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=2026)),
    ])
    model.fit(x, y)
    return model


def predict_models(models, x):
    return np.column_stack([np.full(len(x), m["constant"]) if isinstance(m, dict)
                            else m.predict_proba(x)[:, 1] for m in models])


def ap_report(y, scores):
    per = {}
    for j, label in enumerate(LABELS):
        per[label] = (float(average_precision_score(y[:, j], scores[:, j]))
                      if y[:, j].sum() > 0 else None)
    values = [v for v in per.values() if v is not None]
    return {"macro_ap": float(np.mean(values)) if values else None,
            "per_label_ap": per, "positive_counts": dict(zip(LABELS, map(int, y.sum(0))))}


def thresholds_from_dev(y, scores):
    thresholds = []
    for j in range(len(LABELS)):
        # A single-class dev label cannot supply a useful tuned threshold.
        if len(np.unique(y[:, j])) < 2:
            thresholds.append(0.5)
            continue
        candidates = np.unique(np.r_[0.5, scores[:, j], 1.0])
        f1s = [f1_score(y[:, j], scores[:, j] >= t, zero_division=0) for t in candidates]
        thresholds.append(float(candidates[int(np.argmax(f1s))]))
    return np.array(thresholds)


def test_report(y, scores, thresholds):
    result = ap_report(y, scores)
    pred = scores >= thresholds
    result["thresholds_from_dev"] = dict(zip(LABELS, map(float, thresholds)))
    result["per_label"] = {}
    for j, label in enumerate(LABELS):
        result["per_label"][label] = {
            "f1": float(f1_score(y[:, j], pred[:, j], zero_division=0)),
            "precision": float(precision_score(y[:, j], pred[:, j], zero_division=0)),
            "recall": float(recall_score(y[:, j], pred[:, j], zero_division=0)),
            "roc_auc": float(roc_auc_score(y[:, j], scores[:, j])) if len(np.unique(y[:, j])) == 2 else None}
    result["macro_f1"] = float(np.mean([v["f1"] for v in result["per_label"].values()]))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    if out.exists():
        raise FileExistsError(f"Choose a fresh output directory: {out}")
    df, labels, hashes = load_manifest(args.manifest.resolve())
    out.mkdir(parents=True)
    audit = {"n": len(df), "groups_by_split": df.groupby("split").group_id.nunique().to_dict(),
             "clips_by_split": df.split.value_counts().to_dict(), "sha256": hashes,
             "positive_counts_by_split": {s: dict(zip(LABELS, map(int, labels[df.split == s].sum(0))))
                                          for s in ("train", "dev", "test")}}
    (out / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    stats, pattern = [], []
    for i, p in enumerate(df.resolved_path):
        a, b = extract(p)
        stats.append(a); pattern.append(b)
        if (i+1) % 100 == 0:
            print(f"Extracted {i+1}/{len(df)}", flush=True)
    stats, pattern = np.stack(stats), np.stack(pattern)
    np.savez_compressed(out / "features.npz", stats=stats, pattern=pattern, labels=labels,
                        paths=df.audio_path.to_numpy(), split=df.split.to_numpy())
    matrices = {"stats": stats, "pattern": pattern, "combined": np.hstack((stats, pattern))}
    tr, dv, te = (np.flatnonzero(df.split.to_numpy() == s) for s in ("train", "dev", "test"))
    comparisons = []
    for rep, x in matrices.items():
        for k in (64, 128, 256):
            models = [train_one(x[tr], labels[tr, j], k) for j in range(len(LABELS))]
            scores = predict_models(models, x[dv])
            report = ap_report(labels[dv], scores)
            comparisons.append({"representation": rep, "k": k, "dev_macro_ap": report["macro_ap"],
                                **{f"dev_ap_{name}": report["per_label_ap"][name] for name in LABELS}})
            print(comparisons[-1], flush=True)
    pd.DataFrame(comparisons).to_csv(out / "dev_comparison.csv", index=False)
    valid = [r for r in comparisons if r["dev_macro_ap"] is not None]
    if not valid:
        raise ValueError("Dev contains no positive labels; cannot select a model")
    best = max(valid, key=lambda r: r["dev_macro_ap"])
    x = matrices[best["representation"]]
    # Dev thresholds correspond exactly to train-only-fitted model; no refitting on dev.
    models = [train_one(x[tr], labels[tr, j], best["k"]) for j in range(len(LABELS))]
    dev_scores = predict_models(models, x[dv])
    thresholds = thresholds_from_dev(labels[dv], dev_scores)
    test_scores = predict_models(models, x[te])
    report = test_report(labels[te], test_scores, thresholds)
    (out / "test_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    predictions = df.iloc[te][["audio_path", "group_id"]].copy()
    for j, label in enumerate(LABELS):
        predictions[f"true_{label}"] = labels[te, j]
        predictions[f"score_{label}"] = test_scores[:, j]
    predictions.to_csv(out / "test_predictions.csv", index=False)
    joblib.dump({"models": models, "thresholds": thresholds,
                 "representation": best["representation"], "k": best["k"], "labels": LABELS}, out / "model.joblib")
    config = {"selection": best, "sampling_rate": 16000, "seed": 2026,
              "python": platform.python_version(), "numpy": np.__version__,
              "librosa": librosa.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__}
    (out / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print("Finished; selected:", best, "test macro AP:", report["macro_ap"])


if __name__ == "__main__":
    main()
