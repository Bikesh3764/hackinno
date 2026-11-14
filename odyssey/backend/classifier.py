"""
Terrain classification (ML) with uncertainty decomposition.

Model: Random-Forest ensemble on per-pixel spectral + geomorphic + context features.
Why RF: runs on a weak flight CPU, needs little labelled data, is interpretable
(feature importances), and the tree ensemble gives us *epistemic* uncertainty for free.

Uncertainty:
  total     H[ E_t p_t ]                (predictive entropy)
  aleatoric E_t H[p_t]                  (inherent ambiguity, e.g. dust-covered rock)
  epistemic total - aleatoric           (mutual information = trees disagree -> unknown terrain)
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from sklearn.ensemble import RandomForestClassifier

from terrain_sim import CLASSES

FEATURE_NAMES = [
    "B1/mean", "B2/mean", "B3/mean", "B4/mean", "B5/mean", "B6/mean", "brightness",
    "BD1.9um (hydration)", "BD2.3um (clay OH)", "red slope B3/B1", "thermal inertia",
    "slope", "roughness", "curvature", "relative elevation", "texture (local std)",
    "ctx BD1.9um", "ctx thermal inertia", "ctx roughness",
]


def build_features(world, geo) -> np.ndarray:
    b = world.bands.astype(np.float64)
    bright = b.mean(0)
    norm = b / bright                         # removes illumination / shading
    bd19 = 1 - b[4] / (0.5 * (b[3] + b[5]))   # 1.9 um band depth vs continuum
    bd23 = 1 - b[5] / b[3]
    red = b[2] / b[0]
    ti = world.thermal_obs / 1000.0
    texture = np.sqrt(np.clip(ndi.uniform_filter(bright ** 2, 5) - ndi.uniform_filter(bright, 5) ** 2, 0, None))
    feats = [*norm, bright, bd19, bd23, red, ti,
             geo["slope"] / 30, geo["rough"], geo["curv"] * 10, geo["rel_elev"] / 10, texture * 10,
             ndi.gaussian_filter(bd19, 2), ndi.gaussian_filter(ti, 2), ndi.gaussian_filter(geo["rough"], 2)]
    X = np.stack(feats, -1).reshape(-1, len(feats)).astype(np.float32)
    return X


def sample_training_patches(world, per_class=7, half=2, label_noise=0.04, seed=0):
    """Simulates an expert annotation campaign: a few small labelled patches per class
    (~4% of the map) with some label noise. Everything else is unlabelled."""
    rng = np.random.default_rng(seed + 101)
    n = world.n
    truth = world.true_class
    lab = np.full((n, n), -1, np.int16)
    pid = np.full((n, n), -1, np.int32)
    k = 0
    for c in range(len(CLASSES)):
        rr, cc = np.nonzero(truth == c)
        if len(rr) == 0:
            continue
        pick = rng.choice(len(rr), size=min(per_class, len(rr)), replace=False)
        for i in pick:
            r0, c0 = rr[i], cc[i]
            sl = (slice(max(r0 - half, 0), r0 + half + 1), slice(max(c0 - half, 0), c0 + half + 1))
            lab[sl] = truth[sl]
            pid[sl] = k
            k += 1
    m = lab >= 0
    flip = m & (rng.random((n, n)) < label_noise)
    lab[flip] = rng.integers(0, len(CLASSES), flip.sum())
    return lab, pid


def train_and_predict(world, geo, n_trees=80, seed=0):
    n, C = world.n, len(CLASSES)
    X = build_features(world, geo)
    lab, pid = sample_training_patches(world, seed=seed)
    lab, pid = lab.ravel(), pid.ravel()
    tr = lab >= 0
    rf = RandomForestClassifier(n_estimators=n_trees, max_depth=16, min_samples_leaf=2,
                                max_features="sqrt", oob_score=True, n_jobs=-1,
                                class_weight="balanced", random_state=seed)
    rf.fit(X[tr], lab[tr])

    # per-tree probabilities -> uncertainty decomposition
    classes = rf.classes_
    mean_p = np.zeros((X.shape[0], C), np.float64)
    exp_ent = np.zeros(X.shape[0])
    sq = np.zeros((X.shape[0], C))
    for t in rf.estimators_:
        p = np.zeros((X.shape[0], C))
        p[:, classes] = t.predict_proba(X)
        mean_p += p
        sq += p ** 2
        exp_ent += -(p * np.log(p + 1e-12)).sum(1)
    T = len(rf.estimators_)
    mean_p /= T
    exp_ent /= T
    std_p = np.sqrt(np.clip(sq / T - mean_p ** 2, 0, None))
    total = -(mean_p * np.log(mean_p + 1e-12)).sum(1)
    epi = np.clip(total - exp_ent, 0, None)
    lnC = np.log(C)

    # ---- calibration: temperature scaling fitted on GROUPED cross-validation
    # (whole labelled patches held out -> no spatial leakage; no test data used).
    raw_ece_probs = mean_p.copy()
    y = lab[tr]
    groups = pid[tr]
    Xtr = X[tr]
    cv = np.zeros((len(y), C))
    folds = np.unique(groups) % 4
    fold_of = folds[np.searchsorted(np.unique(groups), groups)]
    for f in range(4):
        te_m = fold_of == f
        if te_m.all() or not te_m.any():
            continue
        m_ = RandomForestClassifier(n_estimators=40, max_depth=16, min_samples_leaf=2, max_features="sqrt",
                                    n_jobs=-1, class_weight="balanced", random_state=seed + f)
        m_.fit(Xtr[~te_m], y[~te_m])
        cv[np.ix_(te_m, m_.classes_)] = m_.predict_proba(Xtr[te_m])
    best_T, best_nll = 1.0, np.inf
    for T in np.linspace(0.15, 1.5, 28):
        q = _temper(cv, T)
        nll = -np.log(q[np.arange(len(y)), y] + 1e-9).mean()
        if nll < best_nll:
            best_T, best_nll = T, nll
    mean_p = _temper(mean_p, best_T)
    total = -(mean_p * np.log(mean_p + 1e-12)).sum(1)
    exp_ent = np.clip(total - epi, 0, None)  # aleatoric = total - epistemic after calibration

    pred = mean_p.argmax(1)
    conf = mean_p.max(1)
    truth = world.true_class.ravel()
    ev = ~tr
    metrics = _metrics(pred[ev], truth[ev], conf[ev], C)
    metrics["oob_accuracy"] = float(rf.oob_score_)
    metrics["temperature"] = float(best_T)
    metrics["ece_uncalibrated"] = _metrics(raw_ece_probs.argmax(1)[ev], truth[ev], raw_ece_probs.max(1)[ev], C)["ece"]
    metrics["n_train_pixels"] = int(tr.sum())
    metrics["train_fraction"] = float(tr.mean())
    imp = rf.feature_importances_
    order = np.argsort(imp)[::-1]
    metrics["feature_importance"] = [[FEATURE_NAMES[i], float(imp[i])] for i in order[:10]]
    # does uncertainty mean something? accuracy on most-confident X% of pixels
    metrics["selective"] = _selective(pred[ev], truth[ev], total[ev])

    sh = (n, n)
    return dict(
        probs=mean_p.reshape(n, n, C).astype(np.float32),
        probs_std=std_p.reshape(n, n, C).astype(np.float32),
        pred=pred.reshape(sh),
        entropy=(total / lnC).reshape(sh),
        aleatoric=(exp_ent / lnC).reshape(sh),
        epistemic=(epi / lnC).reshape(sh),
        train_mask=tr.reshape(sh),
        metrics=metrics,
        model=rf,
    )


def _temper(p, T):
    """Temperature scaling on probabilities: p^(1/T), renormalised (T<1 sharpens)."""
    q = np.power(np.clip(p, 1e-6, 1), 1.0 / T)
    return q / q.sum(1, keepdims=True)


def _metrics(pred, truth, conf, C):
    cm = np.zeros((C, C), int)
    np.add.at(cm, (truth, pred), 1)
    acc = float((pred == truth).mean())
    per = []
    for c in range(C):
        tp = cm[c, c]
        prec = tp / max(cm[:, c].sum(), 1)
        rec = tp / max(cm[c, :].sum(), 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        per.append(dict(cls=CLASSES[c], precision=float(prec), recall=float(rec), f1=float(f1),
                        support=int(cm[c, :].sum())))
    # expected calibration error
    bins = np.linspace(0, 1, 11)
    ece = 0.0
    rel = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            a, c_ = (pred[m] == truth[m]).mean(), conf[m].mean()
            ece += m.mean() * abs(a - c_)
            rel.append([float(c_), float(a), int(m.sum())])
    return dict(accuracy=acc, confusion=cm.tolist(), per_class=per, ece=float(ece),
                reliability=rel, macro_f1=float(np.mean([p["f1"] for p in per])))


def _selective(pred, truth, unc):
    order = np.argsort(unc)
    correct = (pred == truth)[order]
    out = []
    for cov in (0.25, 0.5, 0.7, 0.9, 1.0):
        k = max(int(cov * len(order)), 1)
        out.append([cov, float(correct[:k].mean())])
    return out
