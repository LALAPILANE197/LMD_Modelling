#!/usr/bin/env python3
"""
Revised porosity screening analysis (MEng thesis revision).

Run from the model folder:
    python porosity_revised.py                    # full run, 5 repeats
    python porosity_revised.py --fast             # quick check (LR + SVM, <=2 repeats)
    python porosity_revised.py --data processed_data.csv --out porosity_outputs --repeats 5

Requires: numpy, pandas, scikit-learn, matplotlib, statsmodels, imbalanced-learn
(SMOTE variants are skipped with a warning if imbalanced-learn is missing).

Parts: A original 80/20 hold-out (extended metrics, Wilson CIs); B repeated nested stratified 5-fold CV on all samples
(primary evidence); C SMOTE vs class-weight comparison; D bootstrap CIs + exploratory threshold analysis;
figures, tables and a run log are written to --out.
"""
import argparse, os, json, time, platform, warnings
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.metrics import (confusion_matrix, balanced_accuracy_score, average_precision_score,
                             roc_auc_score, roc_curve, precision_recall_curve)
from statsmodels.stats.proportion import proportion_confint
warnings.filterwarnings("ignore")

try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
    SMOTE_OK = True
except ImportError:
    SMOTE_OK = False
    print("WARNING: imbalanced-learn not found -> SMOTE variants are skipped (pip install imbalanced-learn).")


def main():
    ap = argparse.ArgumentParser(description="Revised porosity screening analysis (nested CV, class-weight vs SMOTE, extended metrics).")
    ap.add_argument("--data", default=None, help="path to processed_data.csv (default: searched automatically)")
    ap.add_argument("--out", default="porosity_outputs", help="output folder")
    ap.add_argument("--repeats", type=int, default=5, help="outer repeats for nested CV")
    ap.add_argument("--fast", action="store_true", help="LR + SVM only, at most 2 repeats")
    ap.add_argument("--boot", type=int, default=2000, help="bootstrap resamples")
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [args.data, "processed_data.csv", os.path.join(here, "processed_data.csv"),
                  "~/Documents/PlayTime/LMD_Modelling/model/processed_data.csv",
                  "~/Documents/PlayTime/LMD_Modelling/data/processed/processed_data.csv"]
    DATA_PATH = next((c for c in candidates if c and os.path.exists(os.path.expanduser(c))), None)
    if DATA_PATH is None:
        raise SystemExit("processed_data.csv not found. Pass --data /path/to/processed_data.csv")
    OUT_DIR = args.out
    FEATURES = ["Power", "Speed", "Feed"]
    SEED = 42
    N_REPEATS = args.repeats
    FAST_MODE = args.fast
    INNER_SCORING = "balanced_accuracy"   # model selection criterion (avoids majority-class bias)
    N_BOOT = args.boot
    if FAST_MODE:
        N_REPEATS = min(N_REPEATS, 2)
    os.makedirs(OUT_DIR, exist_ok=True)
    print("sklearn", sklearn.__version__, "| SMOTE available:", SMOTE_OK, "| FAST_MODE:", FAST_MODE, "| repeats:", N_REPEATS)

    # ---------- 1. load the corrected dataset and define the label ----------
    path = os.path.expanduser(DATA_PATH)
    df = pd.read_csv(path)
    print("loaded:", path, df.shape)

    # guard: the feed column must NOT be truncated (earlier bug: 4.5 -> 4)
    n_half = int(((df["Feed"] % 1) == 0.5).sum())
    assert n_half > 0, "Feed has no x.5 values: the truncated dataset is still being used!"
    print(f"Feed check passed: {n_half} rows have x.5 g/min values | Feed range {df['Feed'].min()}-{df['Feed'].max()}")

    X = df[FEATURES].reset_index(drop=True)
    y = (df["porosity_%"] > 0).astype(int).reset_index(drop=True)     # 1 = porosity detected, 0 = not detected
    N = len(y)
    counts = y.value_counts().sort_index()
    baseline_acc = max(y.mean(), 1 - y.mean())
    print(f"\nN = {N} | not detected (0): {counts.get(0, 0)} | detected (1): {counts.get(1, 0)}")
    print(f"Majority-class baseline accuracy on all samples: {baseline_acc:.3f}")

    # ---------- 2. helper functions ----------
    def wilson(k, n):
        if n == 0: return (np.nan, np.nan)
        return proportion_confint(k, n, method="wilson")

    def metric_row(y_true, y_pred, y_proba):
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        n = len(y_true)
        return dict(TN=tn, FP=fp, FN=fn, TP=tp,
                    accuracy=(tp + tn) / n,
                    balanced_accuracy=balanced_accuracy_score(y_true, y_pred),
                    recall_porous=tp / (tp + fn) if (tp + fn) else np.nan,
                    specificity=tn / (tn + fp) if (tn + fp) else np.nan,
                    precision_porous=tp / (tp + fp) if (tp + fp) else np.nan,
                    missed_porous_pct=100 * fn / (tp + fn) if (tp + fn) else np.nan,
                    PR_AUC=average_precision_score(y_true, y_proba),
                    ROC_AUC=roc_auc_score(y_true, y_proba) if len(np.unique(y_true)) > 1 else np.nan)

    def make_models():
        """name -> (estimator, param_grid). Imbalance handled either by SMOTE (inside training folds) or by class weights."""
        M = {}
        M["Majority class"] = (DummyClassifier(strategy="most_frequent"), {})
        M["LR | class weight"] = (Pipeline([("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED))]),
            {"lr__C": [0.01, 0.1, 1, 10, 100]})
        M["SVM | class weight"] = (Pipeline([("scaler", StandardScaler()),
            ("svm", SVC(kernel="rbf", class_weight="balanced", probability=True, random_state=SEED))]),
            {"svm__C": [0.1, 1, 10, 100], "svm__gamma": ["scale", 0.01, 0.1]})
        if SMOTE_OK:
            M["LR | SMOTE"] = (ImbPipeline([("smote", SMOTE(random_state=SEED)), ("scaler", StandardScaler()),
                ("lr", LogisticRegression(max_iter=2000, random_state=SEED))]),
                {"lr__C": [0.01, 0.1, 1, 10, 100], "smote__k_neighbors": [3, 5]})
            M["SVM | SMOTE"] = (ImbPipeline([("smote", SMOTE(random_state=SEED)), ("scaler", StandardScaler()),
                ("svm", SVC(kernel="rbf", probability=True, random_state=SEED))]),
                {"svm__C": [0.1, 1, 10, 100], "svm__gamma": ["scale", 0.01, 0.1], "smote__k_neighbors": [3, 5]})
        if not FAST_MODE:
            M["RF | class weight"] = (RandomForestClassifier(class_weight="balanced", random_state=SEED, n_jobs=1),
                {"n_estimators": [200], "max_depth": [None, 5, 10], "min_samples_leaf": [1, 2, 4]})
            if SMOTE_OK:
                M["RF | SMOTE"] = (ImbPipeline([("smote", SMOTE(random_state=SEED)),
                    ("rf", RandomForestClassifier(random_state=SEED, n_jobs=1))]),
                    {"rf__n_estimators": [200], "rf__max_depth": [None, 5, 10], "rf__min_samples_leaf": [1, 2, 4],
                     "smote__k_neighbors": [3, 5]})
                M["GB | SMOTE"] = (ImbPipeline([("smote", SMOTE(random_state=SEED)),
                    ("gb", GradientBoostingClassifier(random_state=SEED))]),
                    {"gb__n_estimators": [100, 200], "gb__learning_rate": [0.05, 0.1], "gb__max_depth": [2, 3],
                     "smote__k_neighbors": [3, 5]})
                M["MLP | SMOTE"] = (ImbPipeline([("smote", SMOTE(random_state=SEED)), ("scaler", StandardScaler()),
                    ("mlp", MLPClassifier(max_iter=2000, early_stopping=False, random_state=SEED))]),
                    {"mlp__hidden_layer_sizes": [(16,), (50,)], "mlp__alpha": [1e-3, 1e-2],
                     "mlp__activation": ["tanh"], "smote__k_neighbors": [3, 5]})
            else:
                M["GB | none"] = (GradientBoostingClassifier(random_state=SEED),
                    {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1], "max_depth": [2, 3]})
        return M

    MODELS = make_models()
    print(len(MODELS), "model configurations:", list(MODELS))

    # ---------- PART A: original 80/20 hold-out (kept for continuity with Table 4-11) ----------
    # Only 18 test samples: one sample moves accuracy by 5.6 percentage points. Wilson 95% intervals are reported.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)
    print(f"train {len(y_train)} (0:{(y_train==0).sum()}, 1:{(y_train==1).sum()}) | test {len(y_test)} (0:{(y_test==0).sum()}, 1:{(y_test==1).sum()})")
    n_t = len(y_test); base_t = max(y_test.mean(), 1 - y_test.mean())
    cv_inner = StratifiedKFold(5, shuffle=True, random_state=SEED)

    rowsA, hold_models = [], {}
    for name, (est, grid) in MODELS.items():
        gs = GridSearchCV(clone(est), grid, cv=cv_inner, scoring=INNER_SCORING, n_jobs=-1).fit(X_train, y_train)
        hold_models[name] = gs.best_estimator_
        pred = gs.predict(X_test); proba = gs.predict_proba(X_test)[:, 1]
        r = metric_row(y_test, pred, proba)
        lo, hi = wilson(r["TP"] + r["TN"], n_t); rlo, rhi = wilson(r["TP"], r["TP"] + r["FN"])
        rowsA.append(dict(model=name, best_params=str(gs.best_params_), cv_balanced_acc=gs.best_score_, **r,
                          accuracy_CI95=f"{lo:.2f}-{hi:.2f}", recall_CI95=f"{rlo:.2f}-{rhi:.2f}", majority_baseline=base_t))
    tableA = pd.DataFrame(rowsA)
    cols = ["model", "TN", "FP", "FN", "TP", "accuracy", "accuracy_CI95", "majority_baseline", "recall_porous", "recall_CI95",
            "missed_porous_pct", "specificity", "precision_porous", "balanced_accuracy", "PR_AUC", "ROC_AUC", "cv_balanced_acc"]
    print(tableA[cols].round(3).to_string(index=False))
    tableA.to_csv(f"{OUT_DIR}/A_holdout_extended.csv", index=False)

    # ---------- PART B (primary evidence): repeated nested stratified 5-fold CV on all samples ----------
    # Tuning (inner 5-fold, balanced accuracy) happens only inside each outer training fold; SMOTE is fitted on training data only.
    Xv, yv = X.values, y.values
    oof_pred  = {m: np.zeros((N_REPEATS, N), dtype=int) for m in MODELS}
    oof_proba = {m: np.zeros((N_REPEATS, N)) for m in MODELS}
    t0 = time.time()
    for r in range(N_REPEATS):
        outer = StratifiedKFold(5, shuffle=True, random_state=SEED + r)
        for k, (tr, te) in enumerate(outer.split(Xv, yv)):
            for name, (est, grid) in MODELS.items():
                gs = GridSearchCV(clone(est), grid, cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
                                  scoring=INNER_SCORING, n_jobs=-1).fit(Xv[tr], yv[tr])
                oof_pred[name][r, te]  = gs.predict(Xv[te])
                oof_proba[name][r, te] = gs.predict_proba(Xv[te])[:, 1]
        print(f"repeat {r + 1}/{N_REPEATS} done | elapsed {time.time() - t0:.0f}s", flush=True)

    metric_cols = ["accuracy", "balanced_accuracy", "recall_porous", "specificity", "precision_porous", "missed_porous_pct", "PR_AUC", "ROC_AUC"]
    rows, per_repeat = [], {}
    for m in MODELS:
        d = pd.DataFrame([metric_row(yv, oof_pred[m][r], oof_proba[m][r]) for r in range(N_REPEATS)])
        per_repeat[m] = d
        row = {"model": m, "majority_baseline": baseline_acc}
        for c in ["TN", "FP", "FN", "TP"]: row[c + "_mean"] = d[c].mean()
        for c in metric_cols:
            row[c + "_mean"] = d[c].mean(); row[c + "_sd"] = d[c].std(ddof=1) if N_REPEATS > 1 else np.nan
        rows.append(row)
    tableB = pd.DataFrame(rows)
    fmt = lambda a, b: f"{a:.3f} ({b:.3f})" if not np.isnan(b) else f"{a:.3f}"
    thesis_B = pd.DataFrame({"model": tableB["model"],
        "accuracy": [fmt(a, b) for a, b in zip(tableB.accuracy_mean, tableB.accuracy_sd)],
        "majority baseline": tableB["majority_baseline"].round(3),
        "balanced acc.": [fmt(a, b) for a, b in zip(tableB.balanced_accuracy_mean, tableB.balanced_accuracy_sd)],
        "recall (porous)": [fmt(a, b) for a, b in zip(tableB.recall_porous_mean, tableB.recall_porous_sd)],
        "specificity": [fmt(a, b) for a, b in zip(tableB.specificity_mean, tableB.specificity_sd)],
        "missed porous %": tableB["missed_porous_pct_mean"].round(1),
        "PR-AUC": [fmt(a, b) for a, b in zip(tableB.PR_AUC_mean, tableB.PR_AUC_sd)],
        "ROC-AUC": [fmt(a, b) for a, b in zip(tableB.ROC_AUC_mean, tableB.ROC_AUC_sd)]})
    print(thesis_B.to_string(index=False))
    tableB.round(4).to_csv(f"{OUT_DIR}/B_nested_cv_summary_numeric.csv", index=False)
    thesis_B.to_csv(f"{OUT_DIR}/B_nested_cv_summary_thesis.csv", index=False)

    cand = tableB[tableB.model != "Majority class"].sort_values(["balanced_accuracy_mean", "recall_porous_mean"], ascending=False)
    best = cand.iloc[0]["model"]
    print("\nBest configuration by mean balanced accuracy:", best)

    # ---------- PART C: SMOTE vs class weights ----------
    tmp = tableB[tableB.model != "Majority class"].copy()
    tmp[["clf", "strategy"]] = tmp.model.str.split(" \\| ", expand=True)
    cmp_rows = []
    for clf, g in tmp.groupby("clf"):
        if g.strategy.nunique() > 1:
            for s, rr in g.set_index("strategy").iterrows():
                cmp_rows.append(dict(classifier=clf, strategy=s, balanced_accuracy=rr.balanced_accuracy_mean,
                                     recall_porous=rr.recall_porous_mean, specificity=rr.specificity_mean, PR_AUC=rr.PR_AUC_mean))
    tableC = pd.DataFrame(cmp_rows)
    print(tableC.round(3).to_string(index=False) if len(tableC) else "No classifier has both variants in this run (SMOTE unavailable or --fast).")
    tableC.round(4).to_csv(f"{OUT_DIR}/C_imbalance_strategy_comparison.csv", index=False)

    # ---------- PART D: bootstrap intervals and exploratory threshold analysis (best configuration) ----------
    # Repeat-averaged out-of-fold probabilities. The threshold table is EXPLORATORY and has not been validated independently.
    p_avg = oof_proba[best].mean(axis=0)
    pred05 = (p_avg >= 0.5).astype(int)
    rng = np.random.default_rng(SEED)
    bs = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, N, N)
        if len(np.unique(yv[idx])) < 2: continue
        tn, fp, fn, tp = confusion_matrix(yv[idx], pred05[idx], labels=[0, 1]).ravel()
        bs.append((balanced_accuracy_score(yv[idx], pred05[idx]), tp / (tp + fn) if tp + fn else np.nan,
                   tn / (tn + fp) if tn + fp else np.nan))
    bs = np.array(bs)
    ci = lambda a: f"{np.nanpercentile(a, 2.5):.2f}-{np.nanpercentile(a, 97.5):.2f}"
    print(f"{best}: balanced acc. 95% bootstrap CI {ci(bs[:,0])} | recall {ci(bs[:,1])} | specificity {ci(bs[:,2])}")

    rows = []
    for th in np.arange(0.10, 0.91, 0.05):
        pr = (p_avg >= th).astype(int)
        tn, fp, fn, tp = confusion_matrix(yv, pr, labels=[0, 1]).ravel()
        rows.append(dict(threshold=round(th, 2), recall_porous=tp / (tp + fn), specificity=tn / (tn + fp) if tn + fp else np.nan,
                         missed_porous_pct=100 * fn / (tp + fn), false_alarms=int(fp), balanced_accuracy=balanced_accuracy_score(yv, pr)))
    tableD = pd.DataFrame(rows)
    print(tableD.round(3).to_string(index=False))
    tableD.round(4).to_csv(f"{OUT_DIR}/D_threshold_analysis_exploratory.csv", index=False)

    # ---------- FIGURES (300 dpi) ----------
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    order = tableB.sort_values("balanced_accuracy_mean")
    ax[0].barh(order.model, order.balanced_accuracy_mean, xerr=order.balanced_accuracy_sd.fillna(0), color="#3498db", edgecolor="k", capsize=3)
    ax[0].axvline(0.5, c="k", ls="--", lw=1, label="chance / majority class"); ax[0].set_xlabel("Balanced accuracy (nested CV)"); ax[0].legend()
    ax[1].barh(order.model, order.recall_porous_mean, xerr=order.recall_porous_sd.fillna(0), color="#e67e22", edgecolor="k", capsize=3)
    ax[1].set_xlabel("Recall for porous class (nested CV)"); ax[1].set_xlim(0, 1.05)
    for a in ax: a.grid(alpha=.3, axis="x")
    plt.tight_layout(); plt.savefig(f"{OUT_DIR}/fig1_balanced_accuracy_recall.png", dpi=300); plt.close("all")

    fpr, tpr, _ = roc_curve(yv, p_avg); prec, rec, _ = precision_recall_curve(yv, p_avg)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
    ax[0].plot(fpr, tpr, lw=2, label=f"{best} (AUC {roc_auc_score(yv, p_avg):.2f})"); ax[0].plot([0, 1], [0, 1], "k--")
    ax[0].set_xlabel("False-positive rate"); ax[0].set_ylabel("True-positive rate"); ax[0].legend(loc="lower right"); ax[0].grid(alpha=.3)
    ax[1].plot(rec, prec, lw=2, label=f"AP {average_precision_score(yv, p_avg):.2f}"); ax[1].axhline(yv.mean(), c="k", ls="--", label=f"prevalence {yv.mean():.2f}")
    ax[1].set_xlabel("Recall"); ax[1].set_ylabel("Precision"); ax[1].legend(); ax[1].grid(alpha=.3)
    plt.tight_layout(); plt.savefig(f"{OUT_DIR}/fig2_roc_pr_best.png", dpi=300); plt.close("all")

    tb = tableB.set_index("model")
    cm = np.array([[tb.loc[best, "TN_mean"], tb.loc[best, "FP_mean"]], [tb.loc[best, "FN_mean"], tb.loc[best, "TP_mean"]]])
    fig, ax = plt.subplots(figsize=(4.4, 3.8)); ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["pred: not detected", "pred: detected"]); ax.set_yticks([0, 1]); ax.set_yticklabels(["true: not detected", "true: detected"])
    for i in range(2):
        for j in range(2): ax.text(j, i, f"{cm[i, j]:.1f}", ha="center", va="center", fontsize=12)
    ax.set_title(f"{best}\nmean counts over {N_REPEATS} repeats"); plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/fig3_confusion_best.png", dpi=300); plt.close("all")

    # ---------- THESIS-READY FACTS and reproducibility log ----------
    rb = tableB.set_index("model").loc[best]
    print("=== THESIS-READY FACTS (check against the tables above) ===")
    print(f"Class counts: {int(counts.get(1,0))} porosity detected, {int(counts.get(0,0))} not detected (N = {N}).")
    print(f"Majority-class baseline accuracy: {baseline_acc:.3f} (all samples); hold-out baseline {base_t:.3f} (n = {n_t}).")
    print(f"Best nested-CV configuration: {best}")
    print(f"  accuracy {rb.accuracy_mean:.3f} | balanced accuracy {rb.balanced_accuracy_mean:.3f} | porous recall {rb.recall_porous_mean:.3f} "
          f"| specificity {rb.specificity_mean:.3f} | PR-AUC {rb.PR_AUC_mean:.3f}")
    print(f"  about {rb.missed_porous_pct_mean:.0f}% of porous samples were missed at the default 0.5 threshold.")
    print("Wording reminders: say 'porosity detected / not detected' (not acceptable / unacceptable); "
          "do not claim a planar boundary or a 'classification ceiling'; describe the reformulation as exploratory; "
          "the 80/20 hold-out has n = 18, so its metrics are indicative only.")

    log = dict(timestamp=time.strftime("%Y-%m-%d %H:%M:%S"), python=platform.python_version(), sklearn=sklearn.__version__,
               numpy=np.__version__, pandas=pd.__version__, smote_available=SMOTE_OK, data_path=path, n_samples=N,
               features=FEATURES, seed=SEED, n_repeats=N_REPEATS, inner_scoring=INNER_SCORING, fast_mode=FAST_MODE,
               feed_half_values=n_half, best_configuration=str(best))
    json.dump(log, open(f"{OUT_DIR}/run_log.json", "w"), indent=2)
    print("\nOutputs written to ./" + OUT_DIR + ":", sorted(os.listdir(OUT_DIR)))


if __name__ == "__main__":
    main()