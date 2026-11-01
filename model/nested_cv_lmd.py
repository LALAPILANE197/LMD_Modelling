"""Nested-CV evaluation + reporting for LMD surrogate models (MEng thesis revision).
Usage:  python nested_cv_lmd.py --data processed_data.csv --repeats 5 --out report_outputs [--ablation] [--learning] [--fast]
"""
import argparse, os, warnings
os.environ["PYTHONWARNINGS"] = "ignore"   # silences GPR warnings in joblib workers
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.stats.multitest import multipletests
from sklearn.model_selection import RepeatedKFold, KFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from sklearn.metrics import mean_absolute_error as MAE, mean_squared_error, r2_score
warnings.filterwarnings("ignore")

SEED, INNER_CV, PRIMARY = 42, 5, "neg_mean_absolute_error"
FEATURES = ["Power", "Speed", "Feed"]
TARGETS = {"height_um": "Height (µm)", "aspect_ratio": "Aspect ratio (-)", "dilution_percent": "Dilution (%)"}
LEVELS = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]

def make_X(df, with_pv=False):
    X = df[FEATURES].to_numpy(float)
    return np.column_stack([X, X[:, 0] / X[:, 1]]) if with_pv else X

def wrap(est, grid):
    pipe = Pipeline([("sx", StandardScaler()), ("m", est)])
    ttr = TransformedTargetRegressor(regressor=pipe, transformer=StandardScaler())
    return ttr, {f"regressor__m__{k}": v for k, v in grid.items()}

def spaces(fast=False):
    s = {
     "Ridge": (Ridge(), {"alpha": [1e-3, 1e-2, 1e-1, 1, 10, 100]}),
     "GPR": (GPR(n_restarts_optimizer=5, random_state=SEED),
             {"alpha": [1e-10, 1e-5, 1e-2], "kernel": [C(1.0)*RBF(1.0), C(1.0)*RBF(1.0)+WhiteKernel(1.0)]}),
     "SVR": (SVR(), {"C": [0.1, 1, 10, 100], "epsilon": [0.01, 0.1, 0.2], "gamma": ["scale", "auto"]}),
    }
    if not fast:
        s["RF"] = (RandomForestRegressor(random_state=SEED),
                   {"n_estimators": [100, 200, 300], "max_depth": [None, 10, 20],
                    "min_samples_leaf": [1, 2, 4], "max_features": ["sqrt", "log2"]})
        s["ANN"] = (MLPRegressor(early_stopping=True, validation_fraction=0.15, n_iter_no_change=20,
                                 max_iter=1000, random_state=SEED),
                    {"hidden_layer_sizes": [(64, 32), (128, 64), (64, 32, 16)], "alpha": [1e-4, 1e-3, 1e-2],
                     "learning_rate_init": [1e-3, 1e-2], "activation": ["relu", "tanh"]})
    return s

def gpr_predict_with_std(gs, Xte):
    b = gs.best_estimator_; pipe = b.regressor_
    mu, sd = pipe[-1].predict(pipe[:-1].transform(Xte), return_std=True)
    s = b.transformer_.scale_[0]
    return b.transformer_.mean_[0] + mu*s, sd*s

def nested_cv(X, y, models, n_repeats=5):
    splits = list(RepeatedKFold(n_splits=5, n_repeats=n_repeats, random_state=SEED).split(X))
    rows = []
    oof = {m: np.full((n_repeats, len(y)), np.nan) for m in models}
    z = []                                   # standardised GPR residuals (y-mu)/sd
    best_params = {m: [] for m in models}
    for name, (est, grid) in models.items():
        for k, (tr, te) in enumerate(splits):
            ttr, g = wrap(est, grid)
            gs = GridSearchCV(ttr, g, cv=INNER_CV, scoring=PRIMARY, n_jobs=-1).fit(X[tr], y[tr])
            p = gs.predict(X[te]); oof[name][k // 5, te] = p
            rows.append(dict(model=name, repeat=k // 5, fold=k % 5, MAE=MAE(y[te], p),
                             RMSE=mean_squared_error(y[te], p)**0.5, R2=r2_score(y[te], p)))
            best_params[name].append(str({a.split("__")[-1]: v for a, v in gs.best_params_.items()}))
            if name == "GPR":
                mu, sd = gpr_predict_with_std(gs, X[te]); z += list((y[te]-mu)/sd)
    return pd.DataFrame(rows), oof, np.array(z), best_params

def summarise(folds, oof, y):
    R = next(iter(oof.values())).shape[0]
    g = folds.groupby("model")[["MAE", "RMSE", "R2"]].agg(["mean", "std"])
    g.columns = [f"{a}_{b}" for a, b in g.columns]
    g["MAE_pct_of_mean_y"] = 100*g["MAE_mean"]/np.mean(y)
    g["R2_pooled"] = [np.mean([r2_score(y, oof[m][r]) for r in range(R)]) for m in g.index]
    return g.sort_values("MAE_mean").round(4)

def rank_stability(oof, y):
    R = next(iter(oof.values())).shape[0]; names = list(oof)
    mae = pd.DataFrame({m: [MAE(y, oof[m][r]) for r in range(R)] for m in names})
    ranks = mae.rank(axis=1)
    return pd.DataFrame({m: [(ranks[m] == k).mean() for k in range(1, len(names)+1)] for m in names},
                        index=[f"rank {k}" for k in range(1, len(names)+1)]).T

def paired_wilcoxon_holm(oof, y):
    names = list(oof); rows = []
    for i in range(len(names)):
        for j in range(i+1, len(names)):
            d = np.abs(y-np.nanmean(oof[names[i]], 0)) - np.abs(y-np.nanmean(oof[names[j]], 0))
            rows.append((names[i], names[j], stats.wilcoxon(d).pvalue, np.median(d)))
    out = pd.DataFrame(rows, columns=["A", "B", "p_raw", "median_diff_absErr(A-B)"])
    out["p_holm"] = multipletests(out.p_raw, method="holm")[1]
    return out

def calibration(z):
    cov = {lv: float(np.mean(np.abs(z) <= stats.norm.ppf(0.5+lv/2))) for lv in LEVELS}
    k95 = float(np.quantile(np.abs(z), 0.95)/stats.norm.ppf(0.975))
    return cov, k95

def hypothesis_table(summary, ph):
    r = []
    for m in ["GPR", "SVR"]:
        if m in summary.index and "ANN" in summary.index:
            for met in ["RMSE", "MAE"]:
                a, b = summary.loc[m, f"{met}_mean"], summary.loc["ANN", f"{met}_mean"]
                r.append(dict(hypothesis="H1", comparison=f"{m} vs ANN", metric=met,
                              pct_lower_than_ANN=round(100*(b-a)/b, 1), threshold_10pct_met=(b-a)/b >= 0.10))
    if "GPR" in summary.index:
        for met in ["RMSE", "MAE"]:
            r.append(dict(hypothesis="H2", comparison="GPR lowest of all", metric=met, pct_lower_than_ANN=np.nan,
                          threshold_10pct_met=summary[f"{met}_mean"].idxmin() == "GPR"))
        q = ph[(ph.A == "GPR") | (ph.B == "GPR")]
        r.append(dict(hypothesis="H2", comparison="GPR vs each rival, Holm p<0.05 (n sig / n rivals)",
                      metric="Wilcoxon", pct_lower_than_ANN=np.nan,
                      threshold_10pct_met=f"{int((q.p_holm < .05).sum())}/{len(q)}"))
    return pd.DataFrame(r)

def md_table(df, floatfmt=4):
    d = df.reset_index() if df.index.name or not isinstance(df.index, pd.RangeIndex) else df
    f = lambda v: f"{v:.{floatfmt}g}" if isinstance(v, (float, np.floating)) else str(v)
    lines = ["| " + " | ".join(map(str, d.columns)) + " |", "|" + "---|"*len(d.columns)]
    lines += ["| " + " | ".join(f(v) for v in row) + " |" for row in d.itertuples(index=False)]
    return "\n".join(lines)

def save_table(df, out, name):
    df.to_csv(f"{out}/{name}.csv")
    open(f"{out}/{name}.md", "w").write(md_table(df))

# ---------------- optional analyses -----------------
def pv_ablation(df, y, n_repeats):
    res = {}
    for tag, pv in [("P,v,F", False), ("P,v,F,P/v", True)]:
        f, o, *_ = nested_cv(make_X(df, pv), y, {"GPR": spaces(True)["GPR"]}, n_repeats)
        res[tag] = (f, o["GPR"])
    d = np.abs(y-np.nanmean(res["P,v,F"][1], 0)) - np.abs(y-np.nanmean(res["P,v,F,P/v"][1], 0))
    out = pd.DataFrame({k: v[0][["MAE", "RMSE", "R2"]].mean() for k, v in res.items()}).T
    out["Wilcoxon_p(vs no P/v)"] = [np.nan, stats.wilcoxon(d).pvalue]
    return out.round(4)

def learning_curve(X, y, fracs=(0.4, 0.55, 0.7, 0.85, 1.0), n_repeats=2):
    rng = np.random.default_rng(SEED); rows = []
    est, grid = spaces(True)["GPR"]
    for r in range(n_repeats):
        for k, (tr, te) in enumerate(KFold(5, shuffle=True, random_state=SEED+r).split(X)):
            for fr in fracs:
                sub = rng.choice(tr, int(round(fr*len(tr))), replace=False)
                ttr, g = wrap(est, grid)
                gs = GridSearchCV(ttr, g, cv=INNER_CV, scoring=PRIMARY, n_jobs=-1).fit(X[sub], y[sub])
                p = gs.predict(X[te]); rows.append(dict(n_train=len(sub), MAE=MAE(y[te], p), R2=r2_score(y[te], p)))
    return pd.DataFrame(rows).groupby("n_train").agg(["mean", "std"])

# ---------------- plots -----------------
COL = {"GPR": "#2ecc71", "ANN": "#3498db", "SVR": "#9b59b6", "RF": "#e67e22", "Ridge": "#7f8c8d"}
def savefig(fig, path): fig.tight_layout(); fig.savefig(path, dpi=300, bbox_inches="tight"); plt.close(fig)

def make_figures(t, label, folds, oof, y, ph, z, cov, rs, out, lc=None):
    order = folds.groupby("model").MAE.mean().sort_values().index.tolist()
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    for a, met in zip(ax, ["MAE", "RMSE", "R2"]):
        a.boxplot([folds[folds.model == m][met] for m in order], labels=order, patch_artist=True,
                  boxprops=dict(facecolor="#dfe6e9"), medianprops=dict(color="k"))
        a.set_title(f"{label}: {met} across outer folds"); a.grid(alpha=.3, axis="y")
    savefig(fig, f"{out}/{t}_fig1_fold_metrics.png")

    best = order[0]; models = [m for m in ["GPR", best] if m in oof]; models = list(dict.fromkeys(models))
    fig, ax = plt.subplots(1, len(models), figsize=(5.2*len(models), 4.8), squeeze=False)
    for a, m in zip(ax[0], models):
        p = np.nanmean(oof[m], 0); a.scatter(y, p, s=22, c=COL.get(m, "k"), edgecolor="k", lw=.3)
        lo, hi = min(y.min(), p.min()), max(y.max(), p.max()); a.plot([lo, hi], [lo, hi], "k--", lw=1)
        a.set_xlabel("Measured"); a.set_ylabel("Predicted (mean out-of-fold)")
        a.set_title(f"{m}: R²={r2_score(y, p):.3f}, MAE={MAE(y, p):.3g}")
    savefig(fig, f"{out}/{t}_fig2_pred_vs_actual.png")

    fig, ax = plt.subplots(1, 2, figsize=(10, 4)); p = np.nanmean(oof[best], 0); res = y-p
    ax[0].scatter(p, res, s=22, c=COL.get(best, "k"), edgecolor="k", lw=.3); ax[0].axhline(0, c="k", ls="--")
    ax[0].set_xlabel("Predicted"); ax[0].set_ylabel("Residual"); ax[0].set_title(f"{best}: residuals vs predicted")
    stats.probplot(res, dist="norm", plot=ax[1]); ax[1].set_title("Residual Q-Q plot (descriptive only)")
    savefig(fig, f"{out}/{t}_fig3_residuals.png")

    names = list(oof); M = pd.DataFrame(np.nan, index=names, columns=names)
    for _, r in ph.iterrows(): M.loc[r.A, r.B] = M.loc[r.B, r.A] = r.p_holm
    fig, ax = plt.subplots(figsize=(5.5, 4.6)); im = ax.imshow(M.values.astype(float), cmap="RdYlGn_r", vmin=0, vmax=0.1)
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names); ax.set_yticks(range(len(names))); ax.set_yticklabels(names)
    for i in range(len(names)):
        for j in range(len(names)):
            if i != j: ax.text(j, i, f"{M.values[i, j]:.3f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, label="Holm-adjusted p (colour capped at 0.1)"); ax.set_title(f"{label}: paired Wilcoxon")
    savefig(fig, f"{out}/{t}_fig4_pvalue_heatmap.png")

    if len(z):
        fig, ax = plt.subplots(figsize=(4.8, 4.6)); obs = [cov[l] for l in LEVELS]
        ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration"); ax.plot(LEVELS, obs, "o-", c=COL["GPR"], label="GPR (out-of-fold)")
        ax.set_xlabel("Nominal central-interval coverage"); ax.set_ylabel("Observed coverage"); ax.legend(); ax.grid(alpha=.3)
        ax.set_title(f"{label}: GPR interval calibration"); savefig(fig, f"{out}/{t}_fig5_gpr_calibration.png")

    fig, ax = plt.subplots(figsize=(6, 3.8)); rs.loc[order].plot(kind="bar", stacked=True, ax=ax, colormap="viridis_r")
    ax.set_ylabel("Share of repeats"); ax.set_title(f"{label}: ranking stability (by pooled MAE)"); ax.legend(fontsize=7, ncol=2)
    plt.setp(ax.get_xticklabels(), rotation=0); savefig(fig, f"{out}/{t}_fig6_rank_stability.png")

    if lc is not None:
        fig, ax = plt.subplots(figsize=(5.2, 4)); ns = lc.index.values
        ax.errorbar(ns, lc[("MAE", "mean")], yerr=lc[("MAE", "std")], marker="o", capsize=3, c=COL["GPR"])
        ax.set_xlabel("Training-set size"); ax.set_ylabel("Held-out MAE"); ax.grid(alpha=.3)
        ax.set_title(f"{label}: GPR learning curve"); savefig(fig, f"{out}/{t}_fig7_learning_curve.png")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="processed_data.csv"); ap.add_argument("--out", default="report_outputs")
    ap.add_argument("--repeats", type=int, default=5); ap.add_argument("--fast", action="store_true")
    ap.add_argument("--ablation", action="store_true"); ap.add_argument("--learning", action="store_true")
    ap.add_argument("--targets", nargs="*", default=list(TARGETS))
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    df = pd.read_csv(a.data); X = make_X(df); models = spaces(a.fast)
    master = []
    for t in a.targets:
        y = df[t].to_numpy(float); label = TARGETS.get(t, t); print(f"\n=== {label} ===")
        folds, oof, z, bp = nested_cv(X, y, models, a.repeats)
        S = summarise(folds, oof, y); ph = paired_wilcoxon_holm(oof, y); rs = rank_stability(oof, y)
        cov, k95 = calibration(z); H = hypothesis_table(S, ph)
        save_table(S, a.out, f"{t}_T1_performance"); save_table(ph.round(4), a.out, f"{t}_T2_pairwise_holm")
        save_table(rs.round(2), a.out, f"{t}_T3_rank_stability"); save_table(H, a.out, f"{t}_T4_hypotheses")
        save_table(pd.DataFrame({"nominal": LEVELS, "observed": [cov[l] for l in LEVELS]}).round(3).set_index("nominal"),
                   a.out, f"{t}_T5_gpr_coverage")
        pd.DataFrame(bp).to_csv(f"{a.out}/{t}_best_hyperparameters_per_fold.csv", index=False)
        pd.DataFrame({m: np.nanmean(oof[m], 0) for m in oof}).assign(measured=y).to_csv(f"{a.out}/{t}_oof_predictions.csv", index=False)
        if a.ablation: save_table(pv_ablation(df, y, max(2, a.repeats//2)), a.out, f"{t}_T6_PV_ablation")
        lc = None
        if a.learning:
            lc = learning_curve(X, y); lc.to_csv(f"{a.out}/{t}_learning_curve.csv")
        make_figures(t, label, folds, oof, y, ph, z, cov, rs, a.out, lc)
        print(S); print(ph.round(4)); print("GPR coverage:", {k: round(v, 3) for k, v in cov.items()}, "| sd inflation for 95%:", round(k95, 2))
        S2 = S.copy(); S2.insert(0, "target", label); master.append(S2.reset_index())
    M = pd.concat(master); M.to_csv(f"{a.out}/ALL_targets_performance.csv", index=False)
    if len(a.targets) > 1:
        fig, ax = plt.subplots(1, len(a.targets), figsize=(5*len(a.targets), 4), squeeze=False)
        for a_, (tn, g) in zip(ax[0], M.groupby("target", sort=False)):
            g = g.sort_values("MAE_mean"); a_.bar(g["model"], g["MAE_mean"], yerr=g["MAE_std"], capsize=3,
                                                  color=[COL.get(m, "gray") for m in g["model"]], edgecolor="k")
            a_.set_title(tn); a_.set_ylabel("MAE (mean ± SD over folds)"); a_.grid(alpha=.3, axis="y")
        savefig(fig, f"{a.out}/SUMMARY_fig_mae_all_targets.png")
    print(f"\nAll tables/figures written to ./{a.out}/")

if __name__ == "__main__":
    main()