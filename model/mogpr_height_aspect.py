"""
Height + aspect-ratio only: multi-output GPR (ICM / LMC) vs single-output GPR.
Run from the same folder as mogpr_vs_single_gpr.py (it re-uses that file's LMCGP class and CV helpers).

  python mogpr_height_aspect.py --data processed_data.csv --repeats 5
  python mogpr_height_aspect.py --data processed_data.csv --repeats 5 --log     # model log(height), log(aspect)

--log: aspect = height/width means log(aspect) = log(height) - log(width). In log space the algebraic link becomes
       LINEAR, which is the structure a coregionalised GP can actually exploit. Metrics are still reported in original
       units (predictions are exp()-back-transformed, i.e. median predictions); interval coverage is in log space.
Adds, beyond mogpr_vs_single_gpr.py: more repeats by default, bootstrap 95% CI of the paired change in MAE, and a
plain-language verdict per target.
"""
import argparse, os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.stats.multitest import multipletests
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import mean_absolute_error as MAE, mean_squared_error, r2_score
import mogpr_vs_single_gpr as m

TARGETS = ["height_um", "aspect_ratio"]
MODELS = ["SO-GPR", "ICM-indep", "ICM", "LMC-2"]
COMPARE = [("ICM", "SO-GPR"), ("LMC-2", "SO-GPR"), ("ICM-indep", "SO-GPR"), ("ICM", "ICM-indep")]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="processed_data.csv"); ap.add_argument("--out", default="mogpr_height_aspect_outputs")
    ap.add_argument("--repeats", type=int, default=5); ap.add_argument("--log", action="store_true")
    ap.add_argument("--boot", type=int, default=5000)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    df = pd.read_csv(a.data); X = df[m.FEATURES].to_numpy(float); Y = df[TARGETS].to_numpy(float); N = len(df)
    Yfit = np.log(Y) if a.log else Y; R = a.repeats; tag = "log" if a.log else "raw"
    splits = list(RepeatedKFold(n_splits=5, n_repeats=R, random_state=m.SEED).split(X)); res = {}
    for name in MODELS:
        if name == "SO-GPR":
            parts = [m.run_variant(X, Yfit[:, [j]], splits, R, "SO-GPR") for j in range(2)]
            oof = np.concatenate([p[0] for p in parts], 2); z = np.hstack([p[1] for p in parts]); corr = None
        else:
            oof, z, corr, _ = m.run_variant(X, Yfit, splits, R, m.VARIANTS[name])
        res[name] = dict(oof=np.exp(oof) if a.log else oof, z=z, corr=corr); print("done", name, flush=True)

    T1 = []
    for name, r in res.items():
        for j, t in enumerate(TARGETS):
            fm = [(MAE(Y[te, j], r["oof"][k//5, te, j]), mean_squared_error(Y[te, j], r["oof"][k//5, te, j])**0.5,
                   r2_score(Y[te, j], r["oof"][k//5, te, j])) for k, (tr, te) in enumerate(splits)]
            fm = np.array(fm)
            T1.append(dict(model=name, target=m.LABEL[t], MAE_mean=fm[:, 0].mean(), MAE_sd=fm[:, 0].std(ddof=1),
                           RMSE_mean=fm[:, 1].mean(), R2_mean=fm[:, 2].mean(), R2_sd=fm[:, 2].std(ddof=1),
                           MAE_pct_mean_y=100*fm[:, 0].mean()/Y[:, j].mean(),
                           R2_pooled=np.mean([r2_score(Y[:, j], r["oof"][k, :, j]) for k in range(R)]),
                           cov90=np.mean(np.abs(r["z"][:, j]) <= stats.norm.ppf(.95)),
                           cov95=np.mean(np.abs(r["z"][:, j]) <= stats.norm.ppf(.975))))
    T1 = pd.DataFrame(T1)

    rng = np.random.default_rng(m.SEED); rows = []
    for j, t in enumerate(TARGETS):
        ae = {k: np.abs(Y[:, j]-np.nanmean(r["oof"][:, :, j], 0)) for k, r in res.items()}
        for A, B in COMPARE:
            d = ae[A]-ae[B]; idx = rng.integers(0, N, (a.boot, N))
            pct = 100*(ae[A][idx].mean(1)-ae[B][idx].mean(1))/ae[B][idx].mean(1)
            rep = lambda k: pd.Series([MAE(Y[:, j], res[k]["oof"][r, :, j]) for r in range(R)])
            rows.append(dict(target=m.LABEL[t], A=A, B=B, MAE_A=ae[A].mean(), MAE_B=ae[B].mean(),
                             pct_change_MAE=100*(ae[A].mean()-ae[B].mean())/ae[B].mean(),
                             CI95_lo=np.percentile(pct, 2.5), CI95_hi=np.percentile(pct, 97.5),
                             repeats_A_better=f"{int((rep(A) < rep(B)).sum())}/{R}", p_raw=stats.wilcoxon(d).pvalue))
    T2 = pd.DataFrame(rows); T2["p_holm"] = multipletests(T2.p_raw, method="holm")[1]
    T1.round(4).to_csv(f"{a.out}/{tag}_T1_performance.csv", index=False); T2.round(4).to_csv(f"{a.out}/{tag}_T2_paired_tests.csv", index=False)
    pd.DataFrame(res["ICM"]["corr"], index=TARGETS, columns=TARGETS).round(3).to_csv(f"{a.out}/{tag}_T3_corr_ICM.csv")

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for j, (a_, t) in enumerate(zip(ax, TARGETS)):
        s = T2[(T2.target == m.LABEL[t]) & (T2.B == "SO-GPR")]
        y = s.pct_change_MAE.values; err = np.vstack([y-s.CI95_lo.values, s.CI95_hi.values-y])
        a_.bar(s.A, y, yerr=err, capsize=4, color=["#2ecc71", "#e67e22", "#3498db"], edgecolor="k")
        a_.axhline(0, c="k", lw=.8); a_.set_title(f"{m.LABEL[t]} ({tag} space)")
        a_.set_ylabel("MAE change vs SO-GPR (%), bootstrap 95% CI  [<0 = better]"); a_.grid(alpha=.3, axis="y")
    fig.tight_layout(); fig.savefig(f"{a.out}/{tag}_fig1_delta_mae.png", dpi=300); plt.close(fig)

    print(T1.round(4).to_string(index=False)); print(T2.round(4).to_string(index=False))
    print("ICM learned correlation:\n", pd.DataFrame(res["ICM"]["corr"], index=TARGETS, columns=TARGETS).round(3))
    print("\nVERDICT (ICM vs SO-GPR):")
    for t in TARGETS:
        r = T2[(T2.target == m.LABEL[t]) & (T2.A == "ICM") & (T2.B == "SO-GPR")].iloc[0]
        ok = r.CI95_hi < 0 and r.p_holm < 0.05
        print(f" {m.LABEL[t]:18s} change {r.pct_change_MAE:+.1f}% [{r.CI95_lo:+.1f}, {r.CI95_hi:+.1f}], Holm p={r.p_holm:.3f} ->",
              "SUPPORTED improvement" if ok else ("CI excludes 0 but not Holm-significant" if r.CI95_hi < 0 else
              ("significantly WORSE than single-output" if (r.CI95_lo > 0 and r.p_holm < 0.05) else "no demonstrated improvement")))
    print(f"\nOutputs in ./{a.out}/")

if __name__ == "__main__":
    main()