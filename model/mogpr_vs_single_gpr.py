
"""
Proper multi-output GPR (Linear Model of Coregionalisation) vs single-output GPR for LMD targets.
Dependencies: numpy, scipy, pandas, matplotlib, scikit-learn, statsmodels (no GPy/GPyTorch needed).

Model:   cov(f_t(x), f_s(x')) = sum_q B_q[t,s] * k_q(x,x'),   B_q = a_q a_q^T + diag(kappa_q)
         k_q = ARD-RBF.   Q=1 -> ICM (shared input kernel);  Q>=2 -> LMC (separate length-scales per latent).
         Each task also has its own noise variance.  All hyper-parameters are learned by maximising the
         marginal likelihood on the TRAINING fold only (analytic gradients, multi-start L-BFGS-B).
Variants compared, on identical repeated 5-fold outer splits, with in-fold standardisation of X and Y:
   SO-GPR     one independent GP per target (own length-scales)           <- single-output reference
   ICM-indep  shared kernel, B diagonal (NO cross-target correlation)     <- control: isolates kernel sharing
   ICM        shared kernel, learned cross-target correlation
   LMC-2      two latent kernels, learned cross-target correlation
   skGPR      scikit-learn ARD GPR (sanity check of the SO implementation)
Because the three LMD responses are algebraically linked (aspect ratio = height/width), also run a
target set without that pair, e.g. --sets height_um,aspect_ratio,dilution_percent height_um,dilution_percent
"""
import argparse, os, warnings
os.environ["PYTHONWARNINGS"] = "ignore"
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from scipy.linalg import cholesky, cho_solve, solve_triangular
from scipy.optimize import minimize
from statsmodels.stats.multitest import multipletests
from sklearn.model_selection import RepeatedKFold
from sklearn.gaussian_process import GaussianProcessRegressor as SKGPR
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from sklearn.metrics import mean_absolute_error as MAE, mean_squared_error, r2_score
warnings.filterwarnings("ignore")
SEED = 42
LEVELS = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
FEATURES = ["Power", "Speed", "Feed"]
LABEL = {"height_um": "Height (µm)", "aspect_ratio": "Aspect ratio (-)", "dilution_percent": "Dilution (%)",
         "width_um": "Width (µm)"}

class LMCGP:
    """Multi-output GP (LMC/ICM). With T=1, Q=1, coupled=False it is an ordinary ARD-RBF GP."""
    def __init__(self, n_latent=1, coupled=True, n_restarts=5, seed=SEED):
        self.Q, self.coupled, self.n_restarts, self.seed = n_latent, coupled, n_restarts, seed

    def _unpack(self, th, d, T):
        i, lat = 0, []
        for _ in range(self.Q):
            ls = np.exp(th[i:i+d]); i += d
            if self.coupled: a = th[i:i+T].copy(); i += T
            else: a = np.zeros(T)
            kap = np.exp(th[i:i+T]); i += T
            lat.append((ls, a, kap))
        return lat, np.exp(th[i:i+T])

    def _bounds(self, d, T):
        b = []
        for _ in range(self.Q):
            b += [(-3, 3)]*d + ([(-4, 4)]*T if self.coupled else []) + [(-9, 3)]*T
        return b + [(-9, 1)]*T

    def _nll(self, th, X, Y, diffsq):
        T, N = Y.shape; d = X.shape[1]
        lat, noise = self._unpack(th, d, T)
        K = np.kron(np.diag(noise), np.eye(N)) + 1e-8*np.eye(T*N); Kxs, Bs = [], []
        for ls, a, kap in lat:
            Kx = np.exp(-0.5*(diffsq/ls**2).sum(-1)); B = np.outer(a, a) + np.diag(kap)
            K += np.kron(B, Kx); Kxs.append(Kx); Bs.append(B)
        try: L = cholesky(K, lower=True)
        except Exception: return 1e10, np.zeros_like(th)
        y = Y.ravel(); alpha = cho_solve((L, True), y)
        nll = 0.5*y@alpha + np.log(np.diag(L)).sum() + 0.5*T*N*np.log(2*np.pi)
        M4 = (cho_solve((L, True), np.eye(T*N)) - np.outer(alpha, alpha)).reshape(T, N, T, N)
        g = []
        for (ls, a, kap), Kx, B in zip(lat, Kxs, Bs):
            for j in range(d):
                H = np.einsum("tisj,ij->ts", M4, Kx*diffsq[:, :, j]/ls[j]**2)
                g.append(0.5*np.sum(B*H))
            G = np.einsum("tisj,ij->ts", M4, Kx)
            if self.coupled: g += list(G@a)
            g += list(0.5*kap*np.diag(G))
        g += [0.5*noise[t]*np.trace(M4[t, :, t, :]) for t in range(T)]
        return nll, np.array(g)

    def _init(self, rng, d, T):
        th = []
        for q in range(self.Q):
            th += list(np.log(rng.uniform(0.5, 3.0, d)*(1+2*q)))
            if self.coupled: th += list(rng.normal(0, 0.7, T))
            th += list(np.log(0.1)+rng.normal(0, .3, T))
        return np.array(th + list(np.log(0.05)+rng.normal(0, .3, T)))

    def fit(self, X, Y):
        X, Y = np.asarray(X, float), np.asarray(Y, float)
        self.xm, self.xs = X.mean(0), X.std(0); self.ym, self.ys = Y.mean(0), Y.std(0)
        Xs, Ys = (X-self.xm)/self.xs, ((Y-self.ym)/self.ys).T
        self.d, self.T = Xs.shape[1], Ys.shape[0]
        diffsq = (Xs[:, None, :]-Xs[None, :, :])**2; rng = np.random.default_rng(self.seed); best = None
        for _ in range(self.n_restarts):
            r = minimize(self._nll, self._init(rng, self.d, self.T), args=(Xs, Ys, diffsq), jac=True,
                         method="L-BFGS-B", bounds=self._bounds(self.d, self.T), options=dict(maxiter=200))
            if best is None or r.fun < best.fun: best = r
        self.theta_, self.nll_ = best.x, best.fun
        self.lat_, self.noise_ = self._unpack(self.theta_, self.d, self.T)
        self.Xs_, self.Ys_ = Xs, Ys
        N = Xs.shape[0]; K = np.kron(np.diag(self.noise_), np.eye(N)) + 1e-8*np.eye(self.T*N)
        for ls, a, kap in self.lat_:
            K += np.kron(np.outer(a, a)+np.diag(kap), np.exp(-0.5*(diffsq/ls**2).sum(-1)))
        self.L_ = cholesky(K, lower=True); self.alpha_ = cho_solve((self.L_, True), Ys.ravel())
        return self

    def predict(self, Xnew):
        Xs = (np.asarray(Xnew, float)-self.xm)/self.xs; Ns, N, T = len(Xs), len(self.Xs_), self.T
        d2 = (Xs[:, None, :]-self.Xs_[None, :, :])**2; Ks = np.zeros((T*Ns, T*N)); prior = np.zeros(T)
        for ls, a, kap in self.lat_:
            B = np.outer(a, a)+np.diag(kap); Ks += np.kron(B, np.exp(-0.5*(d2/ls**2).sum(-1))); prior += np.diag(B)
        mu = (Ks@self.alpha_).reshape(T, Ns)
        V = solve_triangular(self.L_, Ks.T, lower=True)
        var = (prior[:, None] + self.noise_[:, None] - (V**2).sum(0).reshape(T, Ns)).clip(1e-12)  # predictive incl. noise
        return (mu.T*self.ys+self.ym), (np.sqrt(var).T*self.ys)

    def task_corr(self):
        B = sum(np.outer(a, a)+np.diag(k) for _, a, k in self.lat_); s = np.sqrt(np.diag(B))
        return B/np.outer(s, s)

def sk_fit_predict(Xtr, ytr, Xte):
    xm, xs, ym, ys = Xtr.mean(0), Xtr.std(0), ytr.mean(), ytr.std()
    k = C(1.0)*RBF(np.ones(Xtr.shape[1]), (1e-2, 1e2)) + WhiteKernel(0.05, (1e-6, 1))
    g = SKGPR(k, n_restarts_optimizer=3, random_state=SEED).fit((Xtr-xm)/xs, (ytr-ym)/ys)
    mu, sd = g.predict((Xte-xm)/xs, return_std=True); return mu*ys+ym, sd*ys

def run_variant(X, Y, splits, R, cfg):
    N, T = Y.shape; oof = np.full((R, N, T), np.nan); z = []; corr = []; rows = []
    for k, (tr, te) in enumerate(splits):
        if cfg == "skGPR":
            out = [sk_fit_predict(X[tr], Y[tr, t], X[te]) for t in range(T)]
            mu = np.column_stack([o[0] for o in out]); sd = np.column_stack([o[1] for o in out])
        elif cfg == "SO-GPR":
            out = [LMCGP(1, False).fit(X[tr], Y[tr][:, [t]]).predict(X[te]) for t in range(T)]
            mu = np.column_stack([o[0][:, 0] for o in out]); sd = np.column_stack([o[1][:, 0] for o in out])
        else:
            m = LMCGP(cfg["Q"], cfg["coupled"]).fit(X[tr], Y[tr]); mu, sd = m.predict(X[te])
            if cfg["coupled"]: corr.append(m.task_corr())
        oof[k//5, te] = mu; z.append((Y[te]-mu)/sd)
        for t in range(T):
            rows.append(dict(repeat=k//5, fold=k % 5, target_idx=t, MAE=MAE(Y[te, t], mu[:, t]),
                             RMSE=mean_squared_error(Y[te, t], mu[:, t])**0.5, R2=r2_score(Y[te, t], mu[:, t])))
    return oof, np.vstack(z), (np.mean(corr, 0) if corr else None), pd.DataFrame(rows)

VARIANTS = {"ICM-indep": dict(Q=1, coupled=False), "ICM": dict(Q=1, coupled=True), "LMC-2": dict(Q=2, coupled=True)}

def run_set(df, targets, R, cache):
    X = df[FEATURES].to_numpy(float); Y = df[targets].to_numpy(float); N = len(df)
    splits = list(RepeatedKFold(n_splits=5, n_repeats=R, random_state=SEED).split(X))
    res = {}
    for name, cfg in [("SO-GPR", "SO-GPR"), ("skGPR", "skGPR")] + list(VARIANTS.items()):
        key = (name, tuple(targets)) if name in VARIANTS else (name, R, tuple(targets))
        # SO / skGPR do not depend on the other targets, so cache them per single target
        if name in ("SO-GPR", "skGPR"):
            parts = []
            for j, t in enumerate(targets):
                ck = (name, t, R)
                if ck not in cache: cache[ck] = run_variant(X, df[[t]].to_numpy(float), splits, R, cfg)
                parts.append(cache[ck])
            oof = np.concatenate([p[0] for p in parts], 2); z = np.hstack([p[1] for p in parts]); corr = None
            fr = pd.concat([p[3].assign(target_idx=j) for j, p in enumerate(parts)])
        else:
            oof, z, corr, fr = run_variant(X, Y, splits, R, cfg)
        print(f"   done: {name}", flush=True); res[name] = dict(oof=oof, z=z, corr=corr, folds=fr)
    return res, Y

def tables(res, Y, targets):
    T1, T3 = [], []
    for name, r in res.items():
        for j, t in enumerate(targets):
            f = r["folds"][r["folds"].target_idx == j]
            T1.append(dict(model=name, target=LABEL.get(t, t), MAE_mean=f.MAE.mean(), MAE_sd=f.MAE.std(),
                           RMSE_mean=f.RMSE.mean(), R2_mean=f.R2.mean(), R2_sd=f.R2.std(),
                           MAE_pct_mean_y=100*f.MAE.mean()/Y[:, j].mean(),
                           R2_pooled=np.mean([r2_score(Y[:, j], r["oof"][k, :, j]) for k in range(r["oof"].shape[0])])))
            T3.append(dict(model=name, target=LABEL.get(t, t),
                           **{f"cov{int(l*100)}": np.mean(np.abs(r["z"][:, j]) <= stats.norm.ppf(.5+l/2)) for l in (0.5, 0.8, 0.9, 0.95)}))
    rows = []
    for j, t in enumerate(targets):
        ae = {m: np.abs(Y[:, j]-np.nanmean(r["oof"][:, :, j], 0)) for m, r in res.items()}
        for a, b in [("ICM", "SO-GPR"), ("LMC-2", "SO-GPR"), ("ICM-indep", "SO-GPR"), ("ICM", "ICM-indep"), ("SO-GPR", "skGPR")]:
            d = ae[a]-ae[b]; mm = res
            rA = res[a]["folds"].query("target_idx==@j").groupby("repeat").MAE.mean()
            rB = res[b]["folds"].query("target_idx==@j").groupby("repeat").MAE.mean()
            rows.append(dict(target=LABEL.get(t, t), A=a, B=b, MAE_A=np.mean(ae[a]), MAE_B=np.mean(ae[b]),
                             pct_change_MAE=100*(np.mean(ae[a])-np.mean(ae[b]))/np.mean(ae[b]),
                             repeats_A_better=f"{int((rA<rB).sum())}/{len(rA)}", p_raw=stats.wilcoxon(d).pvalue))
    T2 = pd.DataFrame(rows); T2["p_holm"] = multipletests(T2.p_raw, method="holm")[1]
    return pd.DataFrame(T1), T2, pd.DataFrame(T3)

def figures(tag, T1, T2, res, targets, out):
    models = ["SO-GPR", "ICM-indep", "ICM", "LMC-2"]; cols = ["#7f8c8d", "#3498db", "#2ecc71", "#e67e22"]
    labs = [LABEL.get(t, t) for t in targets]
    fig, ax = plt.subplots(1, len(targets), figsize=(4.6*len(targets), 4), squeeze=False)
    for a, lab in zip(ax[0], labs):
        s = T1[T1.target == lab].set_index("model").loc[models]
        a.bar(models, s.MAE_mean, yerr=s.MAE_sd, color=cols, edgecolor="k", capsize=3)
        a.set_title(lab); a.set_ylabel("MAE (mean ± SD over folds)"); a.grid(alpha=.3, axis="y"); plt.setp(a.get_xticklabels(), rotation=20)
    savefig(fig, f"{out}/{tag}_fig1_mae_by_model.png")
    fig, ax = plt.subplots(figsize=(5.5, 3.8)); w = 0.25; x = np.arange(len(targets))
    for i, (b, c) in enumerate(zip(["ICM", "LMC-2", "ICM-indep"], cols[2:] + [cols[1]])):
        v = [T2[(T2.target == lab) & (T2.A == b) & (T2.B == "SO-GPR")].pct_change_MAE.iloc[0] for lab in labs]
        ax.bar(x+(i-1)*w, v, w, label=b, color=c, edgecolor="k")
    ax.axhline(0, c="k", lw=.8); ax.set_xticks(x); ax.set_xticklabels(labs); ax.set_ylabel("Change in MAE vs SO-GPR (%)  [<0 = better]")
    ax.legend(); ax.grid(alpha=.3, axis="y"); savefig(fig, f"{out}/{tag}_fig2_delta_mae_vs_single.png")
    for nm in ["ICM", "LMC-2"]:
        Cm = res[nm]["corr"]
        fig, ax = plt.subplots(figsize=(4.4, 3.8)); im = ax.imshow(Cm, vmin=-1, vmax=1, cmap="RdBu_r")
        ax.set_xticks(range(len(labs))); ax.set_xticklabels(labs, rotation=25); ax.set_yticks(range(len(labs))); ax.set_yticklabels(labs)
        for i in range(len(labs)):
            for j in range(len(labs)): ax.text(j, i, f"{Cm[i, j]:.2f}", ha="center", va="center")
        fig.colorbar(im); ax.set_title(f"{nm}: learned task correlation\n(mean over outer folds)"); savefig(fig, f"{out}/{tag}_fig3_corr_{nm}.png")
    fig, ax = plt.subplots(1, len(targets), figsize=(4.4*len(targets), 4), squeeze=False)
    for j, a in enumerate(ax[0]):
        a.plot([0, 1], [0, 1], "k--")
        for m, c in zip(models, cols):
            a.plot(LEVELS, [np.mean(np.abs(res[m]["z"][:, j]) <= stats.norm.ppf(.5+l/2)) for l in LEVELS], "o-", c=c, label=m)
        a.set_title(labs[j]); a.set_xlabel("Nominal coverage"); a.set_ylabel("Observed coverage"); a.grid(alpha=.3)
    ax[0][0].legend(fontsize=7); savefig(fig, f"{out}/{tag}_fig4_calibration.png")

def savefig(fig, p): fig.tight_layout(); fig.savefig(p, dpi=300, bbox_inches="tight"); plt.close(fig)

def derived_aspect(df, res, targets):
    h, w = targets.index("height_um"), targets.index("width_um"); asp = df["aspect_ratio"].to_numpy(float)
    chk = np.corrcoef(asp, df.height_um/df.width_um)[0, 1]; rows = []
    for m, r in res.items():
        p = np.nanmean(r["oof"][:, :, h]/r["oof"][:, :, w], 0); rows.append(dict(model=m, MAE_aspect_derived=MAE(asp, p), R2=r2_score(asp, p)))
    return pd.DataFrame(rows), chk

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="processed_data.csv"); ap.add_argument("--out", default="mogpr_outputs")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--sets", nargs="*", default=["height_um,aspect_ratio,dilution_percent", "height_um,dilution_percent"])
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); df = pd.read_csv(a.data); cache = {}
    for s in a.sets:
        targets = s.split(","); tag = "set_" + "-".join(t.split("_")[0] for t in targets); print(f"\n=== {targets} ===")
        res, Y = run_set(df, targets, a.repeats, cache); T1, T2, T3 = tables(res, Y, targets)
        for n, t in [("T1_performance", T1), ("T2_paired_tests_holm", T2), ("T3_interval_coverage", T3)]:
            t.round(4).to_csv(f"{a.out}/{tag}_{n}.csv", index=False)
        for nm in ("ICM", "LMC-2"): pd.DataFrame(res[nm]["corr"], index=targets, columns=targets).round(3).to_csv(f"{a.out}/{tag}_T4_corr_{nm}.csv")
        if "height_um" in targets and "width_um" in targets and "aspect_ratio" in df:
            d, chk = derived_aspect(df, res, targets); d.round(4).to_csv(f"{a.out}/{tag}_T5_derived_aspect.csv", index=False)
            print(f"corr(aspect, height/width) = {chk:.4f}\n", d.round(4))
        figures(tag, T1, T2, res, targets, a.out)
        print(T1.round(4).to_string(index=False)); print(T2.round(4).to_string(index=False))
        for nm in ("ICM", "LMC-2"): print(f"{nm} learned correlation:\n", pd.DataFrame(res[nm]["corr"], index=targets, columns=targets).round(2))
    print(f"\nOutputs in ./{a.out}/")

if __name__ == "__main__":
    main()
