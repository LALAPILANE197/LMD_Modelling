# ============================================================
# 10_nsga2_optimisation.py
# Phase 1B — NSGA-II Multi-Objective Inverse Optimisation
# Surrogate : 3-output Multi-Output GPR
#             (Height µm  |  Aspect Ratio  |  Dilution %)
# Targets   : LMD Al-Si bead geometry inverse optimisation
# ============================================================
#
# ALGORITHM OVERVIEW
# ─────────────────────────────────────────────────────────
# Why NSGA-II alone is sufficient here:
#   • Only 3 objectives and 3 decision variables — NSGA-II
#     crowding distance handles this well (Ishibuchi 2016)
#   • Pareto surface is smooth and physics-driven — all
#     three targets share the same laser thermodynamics
#   • Domain literature (laser cladding / LMD) consistently
#     uses NSGA-II for 3-objective bead geometry problems
#   • GPR surrogate evaluations are fast (µs each), so a
#     large population + many generations is affordable
#
# Strategy:
#   • Population initialised with Latin Hypercube Sampling
#     (better space-filling than pure random at no cost)
#   • Hard dilution constraints enforced via pymoo's
#     constraint-handling (g ≤ 0 convention)
#   • GPR posterior σ computed post-optimisation to flag
#     unreliable Pareto candidates (extrapolation risk)
#   • Publication-quality figures at 600 dpi
#
# References
#   Deb K et al. IEEE Trans Evol Comput 2002;6(2):182-197
#   Ishibuchi H et al. CEC 2016 — NSGA-II vs NSGA-III study
#   Xue J, Shen B. Systems Sci & Control Eng 2020;8(1):22-34
# ============================================================

import numpy as np
import pandas as pd
import joblib
import time
import warnings
from scipy.stats import qmc                       # LHS initialisation

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D           # 3-D Pareto surface

from pymoo.core.problem         import Problem
from pymoo.core.sampling        import Sampling
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm   import PM
from pymoo.termination             import get_termination
from pymoo.optimize                import minimize
from pymoo.decomposition.asf       import ASF
from pymoo.indicators.hv           import HV      # Pareto quality metric

warnings.filterwarnings('ignore')
np.random.seed(42)

# ─────────────────────────────────────────────────────────────
# PUBLICATION STYLE — applied globally to all figures
# ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family'       : 'DejaVu Serif',
    'font.size'         : 11,
    'axes.titlesize'    : 12,
    'axes.labelsize'    : 11,
    'axes.titleweight'  : 'bold',
    'axes.labelweight'  : 'bold',
    'xtick.labelsize'   : 9,
    'ytick.labelsize'   : 9,
    'legend.fontsize'   : 9,
    'figure.dpi'        : 150,
    'savefig.dpi'       : 600,      # journal-quality output
    'lines.linewidth'   : 1.8,
    'axes.grid'         : True,
    'grid.alpha'        : 0.3,
    'axes.spines.top'   : False,
    'axes.spines.right' : False,
})

# Consistent colour palette across all panels
C_BLUE   = '#2c7bb6'
C_RED    = '#d7191c'
C_GREEN  = '#1a9641'
C_ORANGE = '#fdae61'
C_PURPLE = '#762a83'

# ─────────────────────────────────────────────────────────────
# SECTION 1 — LOAD 3-OUTPUT GPR AND ALL SCALERS
# ─────────────────────────────────────────────────────────────
# Saved from notebook Cell 16 (Improvement D):
#   multi_gpr.estimators_[0] → Height     GPR
#   multi_gpr.estimators_[1] → Aspect Ratio GPR
#   multi_gpr.estimators_[2] → Dilution % GPR
# ─────────────────────────────────────────────────────────────

print("=" * 65)
print("STEP 1 — Loading 3-output GPR surrogate and scalers")
print("=" * 65)

multi_gpr = joblib.load("multi_output_gpr.pkl")
scaler_X  = joblib.load("scaler_X.pkl")
scaler_H  = joblib.load("scaler_height.pkl")
scaler_A  = joblib.load("scaler_aspect.pkl")
scaler_D  = joblib.load("scaler_dilution.pkl")

# Validate loaded objects — catches stale .pkl files early
assert hasattr(multi_gpr, 'estimators_'), \
    ("multi_output_gpr.pkl is not a fitted MultiOutputRegressor. "
     "Re-run notebook Cell 16 (Improvement D) and re-save.")
assert len(multi_gpr.estimators_) == 3, (
    f"Expected 3 inner estimators [Height, AR, Dilution], "
    f"got {len(multi_gpr.estimators_)}. Re-run Cell 16.")

print(f"  multi_output_gpr.pkl : {len(multi_gpr.estimators_)} "
      f"inner estimators ✓")
print(f"  scaler_X             : {scaler_X.n_features_in_} features")
print(f"  scaler_height        : µ={scaler_H.mean_[0]:.1f} µm  "
      f"σ={scaler_H.scale_[0]:.1f} µm")
print(f"  scaler_aspect        : µ={scaler_A.mean_[0]:.4f}  "
      f"σ={scaler_A.scale_[0]:.4f}")
print(f"  scaler_dilution      : µ={scaler_D.mean_[0]:.2f}%  "
      f"σ={scaler_D.scale_[0]:.2f}%")

# ─────────────────────────────────────────────────────────────
# SECTION 2 — PROBLEM SPECIFICATION
# Edit only this section to change optimisation targets.
# ─────────────────────────────────────────────────────────────

# ── Desired bead geometry (inverse problem targets) ──────────
H_TARGET  = 1200.0   # µm  — desired bead height
AR_TARGET =    0.46  #     — desired aspect ratio (Width / Height)
D_TARGET  =   35.0   # %   — desired dilution percentage

# ── Dilution hard constraints (metallurgical engineering bounds)
# D < 15% → insufficient substrate bonding (shallow penetration)
# D > 45% → excessive base-metal mixing, degrades alloy chemistry
D_MIN = 15.0          # %
D_MAX = 45.0          # %

# ── Normalisation: σ of each target from training data
#    Makes all three objectives dimensionless and equally scaled
SCALE_H  = float(scaler_H.scale_[0])   # ≈ 426 µm
SCALE_AR = float(scaler_A.scale_[0])   # ≈ 0.215
SCALE_D  = float(scaler_D.scale_[0])   # ≈ 14.2 %

# ── LMD machine physical bounds ─────────────────────────────
BOUNDS_LOWER = np.array([1600.0, 0.8, 2.0])  # [Power W, Speed mm/s, Feed g/min]
BOUNDS_UPPER = np.array([2800.0, 1.6, 8.0])
N_VARS = 3

# ── NSGA-II hyperparameters ──────────────────────────────────
POP_SIZE = 200    # Population size — larger than minimum (150) for
                  # better Pareto coverage at 3 objectives
N_GEN    = 300    # Generations — sufficient for convergence at N=89
                  # GPR evaluations are µs-fast so runtime is low

print(f"\n  Targets    : H={H_TARGET:.1f} µm | "
      f"AR={AR_TARGET:.3f} | D={D_TARGET:.1f}%")
print(f"  D bounds   : [{D_MIN:.0f}%, {D_MAX:.0f}%]  (hard constraint)")
print(f"  σ scales   : σ_H={SCALE_H:.1f} µm | "
      f"σ_AR={SCALE_AR:.4f} | σ_D={SCALE_D:.2f}%")
print(f"  NSGA-II    : pop={POP_SIZE}  gen={N_GEN}")
print(f"  Bounds     : P {BOUNDS_LOWER[0]:.0f}–{BOUNDS_UPPER[0]:.0f} W | "
      f"S {BOUNDS_LOWER[1]:.1f}–{BOUNDS_UPPER[1]:.1f} mm/s | "
      f"F {BOUNDS_LOWER[2]:.1f}–{BOUNDS_UPPER[2]:.1f} g/min")

# ─────────────────────────────────────────────────────────────
# SECTION 3 — GPR PREDICTION FUNCTIONS
# ─────────────────────────────────────────────────────────────

def _build_features(X_raw: np.ndarray) -> np.ndarray:
    """
    Reconstruct 4-column feature matrix from raw [P, S, F].
    Mirrors Cell 3 of the notebook exactly:
      columns → [Power, Speed, Feed, HeatInput = Power/(Speed+ε)]
    ε = 1e-9 prevents division-by-zero at Speed → 0.
    """
    X_raw = np.atleast_2d(X_raw)
    heat  = X_raw[:, 0] / (X_raw[:, 1] + 1e-9)
    return np.column_stack([X_raw, heat])


def predict_joint(X_raw: np.ndarray):
    """
    Point prediction for all three targets (no uncertainty).
    Called inside NSGA-II _evaluate() — must be fast.

    Parameters
    ----------
    X_raw : (n, 3)  [Power W, Speed mm/s, Feed g/min]

    Returns
    -------
    H_p  : (n,) µm
    AR_p : (n,) dimensionless
    D_p  : (n,) %
    """
    X_sc = scaler_X.transform(_build_features(np.atleast_2d(X_raw)))
    Y_sc = multi_gpr.predict(X_sc)          # (n, 3) scaled outputs
    H_p  = scaler_H.inverse_transform(Y_sc[:, 0].reshape(-1, 1)).ravel()
    AR_p = scaler_A.inverse_transform(Y_sc[:, 1].reshape(-1, 1)).ravel()
    D_p  = scaler_D.inverse_transform(Y_sc[:, 2].reshape(-1, 1)).ravel()
    return H_p, AR_p, D_p


def predict_with_uncertainty(X_raw: np.ndarray):
    """
    Point prediction + GPR posterior standard deviations.

    MultiOutputRegressor does not expose return_std at the
    wrapper level — inner estimators are accessed directly:
      estimators_[0] → Height     GPR
      estimators_[1] → Aspect Ratio GPR
      estimators_[2] → Dilution % GPR

    Back-transform: σ_orig = σ_scaled × scaler.scale_
    (valid for StandardScaler — affine transformation)

    Returns
    -------
    H_p, AR_p, D_p   : point predictions  (original units)
    σ_H, σ_AR, σ_D   : posterior std devs (original units)
    """
    X_raw = np.atleast_2d(X_raw)
    X_sc  = scaler_X.transform(_build_features(X_raw))

    H_sc,  sH  = multi_gpr.estimators_[0].predict(X_sc, return_std=True)
    AR_sc, sAR = multi_gpr.estimators_[1].predict(X_sc, return_std=True)
    D_sc,  sD  = multi_gpr.estimators_[2].predict(X_sc, return_std=True)

    H_p  = scaler_H.inverse_transform(H_sc.reshape(-1,  1)).ravel()
    AR_p = scaler_A.inverse_transform(AR_sc.reshape(-1, 1)).ravel()
    D_p  = scaler_D.inverse_transform(D_sc.reshape(-1,  1)).ravel()

    return (H_p, AR_p, D_p,
            sH  * scaler_H.scale_[0],    # σ_H  [µm]
            sAR * scaler_A.scale_[0],    # σ_AR [—]
            sD  * scaler_D.scale_[0])    # σ_D  [%]

# ─────────────────────────────────────────────────────────────
# SECTION 4 — LHS INITIALISATION SAMPLING
# ─────────────────────────────────────────────────────────────

class LHSSampling(Sampling):
    """
    Latin Hypercube Sampling for NSGA-II initial population.

    LHS divides each dimension into n equal-probability strata
    and places exactly one sample per stratum — this guarantees
    much better coverage of the [P, S, F] design space than
    pure random initialisation, at zero extra computational cost.

    Better coverage → NSGA-II finds diverse non-dominated
    solutions in fewer generations.
    """

    def _do(self, problem, n_samples, **kwargs):
        sampler = qmc.LatinHypercube(d=N_VARS, seed=42)
        sample  = sampler.random(n=n_samples)
        # Scale from [0,1]^3 to physical machine bounds
        return qmc.scale(sample, BOUNDS_LOWER, BOUNDS_UPPER)

# ─────────────────────────────────────────────────────────────
# SECTION 5 — THREE-OBJECTIVE PYMOO PROBLEM DEFINITION
# ─────────────────────────────────────────────────────────────

class LMDThreeObjective(Problem):
    """
    Three-objective LMD bead geometry inverse optimisation.

    Decision variables
    ------------------
    x[0] : Laser Power   [W]      ∈ [1600, 2800]
    x[1] : Scan Speed    [mm/s]   ∈ [0.8,  1.6]
    x[2] : Powder Feed   [g/min]  ∈ [2.0,  8.0]

    Objectives  (all minimised)
    ---------------------------
    f[0] : |H_pred  − H_target|    [µm]
    f[1] : |AR_pred − AR_target|   [—]
    f[2] : |D_pred  − D_target|    [%]

    Constraints  (pymoo: g ≤ 0 convention)
    ----------------------------------------
    g[0] : D_MIN − D_pred ≤ 0   →  D_pred ≥ D_MIN (15%)
    g[1] : D_pred − D_MAX ≤ 0   →  D_pred ≤ D_MAX (45%)

    Surrogate performance (5-fold CV, N=89):
      Height   R²=0.9552  MAE=64.6 µm
      AR       R²=0.8739  MAE=0.038
      Dilution R²=0.9023  MAE=3.4%
    """

    def __init__(self):
        super().__init__(
            n_var    = N_VARS,
            n_obj    = 3,   # H error | AR deviation | D error
            n_constr = 2,   # D lower bound | D upper bound
            xl       = BOUNDS_LOWER.copy(),
            xu       = BOUNDS_UPPER.copy(),
        )

    def _evaluate(self, X, out, *args, **kwargs):
        H_p, AR_p, D_p = predict_joint(X)

        # Objectives — all minimised
        out["F"] = np.column_stack([
            np.abs(H_p  - H_TARGET),    # f0 [µm]
            np.abs(AR_p - AR_TARGET),   # f1 [—]
            np.abs(D_p  - D_TARGET),    # f2 [%]
        ])

        # Hard constraints — g ≤ 0 must hold for feasibility
        out["G"] = np.column_stack([
            D_MIN - D_p,   # g0: D_pred ≥ D_MIN  (infeasible if > 0)
            D_p - D_MAX,   # g1: D_pred ≤ D_MAX  (infeasible if > 0)
        ])

# ─────────────────────────────────────────────────────────────
# SECTION 6 — RUN NSGA-II
# ─────────────────────────────────────────────────────────────

print("\n" + "=" * 65)
print("STEP 2 — Running NSGA-II  (3 objectives + dilution constraints)")
print(f"  Population : {POP_SIZE}")
print(f"  Generations: {N_GEN}")
print(f"  Sampling   : Latin Hypercube (better than pure random)")
print(f"  Crossover  : SBX  (η=15, prob=0.9)")
print(f"  Mutation   : PM   (η=20, prob=1/{N_VARS}≈{1/N_VARS:.2f})")
print(f"  Constraints: {D_MIN}% ≤ D_pred ≤ {D_MAX}%")
print("=" * 65)

problem   = LMDThreeObjective()
algorithm = NSGA2(
    pop_size            = POP_SIZE,
    sampling            = LHSSampling(),       # LHS init
    crossover           = SBX(eta=15, prob=0.9),
    mutation            = PM(eta=20, prob=1.0 / N_VARS),
    eliminate_duplicates= True,                # no duplicate solutions
)
termination = get_termination("n_gen", N_GEN)

t0     = time.time()
result = minimize(
    problem,
    algorithm,
    termination,
    seed         = 42,
    save_history = True,   # needed for convergence plots
    verbose      = True,
)
t_nsga = time.time() - t0

print(f"\n  ✓ NSGA-II complete in {t_nsga:.1f}s")
print(f"    Pareto-optimal solutions: {len(result.X)}")

# ─────────────────────────────────────────────────────────────
# SECTION 7 — ENRICH PARETO FRONT WITH UNCERTAINTY
# ─────────────────────────────────────────────────────────────

pareto_X = result.X   # (n_pareto, 3) — optimal input parameters
pareto_F = result.F   # (n_pareto, 3) — objective values

# Re-evaluate with full uncertainty — called once post-optimisation
# (return_std=True is slower; no need to call during evolution)
H_p, AR_p, D_p, sig_H, sig_AR, sig_D = \
    predict_with_uncertainty(pareto_X)

# Feasibility flag — dilution within metallurgical bounds
feasible = (D_p >= D_MIN) & (D_p <= D_MAX)
n_feas   = int(feasible.sum())

# Composite Tchebycheff score for ranking balanced solutions
# f(x) = max(dH, dAR, dD) + ρ·(dH + dAR + dD)
# where d_i = |f_i| / σ_i  (dimensionless normalised error)
RHO    = 0.05
dH     = pareto_F[:, 0] / SCALE_H
dAR    = pareto_F[:, 1] / SCALE_AR
dD_obj = pareto_F[:, 2] / SCALE_D
comp_f = (np.maximum(np.maximum(dH, dAR), dD_obj)
          + RHO * (dH + dAR + dD_obj))

# Uncertainty score — sum of normalised posterior σ values
# Low score = GPR confident = solution is near training data
unc_score = sig_H / SCALE_H + sig_AR / SCALE_AR + sig_D / SCALE_D

pareto_df = pd.DataFrame({
    # ── Input parameters ──────────────────────────────────
    "Power_W"          : pareto_X[:, 0],
    "Speed_mm_s"       : pareto_X[:, 1],
    "Feed_g_min"       : pareto_X[:, 2],
    # ── GPR point predictions ─────────────────────────────
    "H_pred_um"        : H_p,
    "AR_pred"          : AR_p,
    "D_pred_pct"       : D_p,
    # ── Objective values (errors from target) ─────────────
    "Height_Err_um"    : pareto_F[:, 0],
    "AR_Dev"           : pareto_F[:, 1],
    "Dilution_Err_pct" : pareto_F[:, 2],
    # ── GPR uncertainty (posterior std dev) ───────────────
    "Sigma_H_um"       : sig_H,
    "Sigma_AR"         : sig_AR,
    "Sigma_D_pct"      : sig_D,
    # ── Derived metrics ───────────────────────────────────
    "Feasible_D"       : feasible,
    "Uncertainty_Score": unc_score,
    "Composite_f"      : comp_f,
})
pareto_df.sort_values("Height_Err_um", inplace=True)
pareto_df.reset_index(drop=True, inplace=True)

print(f"\n  Dilution-feasible Pareto solutions: "
      f"{n_feas}/{len(pareto_df)} "
      f"({100*n_feas/max(len(pareto_df), 1):.0f}%)")

# ─────────────────────────────────────────────────────────────
# SECTION 8 — KNEE-POINT AND CANDIDATE SELECTION
# ─────────────────────────────────────────────────────────────
# ASF (Achievement Scalarisation Function) identifies the knee:
# the Pareto solution making the most balanced progress on all
# three objectives simultaneously (equal weights = 1/3 each).
# Knee search is restricted to dilution-feasible solutions.
# ─────────────────────────────────────────────────────────────

weights = np.array([1/3, 1/3, 1/3])
decomp  = ASF()

if n_feas > 0:
    feas_mask  = pareto_df["Feasible_D"].values.astype(bool)
    feas_F     = pareto_F[feas_mask]
    feas_idx   = np.where(feas_mask)[0]
    knee_local = decomp.do(feas_F, 1.0 / weights).argmin()
    knee_idx   = int(feas_idx[knee_local])
    print("  Knee selected from feasible-dilution subset ✅")
else:
    knee_idx = int(decomp.do(pareto_F, 1.0 / weights).argmin())
    print("  ⚠️  No feasible solutions — knee from full Pareto front")

knee_params = pareto_X[knee_idx]
knee_objs   = pareto_F[knee_idx]
H_k, AR_k, D_k, sH_k, sAR_k, sD_k = \
    predict_with_uncertainty(knee_params.reshape(1, -1))
feas_k = D_MIN <= D_k[0] <= D_MAX

print("\n" + "=" * 65)
print("STEP 3 — Knee-Point (Best Balanced Trade-Off Solution)")
print("=" * 65)
print(f"  Laser Power    : {knee_params[0]:.1f} W")
print(f"  Scan Speed     : {knee_params[1]:.4f} mm/s")
print(f"  Powder Feed    : {knee_params[2]:.3f} g/min")
print(f"  Pred. Height   : {H_k[0]:.1f} µm   "
      f"(target {H_TARGET:.0f} µm,  err ±{knee_objs[0]:.1f} µm)")
print(f"  Pred. AR       : {AR_k[0]:.4f}      "
      f"(target {AR_TARGET:.3f},    dev ±{knee_objs[1]:.4f})")
print(f"  Pred. Dilution : {D_k[0]:.2f}%      "
      f"(target {D_TARGET:.1f}%,    err ±{knee_objs[2]:.2f}%)")
print(f"  GPR σ          : "
      f"σ_H=±{sH_k[0]:.1f} µm | "
      f"σ_AR=±{sAR_k[0]:.4f} | "
      f"σ_D=±{sD_k[0]:.2f}%")
print(f"  Dilution OK    : {'✅ feasible' if feas_k else '❌ infeasible'}")

# ── Named candidates for physical validation experiments ──────
candidates = {}
candidates["Knee (balanced)"]    = pareto_df.loc[knee_idx]
candidates["Min Height Error"]   = pareto_df.loc[
    pareto_df["Height_Err_um"].idxmin()]
candidates["Min AR Deviation"]   = pareto_df.loc[
    pareto_df["AR_Dev"].idxmin()]
candidates["Min Dilution Error"] = pareto_df.loc[
    pareto_df["Dilution_Err_pct"].idxmin()]
candidates["Min Uncertainty"]    = pareto_df.loc[
    pareto_df["Uncertainty_Score"].idxmin()]
if n_feas > 0:
    feas_df = pareto_df[pareto_df["Feasible_D"]]
    candidates["Best Feasible (D)"] = feas_df.loc[
        feas_df["Composite_f"].idxmin()]

print("\n" + "=" * 65)
print("STEP 4 — Recommended Experimental Candidates")
print("=" * 65)
print(f"{'Label':<24} {'P(W)':>7} {'S':>7} {'F':>6} "
      f"{'H(µm)':>8} {'AR':>7} {'D(%)':>7} "
      f"{'σH':>6} {'σAR':>6} {'σD':>5} {'Feas':>5}")
print("─" * 90)
for lbl, row in candidates.items():
    print(f"{lbl:<24} {row.Power_W:>7.1f} {row.Speed_mm_s:>7.4f} "
          f"{row.Feed_g_min:>6.2f} {row.H_pred_um:>8.1f} "
          f"{row.AR_pred:>7.4f} {row.D_pred_pct:>7.2f} "
          f"{row.Sigma_H_um:>6.1f} {row.Sigma_AR:>6.4f} "
          f"{row.Sigma_D_pct:>5.2f} "
          f"{'✅' if row.Feasible_D else '❌':>5}")

# ─────────────────────────────────────────────────────────────
# SECTION 9 — HYPERVOLUME CONVERGENCE HISTORY
# Hypervolume = single scalar that captures overall quality
# of the Pareto front — better than per-objective min tracking
# ─────────────────────────────────────────────────────────────

nadir   = pareto_F.max(axis=0) * 1.10   # reference point = nadir + 10%
hv_ind  = HV(ref_point=nadir)
hv_hist = []

if result.history:
    for gen in result.history:
        F_gen = gen.opt.get("F")
        try:
            hv_hist.append(float(hv_ind(F_gen)))
        except Exception:
            hv_hist.append(np.nan)

# ─────────────────────────────────────────────────────────────
# SECTION 10 — PUBLICATION-QUALITY FIGURES (600 dpi)
# ─────────────────────────────────────────────────────────────

print("\n" + "=" * 65)
print("STEP 5 — Generating publication-quality figures")
print("=" * 65)

unc_vals  = pareto_df["Uncertainty_Score"].values
feas_vals = pareto_df["Feasible_D"].values.astype(float)

def _mark_knee(ax, xv, yv):
    """Consistent knee-point marker across all 2-D panels."""
    ax.scatter(xv, yv, c=C_BLUE, s=260, zorder=9,
               marker='*', edgecolors='navy', linewidths=0.8,
               label='Knee point', clip_on=False)

# ─────────────────────────────────────────────────────────────
# FIGURE 1 — Three Pareto front projections (2-D slices)
# Main result figure for thesis / journal
# ─────────────────────────────────────────────────────────────
fig1, axes1 = plt.subplots(1, 3, figsize=(16, 5))
fig1.suptitle(
    "NSGA-II Three-Objective Pareto Front — LMD Bead Geometry\n"
    f"Surrogate: 3-output GPR (N=89)  |  "
    f"Targets: H={H_TARGET:.0f} µm, AR={AR_TARGET:.2f}, D={D_TARGET:.0f}%",
    fontsize=12, fontweight='bold', y=1.02
)

# Panel A: Height Error vs AR Deviation (coloured by uncertainty)
ax = axes1[0]
sc = ax.scatter(pareto_F[:, 0], pareto_F[:, 1],
                c=unc_vals, cmap='RdYlGn_r',
                s=50, alpha=0.80, edgecolors='none', zorder=3)
plt.colorbar(sc, ax=ax, pad=0.02, label='Uncertainty Score')
_mark_knee(ax, knee_objs[0], knee_objs[1])
ax.set_xlabel('Height Error (µm)')
ax.set_ylabel('AR Deviation')
ax.set_title('f₀ vs f₁\nHeight Error vs AR Deviation')
ax.legend(fontsize=8, loc='upper right')

# Panel B: Height Error vs Dilution Error (coloured by feasibility)
ax = axes1[1]
sc = ax.scatter(pareto_F[:, 0], pareto_F[:, 2],
                c=feas_vals, cmap='RdYlGn',
                s=50, alpha=0.80, edgecolors='none', zorder=3)
plt.colorbar(sc, ax=ax, pad=0.02,
             label='Dilution Feasible\n(1 = ✅   0 = ❌)')
_mark_knee(ax, knee_objs[0], knee_objs[2])
ax.set_xlabel('Height Error (µm)')
ax.set_ylabel('Dilution Error (%)')
ax.set_title('f₀ vs f₂\nHeight Error vs Dilution Error')
ax.legend(fontsize=8, loc='upper right')

# Panel C: AR Deviation vs Dilution Error (coloured by uncertainty)
ax = axes1[2]
sc = ax.scatter(pareto_F[:, 1], pareto_F[:, 2],
                c=unc_vals, cmap='RdYlGn_r',
                s=50, alpha=0.80, edgecolors='none', zorder=3)
plt.colorbar(sc, ax=ax, pad=0.02, label='Uncertainty Score')
_mark_knee(ax, knee_objs[1], knee_objs[2])
ax.set_xlabel('AR Deviation')
ax.set_ylabel('Dilution Error (%)')
ax.set_title('f₁ vs f₂\nAR Deviation vs Dilution Error')
ax.legend(fontsize=8, loc='upper right')

plt.tight_layout()
fig1.savefig("fig_pareto_projections2.png", bbox_inches='tight')
plt.close(fig1)
print("  ✓ Saved: fig_pareto_projections2.png")

# ─────────────────────────────────────────────────────────────
# FIGURE 2 — 3-D Pareto surface
# Gives reviewers the full trade-off geometry in one view
# ─────────────────────────────────────────────────────────────
fig2 = plt.figure(figsize=(9, 7))
ax3d = fig2.add_subplot(111, projection='3d')

sc3d = ax3d.scatter(
    pareto_F[:, 0], pareto_F[:, 1], pareto_F[:, 2],
    c=unc_vals, cmap='RdYlGn_r',
    s=55, alpha=0.80, edgecolors='none', zorder=3
)
# Knee point marker
ax3d.scatter(
    [knee_objs[0]], [knee_objs[1]], [knee_objs[2]],
    c=C_BLUE, s=280, marker='*',
    edgecolors='navy', linewidths=0.8,
    zorder=10, label='Knee point'
)
cb3d = fig2.colorbar(sc3d, ax=ax3d, pad=0.12, shrink=0.6)
cb3d.set_label('Uncertainty Score\n(green = GPR-confident)', fontsize=9)
ax3d.set_xlabel('Height Error (µm)', labelpad=8)
ax3d.set_ylabel('AR Deviation',       labelpad=8)
ax3d.set_zlabel('Dilution Error (%)', labelpad=8)
ax3d.set_title(
    'Three-Objective Pareto Surface\nNSGA-II | 3-output GPR Surrogate',
    fontweight='bold'
)
ax3d.legend(fontsize=9)
ax3d.view_init(elev=25, azim=135)
fig2.tight_layout()
fig2.savefig("fig_pareto_3d_surface2.png", bbox_inches='tight')
plt.close(fig2)
print("  ✓ Saved: fig_pareto_3d_surface2.png")

# ─────────────────────────────────────────────────────────────
# FIGURE 3 — NSGA-II convergence + Hypervolume
# Demonstrates algorithm convergence for methodology section
# ─────────────────────────────────────────────────────────────
fig3, axes3 = plt.subplots(1, 2, figsize=(13, 5))
fig3.suptitle(
    "NSGA-II Convergence Diagnostics",
    fontsize=12, fontweight='bold'
)

# Panel A — Per-generation best objective values
ax = axes3[0]
if result.history:
    gen_h  = [np.min(g.opt.get("F")[:, 0]) for g in result.history]
    gen_ar = [np.min(g.opt.get("F")[:, 1]) for g in result.history]
    gen_d  = [np.min(g.opt.get("F")[:, 2]) for g in result.history]
    gens   = range(1, len(gen_h) + 1)
    ax.plot(gens, gen_h,  color=C_BLUE,   lw=1.8, label='Min Height Err (µm)')
    ax.plot(gens, gen_ar, color=C_RED,    lw=1.8, label='Min AR Deviation')
    ax.plot(gens, gen_d,  color=C_GREEN,  lw=1.8, label='Min Dilution Err (%)')
ax.set_xlabel('Generation')
ax.set_ylabel('Best Objective Value')
ax.set_title(f'Best Objective per Generation\n'
             f'(pop={POP_SIZE}, gen={N_GEN}, LHS init)')
ax.legend()

# Panel B — Hypervolume indicator per generation
ax = axes3[1]
if hv_hist:
    ax.plot(range(1, len(hv_hist) + 1), hv_hist,
            color=C_PURPLE, lw=2, label='Hypervolume')
    # Mark final HV value
    ax.axhline(hv_hist[-1], color=C_RED, ls='--', lw=1.4,
               label=f'Final HV = {hv_hist[-1]:.4f}')
ax.set_xlabel('Generation')
ax.set_ylabel('Hypervolume Indicator')
ax.set_title('Pareto Front Quality (Hypervolume)\n'
             'Higher = larger dominated space = better front')
ax.legend()

plt.tight_layout()
fig3.savefig("fig_convergence_diagnostics2.png", bbox_inches='tight')
plt.close(fig3)
print("  ✓ Saved: fig_convergence_diagnostics2.png")

# ─────────────────────────────────────────────────────────────
# FIGURE 4 — Pareto solutions in LMD parameter space
# Shows which machine settings produce Pareto-optimal beads
# ─────────────────────────────────────────────────────────────
fig4, axes4 = plt.subplots(1, 3, figsize=(16, 5))
fig4.suptitle(
    "Pareto-Optimal Solutions in LMD Parameter Space",
    fontsize=12, fontweight='bold'
)

# Panel A — Power vs Feed  (coloured by Height Error)
ax = axes4[0]
sc = ax.scatter(pareto_df["Power_W"], pareto_df["Feed_g_min"],
                c=pareto_df["Height_Err_um"],
                cmap='coolwarm', s=50, alpha=0.80, edgecolors='none')
plt.colorbar(sc, ax=ax, pad=0.02, label='Height Error (µm)')
ax.scatter(knee_params[0], knee_params[2],
           c=C_BLUE, s=230, marker='*', edgecolors='navy',
           linewidths=0.8, zorder=6, label='Knee point')
ax.set_xlabel('Laser Power (W)')
ax.set_ylabel('Powder Feed (g/min)')
ax.set_title('Power vs Feed\n(colour = Height Error)')
ax.legend(fontsize=8)

# Panel B — Power vs Speed  (coloured by Predicted Dilution)
ax = axes4[1]
sc = ax.scatter(pareto_df["Power_W"], pareto_df["Speed_mm_s"],
                c=pareto_df["D_pred_pct"],
                cmap='RdYlGn', s=50, alpha=0.80, edgecolors='none')
plt.colorbar(sc, ax=ax, pad=0.02, label='Pred. Dilution (%)')
ax.scatter(knee_params[0], knee_params[1],
           c=C_BLUE, s=230, marker='*', edgecolors='navy',
           linewidths=0.8, zorder=6, label='Knee point')
ax.set_xlabel('Laser Power (W)')
ax.set_ylabel('Scan Speed (mm/s)')
ax.set_title('Power vs Speed\n(colour = Pred. Dilution %)')
ax.legend(fontsize=8)

# Panel C — Speed vs Feed  (coloured by Predicted AR)
ax = axes4[2]
sc = ax.scatter(pareto_df["Speed_mm_s"], pareto_df["Feed_g_min"],
                c=pareto_df["AR_pred"],
                cmap='viridis', s=50, alpha=0.80, edgecolors='none')
plt.colorbar(sc, ax=ax, pad=0.02, label='Pred. Aspect Ratio')
ax.scatter(knee_params[1], knee_params[2],
           c=C_BLUE, s=230, marker='*', edgecolors='navy',
           linewidths=0.8, zorder=6, label='Knee point')
ax.set_xlabel('Scan Speed (mm/s)')
ax.set_ylabel('Powder Feed (g/min)')
ax.set_title('Speed vs Feed\n(colour = Pred. AR)')
ax.legend(fontsize=8)

plt.tight_layout()
fig4.savefig("fig_parameter_space2.png", bbox_inches='tight')
plt.close(fig4)
print("  ✓ Saved: fig_parameter_space2.png")

# ─────────────────────────────────────────────────────────────
# FIGURE 5 — Dilution feasibility + GPR uncertainty scatter
# Critical for validating which candidates are trustworthy
# ─────────────────────────────────────────────────────────────
fig5, axes5 = plt.subplots(1, 2, figsize=(13, 5))
fig5.suptitle(
    "Dilution Feasibility and GPR Surrogate Uncertainty\n"
    "across Pareto-Optimal Solutions",
    fontsize=12, fontweight='bold'
)

# Panel A — Predicted dilution vs Power with feasibility bands
ax = axes5[0]
clrs = [C_GREEN if f else C_RED for f in pareto_df["Feasible_D"]]
ax.scatter(pareto_df["Power_W"], pareto_df["D_pred_pct"],
           c=clrs, s=55, alpha=0.80,
           edgecolors='black', linewidths=0.3, zorder=3)
ax.axhspan(D_MIN, D_MAX, alpha=0.08, color=C_GREEN,
           label=f'Feasible zone [{D_MIN:.0f}%–{D_MAX:.0f}%]')
ax.axhline(D_TARGET, color=C_GREEN,  ls='-',  lw=1.8,
           label=f'Target D = {D_TARGET:.0f}%')
ax.axhline(D_MIN,    color=C_RED,    ls='--', lw=1.4,
           label=f'D_min = {D_MIN:.0f}%')
ax.axhline(D_MAX,    color=C_ORANGE, ls='--', lw=1.4,
           label=f'D_max = {D_MAX:.0f}%')
ax.scatter(knee_params[0], float(D_k[0]),
           c=C_BLUE, s=260, marker='*', edgecolors='navy',
           linewidths=0.8, zorder=8, label='Knee point')
ax.set_xlabel('Laser Power (W)')
ax.set_ylabel('Predicted Dilution (%)')
ax.set_title('Dilution Feasibility Check\nGreen = ✅ Feasible   Red = ❌ Infeasible')
ax.legend(fontsize=8)

# Panel B — GPR σ_H vs σ_D  (coloured by height error)
# Bottom-left quadrant = most reliable candidates
ax = axes5[1]
sc = ax.scatter(pareto_df["Sigma_H_um"], pareto_df["Sigma_D_pct"],
                c=pareto_F[:, 0], cmap='viridis',
                s=55, alpha=0.80, edgecolors='none')
plt.colorbar(sc, ax=ax, pad=0.02, label='Height Error (µm)')
ax.axvline(50,  color=C_RED,    ls='--', lw=1.4, alpha=0.8,
           label='σ_H = 50 µm  (reliability threshold)')
ax.axhline(2.0, color=C_ORANGE, ls='--', lw=1.4, alpha=0.8,
           label='σ_D = 2%  (reliability threshold)')
ax.scatter(float(sH_k[0]), float(sD_k[0]),
           c=C_BLUE, s=260, marker='*', edgecolors='navy',
           linewidths=0.8, zorder=8, label='Knee point')
ax.set_xlabel('GPR σ — Height (µm)')
ax.set_ylabel('GPR σ — Dilution (%)')
ax.set_title('Surrogate Prediction Uncertainty\n'
             'Bottom-left = most reliable (near training data)')
ax.legend(fontsize=8)

plt.tight_layout()
fig5.savefig("fig_feasibility_uncertainty2.png", bbox_inches='tight')
plt.close(fig5)
print("  ✓ Saved: fig_feasibility_uncertainty2.png")

# ─────────────────────────────────────────────────────────────
# FIGURE 6 — Candidate comparison bar chart
# Quick visual summary for thesis results chapter
# ─────────────────────────────────────────────────────────────
cand_labels = list(candidates.keys())
n_cands     = len(cand_labels)
x           = np.arange(n_cands)
width       = 0.25

h_errs  = [candidates[l]["Height_Err_um"]    / SCALE_H  for l in cand_labels]
ar_devs = [candidates[l]["AR_Dev"]           / SCALE_AR for l in cand_labels]
d_errs  = [candidates[l]["Dilution_Err_pct"] / SCALE_D  for l in cand_labels]

fig6, ax = plt.subplots(figsize=(13, 5))
ax.bar(x - width, h_errs,  width, color=C_BLUE,   alpha=0.85,
       label='Norm. Height Error  (|H−H*|/σ_H)')
ax.bar(x,         ar_devs, width, color=C_RED,    alpha=0.85,
       label='Norm. AR Deviation  (|AR−AR*|/σ_AR)')
ax.bar(x + width, d_errs,  width, color=C_GREEN,  alpha=0.85,
       label='Norm. Dilution Error (|D−D*|/σ_D)')
ax.set_xticks(x)
ax.set_xticklabels(cand_labels, rotation=20, ha='right', fontsize=9)
ax.set_ylabel('Normalised Objective Value  (lower = better)')
ax.set_title('Experimental Candidate Comparison\n'
             'All objectives normalised by training-data σ for fair comparison',
             fontweight='bold')
ax.legend(fontsize=9)
ax.axhline(1.0, color='grey', ls=':', lw=1.2,
           label='σ = 1 reference line')
plt.tight_layout()
fig6.savefig("fig_candidate_comparison2.png", bbox_inches='tight')
plt.close(fig6)
print("  ✓ Saved: fig_candidate_comparison2.png")

# ─────────────────────────────────────────────────────────────
# SECTION 11 — SAVE ALL OUTPUTS TO CSV
# ─────────────────────────────────────────────────────────────

# Full Pareto front with all enriched columns
pareto_df.to_csv("nsga2_pareto_full1.csv", index=False)
print("  ✓ Saved: nsga2_pareto_full1.csv")

# Hypervolume convergence history
if hv_hist:
    pd.DataFrame({
        "generation" : range(1, len(hv_hist) + 1),
        "hypervolume": hv_hist,
    }).to_csv("nsga2_hypervolume_history2.csv", index=False)
    print("  ✓ Saved: nsga2_hypervolume_history2.csv")

# Per-generation best objectives
if result.history:
    pd.DataFrame({
        "generation"      : range(1, len(gen_h) + 1),
        "min_height_err"  : gen_h,
        "min_ar_dev"      : gen_ar,
        "min_dilution_err": gen_d,
    }).to_csv("nsga2_generation_history2.csv", index=False)
    print("  ✓ Saved: nsga2_generation_history2.csv")

# Experimental validation candidates
cand_rows = []
for lbl, row in candidates.items():
    cand_rows.append({
        "Label"            : lbl,
        "Power_W"          : round(float(row["Power_W"]),           1),
        "Speed_mm_s"       : round(float(row["Speed_mm_s"]),         4),
        "Feed_g_min"       : round(float(row["Feed_g_min"]),         3),
        "H_pred_um"        : round(float(row["H_pred_um"]),          1),
        "AR_pred"          : round(float(row["AR_pred"]),            4),
        "D_pred_pct"       : round(float(row["D_pred_pct"]),         2),
        "Height_Err_um"    : round(float(row["Height_Err_um"]),      1),
        "AR_Dev"           : round(float(row["AR_Dev"]),             5),
        "Dilution_Err_pct" : round(float(row["Dilution_Err_pct"]),   2),
        "Sigma_H_um"       : round(float(row["Sigma_H_um"]),         1),
        "Sigma_AR"         : round(float(row["Sigma_AR"]),           5),
        "Sigma_D_pct"      : round(float(row["Sigma_D_pct"]),        2),
        "Feasible_D"       : bool(row["Feasible_D"]),
        "Composite_f"      : round(float(row["Composite_f"]),        5),
        "Uncertainty_Score": round(float(row["Uncertainty_Score"]),  5),
    })
pd.DataFrame(cand_rows).to_csv(
    "nsga2_validation_candidates2.csv", index=False)
print("  ✓ Saved: nsga2_validation_candidates2.csv")

# ─────────────────────────────────────────────────────────────
# SECTION 12 — FINAL SUMMARY
# ─────────────────────────────────────────────────────────────

print("\n" + "=" * 65)
print("PHASE 1B COMPLETE — NSGA-II Three-Objective Optimisation")
print("=" * 65)
print(f"  NSGA-II runtime    : {t_nsga:.1f}s")
print(f"  Pareto solutions   : {len(pareto_df)}")
print(f"  Feasible (D)       : {n_feas}/{len(pareto_df)} "
      f"({100*n_feas/max(len(pareto_df), 1):.0f}%)")
print(f"  Knee point         : "
      f"P={knee_params[0]:.1f} W  |  "
      f"S={knee_params[1]:.4f} mm/s  |  "
      f"F={knee_params[2]:.3f} g/min")
print(f"  Knee predictions   : "
      f"H={H_k[0]:.1f} µm  |  "
      f"AR={AR_k[0]:.4f}  |  "
      f"D={D_k[0]:.2f}%")
print(f"  Knee feasible      : {'✅' if feas_k else '❌'}")
print(f"\n  Figures saved:")
print(f"    fig_pareto_projections.png      — 3 × 2D Pareto slices")
print(f"    fig_pareto_3d_surface.png       — 3D Pareto surface")
print(f"    fig_convergence_diagnostics.png — objectives + hypervolume")
print(f"    fig_parameter_space.png         — solutions in [P, S, F] space")
print(f"    fig_feasibility_uncertainty.png — dilution bounds + GPR σ")
print(f"    fig_candidate_comparison.png    — normalised candidate bar chart")
print(f"\n  Data saved:")
print(f"    nsga2_pareto_full.csv")
print(f"    nsga2_validation_candidates.csv")
print(f"    nsga2_hypervolume_history.csv")
print(f"    nsga2_generation_history.csv")
print("=" * 65)
