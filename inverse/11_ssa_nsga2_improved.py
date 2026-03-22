# ============================================================
# 10_ssa_nsga2_optimisation.py  (v3 — 3-target, NSGA-II)
# Phase 1B — Hybrid SSA-NSGA-II Multi-Objective Optimisation
# Surrogate : 3-output Multi-Output GPR
#             (Height µm  |  Aspect Ratio  |  Dilution %)
# Targets   : LMD Al-Si bead geometry inverse optimisation
# ============================================================
#
# ALGORITHM OVERVIEW
# ─────────────────────────────────────────────────────────
# Two-stage hybrid strategy:
#   Stage 1 — SSA pre-search
#     Sparrow Search Algorithm (Xue & Shen 2020) collapses
#     the 3-objective problem to a single Augmented
#     Tchebycheff scalar and rapidly maps the high-quality
#     region of the design space.
#     Init: Latin Hypercube Sampling (better space-filling
#           than pure random at no extra cost)
#
#   Stage 2 — NSGA-II refinement
#     Warm-started with SSA elites (60% of pop_size),
#     NSGA-II expands outward to recover the full 3D Pareto
#     surface using non-dominated sorting + crowding distance.
#     Hard dilution constraints (15% ≤ D ≤ 45%) enforced
#     via pymoo's constraint handling.
#
# References
#   Xue J, Shen B. Systems Sci & Control Eng 2020;8(1):22-34
#   Deb K et al. IEEE Trans Evol Comput 2002;6(2):182-197
#   Ishibuchi H et al. CEC 2016 — NSGA-II vs NSGA-III study
# ============================================================

import numpy as np
import pandas as pd
import joblib
import time
import warnings
from scipy.stats import qmc                      # Latin Hypercube init

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D          # 3-D Pareto surface

from pymoo.core.problem    import Problem
from pymoo.core.sampling   import Sampling
from pymoo.algorithms.moo.nsga2 import NSGA2     # ← NSGA-II (not III)
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm   import PM
from pymoo.termination              import get_termination
from pymoo.optimize                 import minimize
from pymoo.decomposition.asf        import ASF
from pymoo.indicators.hv            import HV    # hypervolume tracking

warnings.filterwarnings('ignore')
np.random.seed(42)

# ─────────────────────────────────────────────────────────────
# PUBLICATION STYLE — applied globally
# All figures use this style for journal / thesis consistency
# ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family'      : 'DejaVu Serif',
    'font.size'        : 11,
    'axes.titlesize'   : 12,
    'axes.labelsize'   : 11,
    'axes.titleweight' : 'bold',
    'axes.labelweight' : 'bold',
    'xtick.labelsize'  : 9,
    'ytick.labelsize'  : 9,
    'legend.fontsize'  : 9,
    'figure.dpi'       : 150,
    'savefig.dpi'      : 600,          # 600 dpi — journal requirement
    'savefig.format'   : 'png',
    'lines.linewidth'  : 1.8,
    'axes.grid'        : True,
    'grid.alpha'       : 0.3,
    'axes.spines.top'  : False,
    'axes.spines.right': False,
})

# Colour palette — consistent across all panels
C_BLUE   = '#2c7bb6'
C_RED    = '#d7191c'
C_GREEN  = '#1a9641'
C_ORANGE = '#fdae61'
C_PURPLE = '#762a83'
C_GOLD   = '#f4a582'

# ─────────────────────────────────────────────────────────────
# SECTION 1 — LOAD 3-OUTPUT GPR AND ALL SCALERS
# ─────────────────────────────────────────────────────────────
# Saved from notebook Cell 16 (Improvement D — corrected):
#   multi_gpr.estimators_[0] → Height GPR
#   multi_gpr.estimators_[1] → Aspect Ratio GPR
#   multi_gpr.estimators_[2] → Dilution % GPR
# ─────────────────────────────────────────────────────────────

print("=" * 65)
print("STEP 1 — Loading 3-output GPR surrogate and scalers")
print("=" * 65)

multi_gpr  = joblib.load("multi_output_gpr.pkl")
scaler_X   = joblib.load("scaler_X.pkl")
scaler_H   = joblib.load("scaler_height.pkl")
scaler_A   = joblib.load("scaler_aspect.pkl")
scaler_D   = joblib.load("scaler_dilution.pkl")

# Hard validation — catches stale .pkl files immediately
assert hasattr(multi_gpr, 'estimators_'), \
    "multi_output_gpr.pkl is not a fitted MultiOutputRegressor. " \
    "Re-run notebook Cell 16 (Improvement D) and re-save."
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
# Edit this section to change the inverse problem targets.
# ─────────────────────────────────────────────────────────────

# ── Desired bead geometry ────────────────────────────────────
H_TARGET  = 1100.0   # µm   — target bead height
AR_TARGET =    0.44  #      — target aspect ratio (Width/Height)
D_TARGET  =   30.0   # %    — target dilution percentage

# ── Dilution hard constraints (metallurgical bounds) ─────────
# < 15 % → insufficient bonding to substrate
# > 45 % → excessive base-metal mixing, degrades alloy quality
D_MIN = 15.0          # %
D_MAX = 45.0          # %

# ── Tchebycheff normalisation: divide by target σ so all
#    objectives are dimensionless and equally weighted ────────
SCALE_H   = float(scaler_H.scale_[0])   # ≈ 426 µm
SCALE_AR  = float(scaler_A.scale_[0])   # ≈ 0.215
SCALE_D   = float(scaler_D.scale_[0])   # ≈ 14.2 %
RHO       = 0.05    # Augmented Tchebycheff regularisation term
                    # prevents degenerate solutions on Pareto edges

# ── LMD machine design space bounds ─────────────────────────
BOUNDS_LOWER = np.array([1600.0, 0.8, 2.0])  # [Power W, Speed mm/s, Feed g/min]
BOUNDS_UPPER = np.array([2800.0, 1.6, 8.0])
N_VARS = 3

print(f"\n  Targets   : H={H_TARGET:.1f} µm | "
      f"AR={AR_TARGET:.3f} | D={D_TARGET:.1f}%")
print(f"  D bounds  : [{D_MIN:.0f}%, {D_MAX:.0f}%]  (hard constraint)")
print(f"  σ scales  : σ_H={SCALE_H:.1f} µm | "
      f"σ_AR={SCALE_AR:.4f} | σ_D={SCALE_D:.2f}%")

# ─────────────────────────────────────────────────────────────
# SECTION 3 — SURROGATE PREDICTION FUNCTIONS
# ─────────────────────────────────────────────────────────────

def _build_features(X_raw: np.ndarray) -> np.ndarray:
    """
    Build 4-column feature matrix from raw [P, S, F].
    Mirrors Cell 3 of notebook exactly:
      columns: [Power, Speed, Feed, HeatInput = Power/(Speed+ε)]
    ε = 1e-9 prevents division-by-zero at Speed → 0.
    """
    X_raw = np.atleast_2d(X_raw)
    heat  = X_raw[:, 0] / (X_raw[:, 1] + 1e-9)
    return np.column_stack([X_raw, heat])


def predict_joint(X_raw: np.ndarray):
    """
    Point prediction for all three targets.

    Parameters
    ----------
    X_raw : (n, 3) — [Power W, Speed mm/s, Feed g/min]

    Returns
    -------
    H_pred  : (n,) µm
    AR_pred : (n,) dimensionless
    D_pred  : (n,) %
    """
    X_sc  = scaler_X.transform(_build_features(np.atleast_2d(X_raw)))
    Y_sc  = multi_gpr.predict(X_sc)           # shape (n, 3)
    H_p   = scaler_H.inverse_transform(Y_sc[:, 0].reshape(-1, 1)).ravel()
    AR_p  = scaler_A.inverse_transform(Y_sc[:, 1].reshape(-1, 1)).ravel()
    D_p   = scaler_D.inverse_transform(Y_sc[:, 2].reshape(-1, 1)).ravel()
    return H_p, AR_p, D_p


def predict_with_uncertainty(X_raw: np.ndarray):
    """
    Point prediction + GPR posterior standard deviations.

    MultiOutputRegressor does not expose return_std at wrapper
    level, so inner estimators are accessed directly.
      estimators_[0] → Height GPR
      estimators_[1] → Aspect Ratio GPR
      estimators_[2] → Dilution % GPR

    σ_orig = σ_scaled × scaler.scale_   (affine back-transform)

    Returns
    -------
    H_p, AR_p, D_p : point predictions (original units)
    σ_H, σ_AR, σ_D : posterior std devs (original units)
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
            sH  * scaler_H.scale_[0],   # σ_H  µm
            sAR * scaler_A.scale_[0],   # σ_AR
            sD  * scaler_D.scale_[0])   # σ_D  %


def composite_scalar(X_raw: np.ndarray) -> np.ndarray:
    """
    Augmented Tchebycheff scalarisation (SSA objective).

      f(x) = max(dH, dAR, dD) + ρ·(dH + dAR + dD)

    where dH = |H-H*|/σ_H,  dAR = |AR-AR*|/σ_AR,  dD = |D-D*|/σ_D

    The max() term ensures no objective is sacrificed entirely.
    ρ·(sum) is a small augmentation that keeps solutions off the
    degenerate edges of the Pareto surface.

    Infeasible dilution (D < D_MIN or D > D_MAX) receives a
    large additive penalty so SSA avoids those regions.
    """
    H_p, AR_p, D_p = predict_joint(X_raw)
    dH  = np.abs(H_p  - H_TARGET)  / SCALE_H
    dAR = np.abs(AR_p - AR_TARGET) / SCALE_AR
    dD  = np.abs(D_p  - D_TARGET)  / SCALE_D
    penalty = np.where((D_p < D_MIN) | (D_p > D_MAX), 10.0, 0.0)
    return np.maximum(np.maximum(dH, dAR), dD) + RHO * (dH + dAR + dD) + penalty

# ─────────────────────────────────────────────────────────────
# SECTION 4 — SPARROW SEARCH ALGORITHM (SSA)
# Pure NumPy — no external swarm library required.
# ─────────────────────────────────────────────────────────────

# ── SSA hyperparameters ──────────────────────────────────────
N_SPARROWS = 80     # Population size (literature: 50–120)
N_SSA_ITER = 150    # Iterations (literature: 100–200)
PD = 0.20           # Producer fraction (0.1–0.3)
SD = 0.10           # Vigilance-scout fraction (0.05–0.2)
ST = 0.80           # Safety threshold (0.5–1.0)
                    # R2 < ST → Lévy-style wide step
                    # R2 ≥ ST → Gaussian escape jump


def ssa_optimise():
    """
    Run SSA on the scalarised 3-objective Tchebycheff function.

    Initialisation: Latin Hypercube Sampling (qmc) for better
    space-filling than pure random — stratified coverage of the
    3D [P, S, F] design space at no additional cost.

    Returns
    -------
    population : (N_SPARROWS, 3) — final sparrow positions
    best_pos   : (3,)            — global best input
    best_fit   : float           — global best composite score
    history    : list[float]     — best score per iteration
    """
    # ── LHS initialisation ───────────────────────────────────
    sampler = qmc.LatinHypercube(d=N_VARS, seed=42)
    pop     = qmc.scale(sampler.random(n=N_SPARROWS),
                        BOUNDS_LOWER, BOUNDS_UPPER)

    fitness  = composite_scalar(pop)
    n_prod   = max(1, int(N_SPARROWS * PD))   # number of producers
    n_scout  = max(1, int(N_SPARROWS * SD))   # number of vigilance scouts

    best_idx = int(np.argmin(fitness))
    best_pos = pop[best_idx].copy()
    best_fit = float(fitness[best_idx])
    history  = [best_fit]

    for t in range(1, N_SSA_ITER + 1):

        # Sort ascending so index 0 = best, -1 = worst
        order     = np.argsort(fitness)
        pop       = pop[order]
        fitness   = fitness[order]
        worst_pos = pop[-1].copy()
        R2        = np.random.rand()    # per-iteration alarm value

        # ── ROLE 1: PRODUCERS — aggressive exploration ────────
        # R2 < ST (safe): exponential decay step
        #   → large displacement early, fine-tuning late
        # R2 ≥ ST (alarm): Gaussian escape to new region
        for i in range(n_prod):
            if R2 < ST:
                alpha  = np.random.rand()
                pop[i] = pop[i] * np.exp(
                    -i / (alpha * N_SSA_ITER + 1e-9))
            else:
                pop[i] = pop[i] + np.random.randn(N_VARS)

        # ── ROLE 2: SCROUNGERS — exploitation ────────────────
        # i > N/2 → hunger-driven: scramble away from worst
        # i ≤ N/2 → follower: move toward best producer
        for i in range(n_prod, N_SPARROWS):
            A     = np.random.choice([-1, 1], size=(1, N_VARS))
            A_pos = A.T / N_VARS   # pseudo-inverse (N_VARS,1)
            if i > N_SPARROWS // 2:
                Q      = np.random.randn(N_VARS)─────────────────────────────────────────────────────────
#
                pop[i] = Q * np.exp(
                    (worst_pos - pop[i]) / (i ** 2 + 1e-9))
            else:
                diff   = np.abs(pop[i] - pop[0])
                pop[i] = pop[0] + diff * A_pos.ravel()

        # ── ROLE 3: VIGILANCE — anti-predator escape ──────────
        # If not at global best → perturb toward best_pos
        # If at global best edge → gradient-like escape
        scout_idx = np.random.choice(N_SPARROWS, n_scout, replace=False)
        f_best    = float(fitness[0])
        f_worst   = float(fitness[-1])
        for idx in scout_idx:
            beta = np.random.randn(N_VARS)
            K    = np.random.uniform(-1, 1)
            if fitness[idx] > f_best:
                pop[idx] = best_pos + beta * np.abs(pop[idx] - best_pos)
            else:
                denom    = fitness[idx] - f_worst + 1e-9
                pop[idx] = pop[idx] + K * (
                    np.abs(pop[idx] - worst_pos) / denom)

        # Clip to physical machine bounds
        pop     = np.clip(pop, BOUNDS_LOWER, BOUNDS_UPPER)
        fitness = composite_scalar(pop)

        new_best = int(np.argmin(fitness))
        if fitness[new_best] < best_fit:
            best_fit = float(fitness[new_best])
            best_pos = pop[new_best].copy()
        history.append(best_fit)

        if t % 25 == 0 or t == 1:
            print(f"  SSA  iter {t:>4}/{N_SSA_ITER}  "
                  f"f={best_fit:.5f}  "
                  f"[P={best_pos[0]:.0f}W  "
                  f"S={best_pos[1]:.3f}mm/s  "
                  f"F={best_pos[2]:.2f}g/min]")

    return pop, best_pos, best_fit, history

# ─────────────────────────────────────────────────────────────
# SECTION 5 — CUSTOM PYMOO SAMPLING (SSA warm-start)
# ─────────────────────────────────────────────────────────────

# ── NSGA-II hyperparameters ──────────────────────────────────
POP_SIZE        = 150   # Population size (100–200 for 3 objectives)
N_GEN           = 400   # Generations (300–500 for convergence)
WARM_START_FRAC = 0.60  # Fraction of initial pop from SSA elites
                        # 0 → pure random baseline; 1 → all SSA
                        # 0.6 balances exploitation + diversity


class SSAWarmStartSampling(Sampling):
    """
    Seeds NSGA-II initial population with:
      • WARM_START_FRAC × POP_SIZE best SSA solutions  (exploitation)
      • (1 - WARM_START_FRAC) × POP_SIZE LHS points   (diversity)

    Using LHS for the random fill (instead of pure uniform)
    preserves even coverage of unvisited design-space regions.
    """

    def __init__(self, ssa_population: np.ndarray):
        super().__init__()
        self.ssa_pop = ssa_population

    def _do(self, problem, n_samples, **kwargs):
        n_warm   = min(int(n_samples * WARM_START_FRAC), len(self.ssa_pop))
        n_random = n_samples - n_warm

        # Select top-n_warm SSA solutions by composite scalar
        fit      = composite_scalar(self.ssa_pop)
        top_idx  = np.argsort(fit)[:n_warm]
        warm_pts = self.ssa_pop[top_idx]

        # LHS random fill for global diversity
        sampler  = qmc.LatinHypercube(d=N_VARS, seed=99)
        rand_pts = qmc.scale(sampler.random(n=n_random),
                             BOUNDS_LOWER, BOUNDS_UPPER)

        return np.vstack([warm_pts, rand_pts])   # (n_samples, 3)

# ─────────────────────────────────────────────────────────────
# SECTION 6 — PYMOO PROBLEM: 3-OBJECTIVE + 2 CONSTRAINTS
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
    f[0] : |H_pred  − H_target|    (µm)
    f[1] : |AR_pred − AR_target|   (dimensionless)
    f[2] : |D_pred  − D_target|    (%)

    Constraints  (g ≤ 0 convention)
    --------------------------------
    g[0] : D_MIN − D_pred ≤ 0   →  D_pred ≥ D_MIN (15%)
    g[1] : D_pred − D_MAX ≤ 0   →  D_pred ≤ D_MAX (45%)

    Surrogate: Multi-Output GPR (3-output, N=89)
      Height   R² = 0.9552 (single) / ≈0.97 (multi)
      AR       R² = 0.8739 (single) / ≈0.97 (multi)
      Dilution R² = 0.9023 (single) / joint model
    """

    def __init__(self):
        super().__init__(
            n_var    = N_VARS,
            n_obj    = 3,   # Height error | AR deviation | Dilution error
            n_constr = 2,   # Dilution lower bound | upper bound
            xl       = BOUNDS_LOWER.copy(),
            xu       = BOUNDS_UPPER.copy(),
        )

    def _evaluate(self, X, out, *args, **kwargs):
        H_p, AR_p, D_p = predict_joint(X)

        # ── Objectives ───────────────────────────────────────
        out["F"] = np.column_stack([
            np.abs(H_p  - H_TARGET),    # f0: height error  µm
            np.abs(AR_p - AR_TARGET),   # f1: AR deviation
            np.abs(D_p  - D_TARGET),    # f2: dilution error %
        ])

        # ── Constraints (g ≤ 0) ─────────────────────────────
        out["G"] = np.column_stack([
            D_MIN - D_p,   # g0 ≤ 0  →  D_pred ≥ 15%
            D_p - D_MAX,   # g1 ≤ 0  →  D_pred ≤ 45%
        ])

# ─────────────────────────────────────────────────────────────
# SECTION 7 — STAGE 1: SSA PRE-SEARCH
# ─────────────────────────────────────────────────────────────

print("\n" + "=" * 65)
print("STEP 2 — Stage 1: SSA Pre-search  (3-objective Tchebycheff)")
print(f"  Sparrows={N_SPARROWS}  Iterations={N_SSA_ITER}  "
      f"PD={PD:.0%}  SD={SD:.0%}  ST={ST}")
print("=" * 65)

t0 = time.time()
ssa_pop, ssa_best, ssa_fit, ssa_hist = ssa_optimise()
t_ssa = time.time() - t0

H_s, AR_s, D_s = predict_joint(ssa_best.reshape(1, -1))
feas_ssa        = D_MIN <= D_s[0] <= D_MAX
print(f"\n  ✓ SSA done in {t_ssa:.1f}s | best_f = {ssa_fit:.5f}")
print(f"    P={ssa_best[0]:.1f}W  S={ssa_best[1]:.4f}mm/s  "
      f"F={ssa_best[2]:.3f}g/min")
print(f"    H={H_s[0]:.1f}µm  AR={AR_s[0]:.4f}  D={D_s[0]:.2f}%  "
      f"{'✅ feasible' if feas_ssa else '❌ infeasible'}")

# ─────────────────────────────────────────────────────────────
# SECTION 8 — STAGE 2: NSGA-II WITH SSA WARM-START
# ─────────────────────────────────────────────────────────────

print("\n" + "=" * 65)
print("STEP 3 — Stage 2: NSGA-II (SSA warm-started, 3 objectives)")
print(f"  Population={POP_SIZE}  Generations={N_GEN}  "
      f"Warm-start={WARM_START_FRAC:.0%} from SSA elites")
print(f"  Crossover : SBX (η=15, prob=0.9)")
print(f"  Mutation  : PM  (η=20, prob=1/{N_VARS})")
print(f"  Constraint: {D_MIN}% ≤ D_pred ≤ {D_MAX}%  (hard)")
print("=" * 65)

problem   = LMDThreeObjective()
sampling  = SSAWarmStartSampling(ssa_pop)
algorithm = NSGA2(
    pop_size            = POP_SIZE,
    sampling            = sampling,
    crossover           = SBX(eta=15, prob=0.9),
    mutation            = PM(eta=20, prob=1.0 / N_VARS),
    eliminate_duplicates= True,
)
termination = get_termination("n_gen", N_GEN)

t0 = time.time()
result = minimize(
    problem,
    algorithm,
    termination,
    seed         = 42,
    save_history = True,
    verbose      = True,
)
t_nsga = time.time() - t0

print(f"\n  ✓ NSGA-II done in {t_nsga:.1f}s")
print(f"    Pareto-optimal solutions: {len(result.X)}")

# ─────────────────────────────────────────────────────────────
# SECTION 9 — ENRICH PARETO FRONT
# Add GPR uncertainty, feasibility flag, composite score
# ─────────────────────────────────────────────────────────────

pareto_X = result.X   # (n_pareto, 3) — decision variables
pareto_F = result.F   # (n_pareto, 3) — objective values

# Full uncertainty + prediction for every Pareto solution
H_p, AR_p, D_p, sig_H, sig_AR, sig_D = \
    predict_with_uncertainty(pareto_X)

# Feasibility: True if dilution inside metallurgical bounds
feasible = (D_p >= D_MIN) & (D_p <= D_MAX)
n_feas   = int(feasible.sum())

# Composite Tchebycheff score (lower = better balance)
comp_f = (np.maximum(np.maximum(pareto_F[:, 0] / SCALE_H,
                                pareto_F[:, 1] / SCALE_AR),
                     pareto_F[:, 2] / SCALE_D)
          + RHO * (pareto_F[:, 0] / SCALE_H
                   + pareto_F[:, 1] / SCALE_AR
                   + pareto_F[:, 2] / SCALE_D))

# Uncertainty score: sum of normalised σ values
unc_score = sig_H / SCALE_H + sig_AR / SCALE_AR + sig_D / SCALE_D

pareto_df = pd.DataFrame({
    "Power_W"          : pareto_X[:, 0],
    "Speed_mm_s"       : pareto_X[:, 1],
    "Feed_g_min"       : pareto_X[:, 2],
    "H_pred_um"        : H_p,
    "AR_pred"          : AR_p,
    "D_pred_pct"       : D_p,
    "Height_Err_um"    : pareto_F[:, 0],
    "AR_Dev"           : pareto_F[:, 1],
    "Dilution_Err_pct" : pareto_F[:, 2],
    "Sigma_H_um"       : sig_H,
    "Sigma_AR"         : sig_AR,
    "Sigma_D_pct"      : sig_D,
    "Feasible_D"       : feasible,
    "Uncertainty_Score": unc_score,
    "Composite_f"      : comp_f,
})
pareto_df.sort_values("Height_Err_um", inplace=True)
pareto_df.reset_index(drop=True, inplace=True)

print(f"\n  Dilution-feasible solutions: "
      f"{n_feas}/{len(pareto_df)} "
      f"({100*n_feas/max(len(pareto_df),1):.0f}%)")

# ─────────────────────────────────────────────────────────────
# SECTION 10 — KNEE-POINT AND CANDIDATE SELECTION
# ─────────────────────────────────────────────────────────────

weights = np.array([1/3, 1/3, 1/3])   # equal weight on all 3 objectives
decomp  = ASF()

# Prefer knee from feasible subset; fall back to full front
if n_feas > 0:
    feas_mask  = pareto_df["Feasible_D"].values.astype(bool)
    feas_F     = pareto_F[feas_mask]
    feas_idx   = np.where(feas_mask)[0]
    knee_local = decomp.do(feas_F, 1.0 / weights).argmin()
    knee_idx   = int(feas_idx[knee_local])
    print("  Knee selected from feasible-dilution subset ✅")
else:
    knee_idx   = int(decomp.do(pareto_F, 1.0 / weights).argmin())
    print("  ⚠️  No feasible solutions — knee from full Pareto front")

knee_params = pareto_X[knee_idx]
knee_objs   = pareto_F[knee_idx]
H_k, AR_k, D_k, sH_k, sAR_k, sD_k = \
    predict_with_uncertainty(knee_params.reshape(1, -1))

print("\n" + "=" * 65)
print("STEP 4 — Knee-Point (Best Balanced Trade-Off)")
print("=" * 65)
print(f"  Laser Power    : {knee_params[0]:.1f} W")
print(f"  Scan Speed     : {knee_params[1]:.4f} mm/s")
print(f"  Powder Feed    : {knee_params[2]:.3f} g/min")
print(f"  Pred. Height   : {H_k[0]:.1f} µm   "
      f"(target {H_TARGET:.0f} µm,  err {knee_objs[0]:.1f} µm)")
print(f"  Pred. AR       : {AR_k[0]:.4f}      "
      f"(target {AR_TARGET:.3f},    dev {knee_objs[1]:.4f})")
print(f"  Pred. Dilution : {D_k[0]:.2f}%      "
      f"(target {D_TARGET:.1f}%,    err {knee_objs[2]:.2f}%)")
print(f"  GPR σ          : ±{sH_k[0]:.1f} µm | "
      f"±{sAR_k[0]:.4f} | ±{sD_k[0]:.2f}%")
feas_k = D_MIN <= D_k[0] <= D_MAX
print(f"  Dilution feasible : {'✅' if feas_k else '❌'}")

# ── Build named candidate set for experimental validation ────
candidates = {}
candidates["Knee (balanced)"]     = pareto_df.loc[knee_idx]
candidates["Min Height Error"]    = pareto_df.loc[
    pareto_df["Height_Err_um"].idxmin()]
candidates["Min AR Deviation"]    = pareto_df.loc[
    pareto_df["AR_Dev"].idxmin()]
candidates["Min Dilution Error"]  = pareto_df.loc[
    pareto_df["Dilution_Err_pct"].idxmin()]
candidates["Min Uncertainty"]     = pareto_df.loc[
    pareto_df["Uncertainty_Score"].idxmin()]
if n_feas > 0:
    feas_df = pareto_df[pareto_df["Feasible_D"]]
    candidates["Best Feasible (D)"] = feas_df.loc[
        feas_df["Composite_f"].idxmin()]

print("\n" + "=" * 65)
print("STEP 5 — Recommended Experimental Candidates")
print("=" * 65)
hdr = (f"{'Label':<24} {'P(W)':>7} {'S(mm/s)':>8} {'F(g/m)':>7} "
       f"{'H(µm)':>8} {'AR':>7} {'D(%)':>7} "
       f"{'σH':>6} {'σAR':>6} {'σD':>5} {'Feas':>5}")
print(hdr)
print("─" * len(hdr))
for lbl, row in candidates.items():
    fstr = "✅" if row["Feasible_D"] else "❌"
    print(f"{lbl:<24} {row.Power_W:>7.1f} {row.Speed_mm_s:>8.4f} "
          f"{row.Feed_g_min:>7.3f} {row.H_pred_um:>8.1f} "
          f"{row.AR_pred:>7.4f} {row.D_pred_pct:>7.2f} "
          f"{row.Sigma_H_um:>6.1f} {row.Sigma_AR:>6.4f} "
          f"{row.Sigma_D_pct:>5.2f} {fstr:>5}")

# ─────────────────────────────────────────────────────────────
# SECTION 11 — HYPERVOLUME CONVERGENCE HISTORY
# ─────────────────────────────────────────────────────────────
# Track HV per generation for convergence diagnosis plot.
# Nadir = worst Pareto values + 10% margin.

nadir   = pareto_F.max(axis=0) * 1.10
hv_ind  = HV(ref_point=nadir)
hv_hist = []
if result.history:
    for gen in result.history:
        F_gen = gen.opt.get("F")
        # Only include feasible (non-violated) solutions if possible
        if gen.opt.get("feasible") is not None:
            feas_mask_gen = gen.opt.get("feasible").ravel()
            F_gen = F_gen[feas_mask_gen] if feas_mask_gen.any() else F_gen
        try:
            hv_hist.append(float(hv_ind(F_gen)))
        except Exception:
            hv_hist.append(np.nan)

# ─────────────────────────────────────────────────────────────
# SECTION 12 — FIGURES (publication quality, 600 dpi)
# ─────────────────────────────────────────────────────────────

print("\n" + "=" * 65)
print("STEP 6 — Generating publication-quality figures")
print("=" * 65)

unc_vals  = pareto_df["Uncertainty_Score"].values
feas_vals = pareto_df["Feasible_D"].values.astype(float)

def _mark_knee(ax, xv, yv, zorder=8):
    """Reusable knee-point marker for scatter panels."""
    ax.scatter(xv, yv, c=C_BLUE, s=260, zorder=zorder,
               marker='*', edgecolors='navy', linewidths=0.8,
               label='Knee point', clip_on=False)

# ── FIGURE 1: Three Pareto projections (2-D slices) ──────────
# Published as main optimisation result figure
fig1, axes1 = plt.subplots(1, 3, figsize=(16, 5))
fig1.suptitle(
    "Hybrid SSA–NSGA-II: Three-Objective Pareto Front\n"
    f"Surrogate: 3-output GPR (N=89)  |  "
    f"Targets: H={H_TARGET:.0f} µm, AR={AR_TARGET:.2f}, D={D_TARGET:.0f}%",
    fontsize=12, fontweight='bold', y=1.02
)

panel_configs = [
    # (x_data,         y_data,         colour,    xlabel,              ylabel,            title)
    (pareto_F[:, 0],  pareto_F[:, 1],  unc_vals,  'Height Error (µm)', 'AR Deviation',    'f₀ vs f₁'),
    (pareto_F[:, 0],  pareto_F[:, 2],  feas_vals, 'Height Error (µm)', 'Dilution Error (%)', 'f₀ vs f₂'),
    (pareto_F[:, 1],  pareto_F[:, 2],  unc_vals,  'AR Deviation',      'Dilution Error (%)', 'f₁ vs f₂'),
]
cmaps = ['RdYlGn_r', 'RdYlGn', 'RdYlGn_r']
clbls = ['Uncertainty Score', 'Dilution Feasible\n(1=✅  0=❌)', 'Uncertainty Score']

for ax, (xd, yd, cd, xl, yl, ttl), cm, cl in \
        zip(axes1, panel_configs, cmaps, clbls):
    sc = ax.scatter(xd, yd, c=cd, cmap=cm,
                    s=50, alpha=0.80, edgecolors='none', zorder=3)
    cb = plt.colorbar(sc, ax=ax, pad=0.02)
    cb.set_label(cl, fontsize=8)
    _mark_knee(ax,
               xd[knee_idx] if len(xd) > knee_idx else xd[-1],
               yd[knee_idx] if len(yd) > knee_idx else yd[-1])
    ax.set_xlabel(xl)
    ax.set_ylabel(yl)
    ax.set_title(ttl)
    ax.legend(fontsize=8, loc='upper right')

plt.tight_layout()
fig1.savefig("fig_pareto_projections.png", bbox_inches='tight')
plt.close(fig1)
print("  ✓ Saved: fig_pareto_projections.png")

# ── FIGURE 2: 3-D Pareto surface ─────────────────────────────
fig2 = plt.figure(figsize=(9, 7))
ax3d = fig2.add_subplot(111, projection='3d')

sc3d = ax3d.scatter(
    pareto_F[:, 0], pareto_F[:, 1], pareto_F[:, 2],
    c=unc_vals, cmap='RdYlGn_r',
    s=55, alpha=0.80, edgecolors='none'
)
# Knee point
ax3d.scatter(
    [knee_objs[0]], [knee_objs[1]], [knee_objs[2]],
    c=C_BLUE, s=280, marker='*', edgecolors='navy',
    linewidths=0.8, zorder=10, label='Knee point'
)
cb3d = fig2.colorbar(sc3d, ax=ax3d, pad=0.12, shrink=0.6)
cb3d.set_label('Uncertainty Score\n(green = reliable)', fontsize=9)
ax3d.set_xlabel('Height Error (µm)', labelpad=8)
ax3d.set_ylabel('AR Deviation',       labelpad=8)
ax3d.set_zlabel('Dilution Error (%)', labelpad=8)
ax3d.set_title(
    'Three-Objective Pareto Surface\n'
    'SSA–NSGA-II | 3-output GPR Surrogate',
    fontweight='bold'
)
ax3d.legend(fontsize=9)
ax3d.view_init(elev=25, azim=135)
fig2.tight_layout()
fig2.savefig("fig_pareto_3d_surface.png", bbox_inches='tight')
plt.close(fig2)
print("  ✓ Saved: fig_pareto_3d_surface.png")

# ── FIGURE 3: Convergence (SSA + NSGA-II + Hypervolume) ──────
fig3, axes3 = plt.subplots(1, 3, figsize=(16, 5))
fig3.suptitle(
    "Algorithm Convergence Diagnostics",
    fontsize=12, fontweight='bold'
)

# Panel A — SSA convergence curve
ax = axes3[0]
ax.plot(range(len(ssa_hist)), ssa_hist, color=C_BLUE, lw=2,
        label='SSA best f')
ax.axhline(ssa_fit, color=C_RED, ls='--', lw=1.5,
           label=f'Final  f = {ssa_fit:.4f}')
ax.set_xlabel('SSA Iteration')
ax.set_ylabel('Composite Objective f')
ax.set_title(f'Stage 1: SSA Convergence\n'
             f'({N_SPARROWS} sparrows, LHS init)')
ax.legend()

# Panel B — NSGA-II per-generation best objectives
ax = axes3[1]
if result.history:
    gen_h  = [np.min(g.opt.get("F")[:, 0]) for g in result.history]
    gen_ar = [np.min(g.opt.get("F")[:, 1]) for g in result.history]
    gen_d  = [np.min(g.opt.get("F")[:, 2]) for g in result.history]
    gens   = range(1, len(gen_h) + 1)
    ax.plot(gens, gen_h,  color=C_BLUE,   lw=1.8, label='Min Height Err (µm)')
    ax.plot(gens, gen_ar, color=C_RED,    lw=1.8, label='Min AR Dev')
    ax.plot(gens, gen_d,  color=C_GREEN,  lw=1.8, label='Min Dilution Err (%)')
ax.set_xlabel('NSGA-II Generation')
ax.set_ylabel('Best Objective Value')
ax.set_title(f'Stage 2: NSGA-II Convergence\n'
             f'({POP_SIZE} pop, {N_GEN} gen, '
             f'{WARM_START_FRAC:.0%} SSA seed)')
ax.legend()

# Panel C — Hypervolume per generation
ax = axes3[2]
if hv_hist:
    ax.plot(range(1, len(hv_hist) + 1), hv_hist,
            color=C_PURPLE, lw=2, label='Hypervolume')
    ax.set_xlabel('NSGA-II Generation')
    ax.set_ylabel('Hypervolume Indicator')
    ax.set_title('Pareto Front Quality\n'
                 '(Hypervolume — higher is better)')
    ax.legend()

plt.tight_layout()
fig3.savefig("fig_convergence_diagnostics.png", bbox_inches='tight')
plt.close(fig3)
print("  ✓ Saved: fig_convergence_diagnostics.png")

# ── FIGURE 4: Parameter space plots ──────────────────────────
fig4, axes4 = plt.subplots(1, 3, figsize=(16, 5))
fig4.suptitle(
    "Pareto Solutions in LMD Parameter Space",
    fontsize=12, fontweight='bold'
)

param_panels = [
    # (x_col,     y_col,       c_col,           xlabel,              ylabel,           title)
    ("Power_W",  "Feed_g_min",  "Height_Err_um",  "Laser Power (W)",   "Feed (g/min)",   "Power vs Feed"),
    ("Power_W",  "Speed_mm_s",  "D_pred_pct",     "Laser Power (W)",   "Speed (mm/s)",   "Power vs Speed\n(colour = Dilution %)"),
    ("Speed_mm_s","Feed_g_min", "AR_pred",         "Speed (mm/s)",      "Feed (g/min)",   "Speed vs Feed\n(colour = Pred. AR)"),
]
cmaps4 = ['coolwarm', 'RdYlGn', 'viridis']
clbls4 = ['Height Error (µm)', 'Pred. Dilution (%)', 'Pred. Aspect Ratio']

for ax, (xc, yc, cc, xl, yl, ttl), cm, cl in \
        zip(axes4, param_panels, cmaps4, clbls4):
    sc = ax.scatter(
        pareto_df[xc], pareto_df[yc], c=pareto_df[cc],
        cmap=cm, s=50, alpha=0.80, edgecolors='none'
    )
    plt.colorbar(sc, ax=ax, pad=0.02, label=cl)
    ax.scatter(knee_params[0] if xc == "Power_W" else
               (knee_params[1] if xc == "Speed_mm_s" else knee_params[1]),
               knee_params[2] if yc == "Feed_g_min" else knee_params[1],
               c=C_BLUE, s=230, marker='*', edgecolors='navy',
               linewidths=0.8, zorder=6, label='Knee point')
    ax.set_xlabel(xl)
    ax.set_ylabel(yl)
    ax.set_title(ttl)
    ax.legend(fontsize=8)

plt.tight_layout()
fig4.savefig("fig_parameter_space.png", bbox_inches='tight')
plt.close(fig4)
print("  ✓ Saved: fig_parameter_space.png")

# ── FIGURE 5: Dilution feasibility + GPR uncertainty ─────────
fig5, axes5 = plt.subplots(1, 2, figsize=(12, 5))
fig5.suptitle(
    "Dilution Feasibility and GPR Surrogate Uncertainty\nacross Pareto Solutions",
    fontsize=12, fontweight='bold'
)

# Panel A — Predicted dilution vs Power (feasibility bands)
ax = axes5[0]
clrs = [C_GREEN if f else C_RED for f in pareto_df["Feasible_D"]]
ax.scatter(pareto_df["Power_W"], pareto_df["D_pred_pct"],
           c=clrs, s=55, alpha=0.80, edgecolors='black',
           linewidths=0.3, zorder=3)
ax.axhspan(D_MIN, D_MAX, alpha=0.08, color=C_GREEN,
           label=f'Feasible zone [{D_MIN}%–{D_MAX}%]')
ax.axhline(D_TARGET, color=C_GREEN,  ls='-',  lw=1.8,
           label=f'Target D = {D_TARGET:.0f}%')
ax.axhline(D_MIN,    color=C_RED,    ls='--', lw=1.4,
           label=f'D_min = {D_MIN:.0f}%')
ax.axhline(D_MAX,    color=C_ORANGE, ls='--', lw=1.4,
           label=f'D_max = {D_MAX:.0f}%')
ax.scatter(knee_params[0], D_k[0],
           c=C_BLUE, s=260, marker='*', edgecolors='navy',
           linewidths=0.8, zorder=8, label='Knee point')
ax.set_xlabel('Laser Power (W)')
ax.set_ylabel('Predicted Dilution (%)')
ax.set_title('Dilution Feasibility Check\nGreen = ✅  Red = ❌')
ax.legend(fontsize=8)

# Panel B — GPR σ_H vs σ_D scatter
ax = axes5[1]
sc5 = ax.scatter(pareto_df["Sigma_H_um"], pareto_df["Sigma_D_pct"],
                 c=pareto_F[:, 0], cmap='viridis',
                 s=55, alpha=0.80, edgecolors='none')
plt.colorbar(sc5, ax=ax, label='Height Error (µm)')
ax.axvline(50,  color=C_RED,    ls='--', lw=1.4, alpha=0.8,
           label='σ_H = 50 µm threshold')
ax.axhline(2.0, color=C_ORANGE, ls='--', lw=1.4, alpha=0.8,
           label='σ_D = 2% threshold')
ax.scatter(sH_k[0], sD_k[0],
           c=C_BLUE, s=260, marker='*', edgecolors='navy',
           linewidths=0.8, zorder=8, label='Knee point')
ax.set_xlabel('GPR σ — Height (µm)')
ax.set_ylabel('GPR σ — Dilution (%)')
ax.set_title('Surrogate Uncertainty\n(bottom-left = most reliable)')
ax.legend(fontsize=8)

plt.tight_layout()
fig5.savefig("fig_feasibility_uncertainty.png", bbox_inches='tight')
plt.close(fig5)
print("  ✓ Saved: fig_feasibility_uncertainty.png")

# ─────────────────────────────────────────────────────────────
# SECTION 13 — SAVE ALL OUTPUTS TO CSV
# ─────────────────────────────────────────────────────────────

# Full Pareto front
pareto_df.to_csv("ssa_nsga2_pareto_full.csv", index=False)
print("  ✓ Saved: ssa_nsga2_pareto_full.csv")

# SSA convergence history
pd.DataFrame({
    "iteration": range(len(ssa_hist)),
    "best_f"   : ssa_hist,
}).to_csv("ssa_convergence_history.csv", index=False)
print("  ✓ Saved: ssa_convergence_history.csv")

# Hypervolume history
if hv_hist:
    pd.DataFrame({
        "generation": range(1, len(hv_hist) + 1),
        "hypervolume": hv_hist,
    }).to_csv("nsga2_hypervolume_history.csv", index=False)
    print("  ✓ Saved: nsga2_hypervolume_history.csv")

# Experimental candidates
cand_rows = []
for lbl, row in candidates.items():
    cand_rows.append({
        "Label"           : lbl,
        "Power_W"         : round(float(row["Power_W"]),          1),
        "Speed_mm_s"      : round(float(row["Speed_mm_s"]),        4),
        "Feed_g_min"      : round(float(row["Feed_g_min"]),        3),
        "H_pred_um"       : round(float(row["H_pred_um"]),         1),
        "AR_pred"         : round(float(row["AR_pred"]),           4),
        "D_pred_pct"      : round(float(row["D_pred_pct"]),        2),
        "Height_Err_um"   : round(float(row["Height_Err_um"]),     1),
        "AR_Dev"          : round(float(row["AR_Dev"]),            5),
        "Dilution_Err_pct": round(float(row["Dilution_Err_pct"]),  2),
        "Sigma_H_um"      : round(float(row["Sigma_H_um"]),        1),
        "Sigma_AR"        : round(float(row["Sigma_AR"]),          5),
        "Sigma_D_pct"     : round(float(row["Sigma_D_pct"]),       2),
        "Feasible_D"      : bool(row["Feasible_D"]),
        "Composite_f"     : round(float(row["Composite_f"]),       5),
        "Uncertainty_Score": round(float(row["Uncertainty_Score"]),5),
    })

cand_df = pd.DataFrame(cand_rows)
cand_df.to_csv("ssa_nsga2_validation_candidates.csv", index=False)
print("  ✓ Saved: ssa_nsga2_validation_candidates.csv")

# ─────────────────────────────────────────────────────────────
# SECTION 14 — FINAL SUMMARY
# ─────────────────────────────────────────────────────────────

print("\n" + "=" * 65)
print("PHASE 1B COMPLETE — Hybrid SSA-NSGA-II (3 objectives)")
print("=" * 65)
print(f"  SSA runtime        : {t_ssa:.1f}s")
print(f"  NSGA-II runtime    : {t_nsga:.1f}s")
print(f"  Total runtime      : {t_ssa + t_nsga:.1f}s")
print(f"  Pareto size        : {len(pareto_df)} solutions")
print(f"  Feasible (D)       : {n_feas}/{len(pareto_df)} "
      f"({100*n_feas/max(len(pareto_df),1):.0f}%)")
print(f"  Knee point         : "
      f"P={knee_params[0]:.1f}W  "
      f"S={knee_params[1]:.4f}mm/s  "
      f"F={knee_params[2]:.3f}g/min")
print(f"  Knee predictions   : "
      f"H={H_k[0]:.1f}µm  "
      f"AR={AR_k[0]:.4f}  "
      f"D={D_k[0]:.2f}%")
print(f"  Knee feasible (D)  : {'✅' if feas_k else '❌'}")
print(f"\n  Outputs:")
print(f"    fig_pareto_projections.png      — main result (3 × 2D slices)")
print(f"    fig_pareto_3d_surface.png       — 3D Pareto surface")
print(f"    fig_convergence_diagnostics.png — SSA + NSGA-II + HV curves")
print(f"    fig_parameter_space.png         — solutions in [P, S, F] space")
print(f"    fig_feasibility_uncertainty.png — dilution bounds + σ scatter")
print(f"    ssa_nsga2_pareto_full.csv")
print(f"    ssa_nsga2_validation_candidates.csv")
print(f"    ssa_convergence_history.csv")
print(f"    nsga2_hypervolume_history.csv")
print("=" * 65)
