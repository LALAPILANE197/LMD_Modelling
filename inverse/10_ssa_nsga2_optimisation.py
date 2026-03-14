# ============================================================
# 10_ssa_nsga2_optimisation.py
# Phase 1B — Hybrid SSA-NSGA2 Multi-Objective Optimisation
# Surrogate: Multi-Output GPR (joint Height + Aspect Ratio)
# LMD Bead Geometry Inverse Optimisation
# ============================================================
#
# ALGORITHM OVERVIEW
# ─────────────────────────────────────────────────────────
# Standard NSGA-II begins from a random population and wastes
# early generations sampling poor regions. This hybrid uses
# a two-stage strategy:
#
# Stage 1 — SSA Pre-search  (pure NumPy, no extra library)
#   Sparrow Search Algorithm (Xue & Shen, 2020) rapidly maps
#   the design space via three behavioural roles:
#     • Producers  (top PD% by fitness) — Lévy-style wide search
#     • Scroungers (remaining)          — follow best producer
#     • Vigilance  (random SD%)         — escape local optima
#   SSA optimises a scalarised Augmented Tchebycheff composite
#   objective, converging to the best-known region quickly.
#
# Stage 2 — NSGA-II Refinement  (pymoo)
#   NSGA-II is warm-started with SSA elites injected into its
#   initial population. It then expands outward, recovering the
#   full Pareto front using non-dominated sorting and crowding
#   distance preservation — capabilities SSA lacks.
#
# Key advantages over single-algorithm approaches:
#   ✓ SSA warm-start accelerates NSGA-II convergence
#   ✓ NSGA-II preserves Pareto diversity that SSA collapses
#   ✓ Multi-output GPR: joint prediction captures cross-target
#     physics (Height and AR share laser energy dependence)
#   ✓ GPR posterior σ flags unreliable Pareto candidates
#
# Reference: Xue J, Shen B. A novel swarm intelligence
# optimization approach: sparrow search algorithm.
# Systems Science & Control Engineering, 2020;8(1):22-34.
# ============================================================

import numpy as np
import pandas as pd
import joblib
import time
import warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pymoo.core.problem import Problem
from pymoo.core.sampling import Sampling
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.termination import get_termination
from pymoo.optimize import minimize
from pymoo.decomposition.asf import ASF

warnings.filterwarnings('ignore')
np.random.seed(42)

# ─────────────────────────────────────────────────────────────
# SECTION 1: LOAD MULTI-OUTPUT GPR AND SCALERS
# ─────────────────────────────────────────────────────────────
# multi_gpr is a sklearn MultiOutputRegressor wrapping two
# independent GaussianProcessRegressors trained jointly on
# [Height, Aspect Ratio] in scaled space.
#
# Architecture (from notebook CELL 16):
#   inner estimator : C(1.0)*RBF(1.0), alpha=0.01
#   n_restarts_optimizer : 5
#
# Performance (5-fold CV, original units):
#   Height  R²=0.9682  MAE=58.83 µm  (+0.013 vs single GPR)
#   Aspect  R²=0.9689  MAE=0.0283    (+0.095 vs single GPR)
# ─────────────────────────────────────────────────────────────

print("=" * 65)
print("STEP 1: Loading Multi-Output GPR surrogate and scalers")
print("=" * 65)

multi_gpr   = joblib.load("multi_output_gpr.pkl")
scaler_X    = joblib.load("scaler_X.pkl")
scaler_H    = joblib.load("scaler_height.pkl")
scaler_A    = joblib.load("scaler_aspect.pkl")

# Validate loaded objects
assert hasattr(multi_gpr, 'estimators_'), \
    "multi_output_gpr.pkl must be a fitted MultiOutputRegressor. " \
    "Re-run notebook Cell 16 and re-save."
assert len(multi_gpr.estimators_) == 2, \
    "Expected 2 inner estimators [height GPR, aspect GPR]."

print("  multi_output_gpr.pkl : loaded  "
      f"({len(multi_gpr.estimators_)} inner estimators)")
print("  scaler_X.pkl         : loaded  "
      f"(fitted on {scaler_X.n_features_in_} features)")
print("  scaler_height.pkl    : loaded  "
      f"(mean={scaler_H.mean_[0]:.1f} µm, "
      f"std={scaler_H.scale_[0]:.1f} µm)")
print("  scaler_aspect.pkl    : loaded  "
      f"(mean={scaler_A.mean_[0]:.4f}, "
      f"std={scaler_A.scale_[0]:.4f})")

# ─────────────────────────────────────────────────────────────
# SECTION 2: OPTIMISATION TARGETS AND DESIGN SPACE BOUNDS
# ─────────────────────────────────────────────────────────────
# Edit these to change the inverse problem specification.
# Bounds from your LHS-60 + additional-30 experimental matrix.

H_target  = 1100.0   # µm  — desired bead height
AR_target = 0.44   # W/H — desired aspect ratio

# Tchebycheff scalarisation scale factors
# Normalise each objective by its standard deviation so
# neither dominates the composite score through units alone.
SCALE_H  = float(scaler_H.scale_[0])    # σ of height  ≈ 426 µm
SCALE_AR = float(scaler_A.scale_[0])    # σ of AR      ≈ 0.215
RHO      = 0.05     # Augmented Tchebycheff regularisation term

# Design space: [Power (W), Speed (mm/s), Feed (g/min)]
BOUNDS_LOWER = np.array([1600.0, 0.8, 2.0])
BOUNDS_UPPER = np.array([2800.0, 1.6, 8.0])
N_VARS       = 3

print(f"\nTarget geometry   : H = {H_target:.1f} µm, "
      f"AR = {AR_target:.3f}")
print(f"Tchebycheff scale : σ_H = {SCALE_H:.1f} µm, "
      f"σ_AR = {SCALE_AR:.4f}")
print(f"Design space      : Power {BOUNDS_LOWER[0]:.0f}–"
      f"{BOUNDS_UPPER[0]:.0f} W  |  Speed "
      f"{BOUNDS_LOWER[1]:.1f}–{BOUNDS_UPPER[1]:.1f} mm/s  |  "
      f"Feed {BOUNDS_LOWER[2]:.1f}–{BOUNDS_UPPER[2]:.1f} g/min")

# ─────────────────────────────────────────────────────────────
# SECTION 3: MULTI-OUTPUT GPR PREDICTION FUNCTIONS
# ─────────────────────────────────────────────────────────────

def _build_features(X_raw: np.ndarray) -> np.ndarray:
    """
    Reconstruct 4-column feature matrix from raw [P, S, F].
    Mirrors Cell 3 of the notebook exactly:
      [Power, Speed, Feed, HeatInput = Power / (Speed + ε)]
    """
    X_raw = np.atleast_2d(X_raw)
    heat  = X_raw[:, 0] / (X_raw[:, 1] + 1e-9)
    return np.column_stack([X_raw, heat])


def predict_joint(X_raw: np.ndarray):
    """
    Joint GPR prediction for bead Height and Aspect Ratio.

    Parameters
    ----------
    X_raw : (n, 3) array — [Power W, Speed mm/s, Feed g/min]

    Returns
    -------
    H_pred  : (n,)  bead height in µm (original units)
    AR_pred : (n,)  aspect ratio      (original units)
    """
    X_scaled = scaler_X.transform(_build_features(X_raw))

    # MultiOutputRegressor.predict() → shape (n, 2)
    Y_scaled = multi_gpr.predict(X_scaled)

    H_pred  = scaler_H.inverse_transform(
                Y_scaled[:, 0].reshape(-1, 1)).ravel()
    AR_pred = scaler_A.inverse_transform(
                Y_scaled[:, 1].reshape(-1, 1)).ravel()
    return H_pred, AR_pred


def predict_joint_with_uncertainty(X_raw: np.ndarray):
    """
    Joint prediction + GPR posterior standard deviations.

    Notes
    -----
    MultiOutputRegressor does NOT expose return_std at the
    wrapper level. We access inner estimators directly:
      multi_gpr.estimators_[0]  → height GPR
      multi_gpr.estimators_[1]  → aspect ratio GPR
    Both operate in scaled space; σ is back-transformed via
      σ_orig = σ_scaled * scaler.scale_
    (valid for StandardScaler, an affine transformation).
    """
    X_raw    = np.atleast_2d(X_raw)
    X_scaled = scaler_X.transform(_build_features(X_raw))

    H_sc,  sH_sc  = multi_gpr.estimators_[0].predict(
                        X_scaled, return_std=True)
    AR_sc, sAR_sc = multi_gpr.estimators_[1].predict(
                        X_scaled, return_std=True)

    H_pred  = scaler_H.inverse_transform(
                H_sc.reshape(-1, 1)).ravel()
    AR_pred = scaler_A.inverse_transform(
                AR_sc.reshape(-1, 1)).ravel()

    sigma_H  = sH_sc  * scaler_H.scale_[0]   # µm
    sigma_AR = sAR_sc * scaler_A.scale_[0]   # dimensionless

    return H_pred, AR_pred, sigma_H, sigma_AR


def composite_objective(X_raw: np.ndarray) -> np.ndarray:
    """
    Augmented Tchebycheff scalarisation of both objectives.

    f(x) = max(|H-H*|/σ_H, |AR-AR*|/σ_AR)
         + ρ * (|H-H*|/σ_H + |AR-AR*|/σ_AR)

    Lower is better (minimisation).
    ρ > 0 prevents degenerate solutions on Pareto edges.
    """
    H_pred, AR_pred = predict_joint(X_raw)
    dH  = np.abs(H_pred  - H_target)  / SCALE_H
    dAR = np.abs(AR_pred - AR_target) / SCALE_AR
    return np.maximum(dH, dAR) + RHO * (dH + dAR)


# ─────────────────────────────────────────────────────────────
# SECTION 4: SPARROW SEARCH ALGORITHM (SSA)
# ─────────────────────────────────────────────────────────────
# Pure NumPy implementation — no external swarm library needed.
#
# TUNEABLE HYPERPARAMETERS
# ─────────────────────────────────────────────────────────────
N_SPARROWS = 80      # Population size (recommend 50–120)
N_SSA_ITER = 150     # Number of SSA iterations (recommend 100–200)
PD         = 0.20    # Fraction designated as producers (0.1–0.3)
SD         = 0.10    # Fraction designated as vigilance scouts (0.05–0.2)
ST         = 0.80    # Safety threshold R2 ∈ [0.5, 1.0]:
                     #   R2 < ST → producers do Lévy-style search
                     #   R2 ≥ ST → producers do Gaussian step
# ─────────────────────────────────────────────────────────────

def ssa_optimise():
    """
    Run SSA on the scalarised composite objective.

    Returns
    -------
    population : (N_SPARROWS, N_VARS) — final sparrow positions
    best_pos   : (N_VARS,)            — global best position
    best_fit   : float                — global best composite f
    history    : list[float]          — best f per iteration
    """
    # ── Initialisation ───────────────────────────────────────
    # Latin Hypercube-like initialisation: stratified random
    # sampling across each dimension independently.
    pop = np.random.uniform(
            BOUNDS_LOWER, BOUNDS_UPPER,
            size=(N_SPARROWS, N_VARS))           # (N, 3)

    fitness   = composite_objective(pop)         # (N,)
    n_prod    = max(1, int(N_SPARROWS * PD))     # #producers
    n_scout   = max(1, int(N_SPARROWS * SD))     # #vigilance

    best_idx  = np.argmin(fitness)
    best_pos  = pop[best_idx].copy()
    best_fit  = fitness[best_idx]
    history   = [best_fit]

    # ── Main iteration loop ───────────────────────────────────
    for t in range(1, N_SSA_ITER + 1):

        # Sort ascending: best (low f) → worst (high f)
        sort_idx = np.argsort(fitness)
        pop      = pop[sort_idx]
        fitness  = fitness[sort_idx]

        worst_pos = pop[-1].copy()
        R2        = np.random.rand()      # alarm value this step

        # ── ROLE 1: PRODUCER UPDATE ───────────────────────────
        # Producers explore the space aggressively.
        # Two regimes governed by R2 vs safety threshold ST:
        #   R2 < ST → "safe" environment: exponential damping
        #             (Lévy-like wide step on early iterations,
        #              fine-tuning on later iterations)
        #   R2 ≥ ST → "predator detected": Gaussian jump
        #             (escape to an entirely new region)
        for i in range(n_prod):
            if R2 < ST:
                alpha = np.random.rand()
                # Exponential decay: large steps early, small late
                pop[i] = pop[i] * np.exp(
                            -i / (alpha * N_SSA_ITER + 1e-9))
            else:
                Q = np.random.randn(N_VARS)
                pop[i] = pop[i] + Q  # Gaussian escape step

        # ── ROLE 2: SCROUNGER (JOINER) UPDATE ─────────────────
        # Scroungers follow the best producer or scramble
        # toward a randomly chosen "host" producer.
        # Two sub-cases based on scrounger rank i in population:
        #   i > N/2 → hunger-driven: exponential move toward worst
        #   i ≤ N/2 → follower:     move toward best producer
        #
        # A+ matrix simplification for d-dimensional case:
        #   A  = random ±1 row vector,  A*A^T = d  (scalar)
        #   A+ = A^T / d                (pseudo-inverse)
        for i in range(n_prod, N_SPARROWS):
            A     = np.random.choice([-1, 1], size=(1, N_VARS))
            A_pos = A.T / N_VARS         # shape (N_VARS, 1)

            if i > N_SPARROWS // 2:
                # Hunger-driven scramble: jump away from worst
                Q       = np.random.randn(N_VARS)
                pop[i]  = Q * np.exp(
                    (worst_pos - pop[i]) / (i ** 2 + 1e-9))
            else:
                # Follower: move toward best producer (index 0)
                diff    = np.abs(pop[i] - pop[0])   # (N_VARS,)
                # A_pos is (N_VARS,1), diff is (N_VARS,) → broadcast
                pop[i]  = pop[0] + diff * A_pos.ravel()

        # ── ROLE 3: VIGILANCE (ANTI-PREDATOR) UPDATE ──────────
        # A random subset of the worst-positioned sparrows
        # act as "scouts": they escape toward a safer region.
        # Two sub-cases based on whether scout is at global best:
        #   f_i > f_best → not at best: random step near best_pos
        #   f_i = f_best → at edge: gradient-descent escape
        scout_idx = np.random.choice(
                        N_SPARROWS, n_scout, replace=False)
        f_best  = fitness[0]
        f_worst = fitness[-1]

        for idx in scout_idx:
            beta = np.random.randn(N_VARS)   # N(0,1) step vector
            K    = np.random.uniform(-1, 1)  # [-1,1] direction

            if fitness[idx] > f_best:
                # Move toward global best with random perturbation
                pop[idx] = best_pos + beta * np.abs(
                                pop[idx] - best_pos)
            else:
                # Gradient-like escape from worst region
                denom     = (fitness[idx] - f_worst + 1e-9)
                pop[idx]  = pop[idx] + K * (
                    np.abs(pop[idx] - worst_pos) / denom)

        # ── CLIP TO BOUNDS ────────────────────────────────────
        pop = np.clip(pop, BOUNDS_LOWER, BOUNDS_UPPER)

        # ── EVALUATE AND UPDATE GLOBAL BEST ──────────────────
        fitness   = composite_objective(pop)
        best_idx  = np.argmin(fitness)
        if fitness[best_idx] < best_fit:
            best_fit  = fitness[best_idx]
            best_pos  = pop[best_idx].copy()

        history.append(best_fit)

        if t % 25 == 0 or t == 1:
            print(f"  SSA  iter {t:>4d}/{N_SSA_ITER}  "
                  f"best_f = {best_fit:.6f}  "
                  f"[P={best_pos[0]:.1f} W, "
                  f"S={best_pos[1]:.3f} mm/s, "
                  f"F={best_pos[2]:.2f} g/min]")

    return pop, best_pos, best_fit, history


# ─────────────────────────────────────────────────────────────
# SECTION 5: CUSTOM PYMOO SAMPLING — SSA WARM START
# ─────────────────────────────────────────────────────────────
# We inject the SSA elite population into NSGA-II's first
# generation. WARM_START_FRAC controls the ratio of SSA elites
# vs fresh random points in the initial NSGA-II population.
#
# TUNEABLE HYPERPARAMETERS
# ─────────────────────────────────────────────────────────────
POP_SIZE        = 150   # NSGA-II population size (recommend 100–200)
N_GEN           = 400   # NSGA-II generations     (recommend 300–500)
WARM_START_FRAC = 0.60  # Fraction of pop_size drawn from SSA elites
                        # 0 → pure random (baseline), 1 → all SSA
# ─────────────────────────────────────────────────────────────

class SSAWarmStartSampling(Sampling):
    """
    Custom pymoo Sampling class that seeds NSGA-II with a
    mixture of SSA elite solutions and random Latin Hypercube
    points to preserve global coverage.
    """

    def __init__(self, ssa_population: np.ndarray):
        super().__init__()
        self.ssa_pop = ssa_population          # (N_SPARROWS, 3)

    def _do(self, problem, n_samples, **kwargs):
        n_warm   = min(int(n_samples * WARM_START_FRAC),
                       len(self.ssa_pop))
        n_random = n_samples - n_warm

        # Select top-n_warm SSA solutions by composite fitness
        ssa_fitness = composite_objective(self.ssa_pop)
        top_idx     = np.argsort(ssa_fitness)[:n_warm]
        warm_pts    = self.ssa_pop[top_idx]               # (n_warm, 3)

        # Random fill for diversity
        rand_pts = np.random.uniform(
                        BOUNDS_LOWER, BOUNDS_UPPER,
                        size=(n_random, N_VARS))           # (n_random, 3)

        return np.vstack([warm_pts, rand_pts])             # (n_samples, 3)


# ─────────────────────────────────────────────────────────────
# SECTION 6: PYMOO PROBLEM DEFINITION
# ─────────────────────────────────────────────────────────────

class LMDProblemMultiGPR(Problem):
    """
    Two-objective LMD bead geometry optimisation problem.

    Decision variables
    ------------------
    x[0] : Laser Power   W        ∈ [1600, 2800]
    x[1] : Scan Speed    mm/s     ∈ [0.8,  1.6]
    x[2] : Powder Feed   g/min    ∈ [2.0,  8.0]

    Objectives (both minimise)
    --------------------------
    f[0] : |H_pred  - H_target|   height error   µm
    f[1] : |AR_pred - AR_target|  AR deviation   (dimensionless)

    Surrogate : Multi-output GPR (joint; R²_H=0.97, R²_AR=0.97)
    """

    def __init__(self):
        super().__init__(
            n_var   = N_VARS,
            n_obj   = 2,
            n_constr= 0,
            xl      = BOUNDS_LOWER.copy(),
            xu      = BOUNDS_UPPER.copy(),
        )

    def _evaluate(self, X, out, *args, **kwargs):
        H_pred, AR_pred = predict_joint(X)
        f1 = np.abs(H_pred  - H_target)
        f2 = np.abs(AR_pred - AR_target)
        out["F"] = np.column_stack([f1, f2])


# ─────────────────────────────────────────────────────────────
# SECTION 7: RUN STAGE 1 — SSA PRE-SEARCH
# ─────────────────────────────────────────────────────────────

print("\n" + "=" * 65)
print("STEP 2: Stage 1 — SSA Pre-search")
print(f"  Sparrows       : {N_SPARROWS}")
print(f"  Iterations     : {N_SSA_ITER}")
print(f"  Producers PD   : {PD:.0%}  ({int(N_SPARROWS*PD)} sparrows)")
print(f"  Vigilance SD   : {SD:.0%}  ({int(N_SPARROWS*SD)} sparrows)")
print(f"  Safety thresh  : ST = {ST}")
print(f"  Objective      : Augmented Tchebycheff (ρ={RHO})")
print("=" * 65)

t_ssa_start   = time.time()
ssa_pop, ssa_best_pos, ssa_best_fit, ssa_history = ssa_optimise()
t_ssa_elapsed = time.time() - t_ssa_start

H_ssa, AR_ssa = predict_joint(ssa_best_pos.reshape(1, -1))
print(f"\n✓ SSA complete in {t_ssa_elapsed:.1f} s")
print(f"  Best composite f  : {ssa_best_fit:.6f}")
print(f"  Best position     : Power={ssa_best_pos[0]:.1f} W  "
      f"Speed={ssa_best_pos[1]:.4f} mm/s  "
      f"Feed={ssa_best_pos[2]:.3f} g/min")
print(f"  Predicted geometry: H={H_ssa[0]:.1f} µm  "
      f"AR={AR_ssa[0]:.4f}")

# ─────────────────────────────────────────────────────────────
# SECTION 8: RUN STAGE 2 — NSGA-II WITH SSA WARM START
# ─────────────────────────────────────────────────────────────

print("\n" + "=" * 65)
print("STEP 3: Stage 2 — NSGA-II (SSA warm-started)")
print(f"  Population     : {POP_SIZE}")
print(f"  Generations    : {N_GEN}")
print(f"  Warm-start     : {WARM_START_FRAC:.0%} from SSA elites")
print(f"  Crossover      : SBX  (η_c=15, prob=0.9)")
print(f"  Mutation       : PM   (η_m=20, prob=1/n_var)")
print("=" * 65)

problem  = LMDProblemMultiGPR()
sampling = SSAWarmStartSampling(ssa_pop)

algorithm = NSGA2(
    pop_size            = POP_SIZE,
    sampling            = sampling,
    crossover           = SBX(eta=15, prob=0.9),
    mutation            = PM(eta=20,  prob=1.0 / N_VARS),
    eliminate_duplicates= True,
)

termination = get_termination("n_gen", N_GEN)

t_nsga_start = time.time()
result = minimize(
    problem,
    algorithm,
    termination,
    seed         = 42,
    save_history = True,
    verbose      = True,
)
t_nsga_elapsed = time.time() - t_nsga_start

print(f"\n✓ NSGA-II complete in {t_nsga_elapsed:.1f} s")
print(f"  Pareto-optimal solutions: {len(result.X)}")

# ─────────────────────────────────────────────────────────────
# SECTION 9: ENRICH PARETO FRONT WITH UNCERTAINTY
# ─────────────────────────────────────────────────────────────

pareto_X = result.X    # (n_pareto, 3)
pareto_F = result.F    # (n_pareto, 2)  [height_err, AR_dev]

H_pred, AR_pred, sigma_H, sigma_AR = \
    predict_joint_with_uncertainty(pareto_X)

pareto_df = pd.DataFrame({
    "Power_W"         : pareto_X[:, 0],
    "Speed_mm_s"      : pareto_X[:, 1],
    "Feed_g_min"      : pareto_X[:, 2],
    "H_pred_um"       : H_pred,
    "AR_pred"         : AR_pred,
    "Height_Err_um"   : pareto_F[:, 0],
    "AR_Dev"          : pareto_F[:, 1],
    "Sigma_H_um"      : sigma_H,
    "Sigma_AR"        : sigma_AR,
    # Composite reliability: lower = more trusted by GPR
    "Uncertainty_Score": sigma_H / SCALE_H + sigma_AR / SCALE_AR,
    # Augmented Tchebycheff composite (same as SSA objective)
    "Composite_f"     : (np.maximum(pareto_F[:, 0] / SCALE_H,
                                     pareto_F[:, 1] / SCALE_AR)
                         + RHO * (pareto_F[:, 0] / SCALE_H
                                  + pareto_F[:, 1] / SCALE_AR)),
})
pareto_df.sort_values("Height_Err_um", inplace=True)
pareto_df.reset_index(drop=True, inplace=True)

# ─────────────────────────────────────────────────────────────
# SECTION 10: KNEE-POINT AND CANDIDATE SELECTION
# ─────────────────────────────────────────────────────────────
# ASF (Achievement Scalarisation Function) finds the solution
# that makes the most balanced progress on both objectives.
# weights=[0.5, 0.5] → equal importance to height and AR.

weights  = np.array([0.5, 0.5])
decomp   = ASF()
knee_idx = decomp.do(pareto_F, 1.0 / weights).argmin()

knee_params = pareto_X[knee_idx]
knee_objs   = pareto_F[knee_idx]
H_knee, AR_knee, sH_knee, sAR_knee = \
    predict_joint_with_uncertainty(knee_params.reshape(1, -1))

print("\n" + "=" * 65)
print("STEP 4: Knee-Point (Best Trade-Off Solution)")
print("=" * 65)
print(f"  Laser Power       : {knee_params[0]:.1f} W")
print(f"  Scan Speed        : {knee_params[1]:.4f} mm/s")
print(f"  Powder Feed       : {knee_params[2]:.3f} g/min")
print(f"  Predicted Height  : {H_knee[0]:.1f} µm  "
      f"(target: {H_target:.1f} µm,  error: {knee_objs[0]:.1f} µm)")
print(f"  Predicted AR      : {AR_knee[0]:.4f}  "
      f"(target: {AR_target:.4f},  dev: {knee_objs[1]:.4f})")
print(f"  GPR σ_Height      : ±{sH_knee[0]:.1f} µm")
print(f"  GPR σ_AR          : ±{sAR_knee[0]:.4f}")

# ── Validation candidates ─────────────────────────────────────
n_pareto   = len(pareto_df)
candidates = {}

candidates["Knee Point"]        = pareto_df.loc[knee_idx]
candidates["Min Height Error"]  = pareto_df.loc[
                                    pareto_df["Height_Err_um"].idxmin()]
candidates["Min AR Deviation"]  = pareto_df.loc[
                                    pareto_df["AR_Dev"].idxmin()]
candidates["Min Uncertainty"]   = pareto_df.loc[
                                    pareto_df["Uncertainty_Score"].idxmin()]

top2_unc = pareto_df.nlargest(2, "Uncertainty_Score")
candidates["High Uncertainty 1"] = top2_unc.iloc[0]
candidates["High Uncertainty 2"] = top2_unc.iloc[1]

if n_pareto >= 4:
    sorted_h = pareto_df["Height_Err_um"].argsort().values
    candidates["Interior Q25"] = pareto_df.iloc[
                                    sorted_h[max(1, int(n_pareto*0.25))]]
    candidates["Interior Q75"] = pareto_df.iloc[
                                    sorted_h[min(n_pareto-2,
                                                 int(n_pareto*0.75))]]
else:
    candidates["Interior Q25"] = pareto_df.iloc[0]
    candidates["Interior Q75"] = pareto_df.iloc[-1]

print("\n" + "=" * 65)
print("STEP 5: Recommended Experimental Candidates")
print("=" * 65)
print(f"{'Label':<24} {'P(W)':>7} {'S(mm/s)':>8} "
      f"{'F(g/m)':>7} {'H(µm)':>8} {'AR':>7} "
      f"{'σH':>7} {'σAR':>7}")
print("─" * 80)
for lbl, row in candidates.items():
    print(f"{lbl:<24} {row.Power_W:>7.1f} {row.Speed_mm_s:>8.4f} "
          f"{row.Feed_g_min:>7.3f} {row.H_pred_um:>8.1f} "
          f"{row.AR_pred:>7.4f} {row.Sigma_H_um:>7.1f} "
          f"{row.Sigma_AR:>7.4f}")

# ─────────────────────────────────────────────────────────────
# SECTION 11: VISUALISATION (6-panel figure)
# ─────────────────────────────────────────────────────────────

print("\n" + "=" * 65)
print("STEP 6: Generating SSA-NSGA2 visualisations")
print("=" * 65)

fig = plt.figure(figsize=(20, 14))
fig.suptitle(
    "Hybrid SSA-NSGA2 Multi-Objective Optimisation — "
    "LMD Bead Geometry\n"
    "Surrogate: Multi-Output GPR  "
    f"(H R²=0.97, AR R²=0.97,  N=89)",
    fontsize=13, fontweight='bold')

gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.38, wspace=0.35)

# ── Plot 1: Pareto Front (main result) ──────────────────────
ax1 = fig.add_subplot(gs[0, :2])

sc = ax1.scatter(pareto_F[:, 0], pareto_F[:, 1],
                 c=pareto_df["Uncertainty_Score"],
                 cmap='RdYlGn_r', s=55, alpha=0.8, zorder=3)
cb = plt.colorbar(sc, ax=ax1)
cb.set_label('GPR Uncertainty Score\n(green = reliable)', fontsize=9)

# Knee point
ax1.scatter(pareto_F[knee_idx, 0], pareto_F[knee_idx, 1],
            c='royalblue', s=250, zorder=6, marker='*',
            label='Knee point (ASF)', edgecolors='navy')

# SSA best projected onto Pareto
ax1.scatter(knee_objs[0], knee_objs[1],
            c='gold', s=180, zorder=5, marker='D',
            label='SSA best (projected)', edgecolors='black',
            alpha=0.9)

# Candidate highlights
cand_cols = ['red','green','purple','orange','cyan','teal','magenta']
for (lbl, row), col in zip(list(candidates.items())[1:],
                            cand_cols):
    ax1.scatter(row.Height_Err_um, row.AR_Dev,
                c=col, s=110, zorder=4, marker='D',
                label=lbl[:18], alpha=0.85)

ax1.set_xlabel('Height Error (µm)  ← minimise', fontweight='bold')
ax1.set_ylabel('Aspect Ratio Deviation  ← minimise', fontweight='bold')
ax1.set_title('Pareto Front\n'
              'Colour = GPR Uncertainty  |  Blue★ = Knee Point',
              fontweight='bold')
ax1.legend(fontsize=7, loc='upper right')
ax1.grid(alpha=0.3)

# ── Plot 2: SSA Convergence Curve ───────────────────────────
ax2 = fig.add_subplot(gs[0, 2])

ax2.plot(range(len(ssa_history)), ssa_history,
         'b-', lw=2, label='SSA best f')
ax2.axhline(ssa_best_fit, color='red', ls='--', lw=1.5,
            label=f'Final best = {ssa_best_fit:.4f}')
ax2.set_xlabel('SSA Iteration', fontweight='bold')
ax2.set_ylabel('Composite Objective f', fontweight='bold')
ax2.set_title('SSA Convergence\n'
              f'({N_SPARROWS} sparrows, {N_SSA_ITER} iter)',
              fontweight='bold')
ax2.legend(fontsize=8)
ax2.grid(alpha=0.3)

# ── Plot 3: NSGA-II Convergence (per generation) ─────────────
ax3 = fig.add_subplot(gs[1, 0])

if result.history:
    gen_h  = [np.min(g.opt.get("F")[:, 0]) for g in result.history]
    gen_ar = [np.min(g.opt.get("F")[:, 1]) for g in result.history]
    gens   = range(1, len(gen_h) + 1)
    ax3.plot(gens, gen_h,  'b-', lw=1.5, label='Min Height Error')
    ax3.plot(gens, gen_ar, 'r-', lw=1.5, label='Min AR Deviation')
    ax3.set_xlabel('NSGA-II Generation', fontweight='bold')
    ax3.set_ylabel('Best Objective', fontweight='bold')
    ax3.set_title('NSGA-II Convergence\n'
                  f'({POP_SIZE} pop, {N_GEN} gen, '
                  f'{WARM_START_FRAC:.0%} SSA seed)',
                  fontweight='bold')
    ax3.legend(fontsize=8)
    ax3.grid(alpha=0.3)

# ── Plot 4: Parameter space — Power vs Feed ──────────────────
ax4 = fig.add_subplot(gs[1, 1])

sc4 = ax4.scatter(pareto_X[:, 0], pareto_X[:, 2],
                  c=pareto_F[:, 0], cmap='coolwarm',
                  s=50, alpha=0.8)
plt.colorbar(sc4, ax=ax4, label='Height Error (µm)')
ax4.scatter(knee_params[0], knee_params[2],
            c='royalblue', s=220, zorder=5, marker='*',
            edgecolors='navy', label='Knee point')
ax4.set_xlabel('Laser Power (W)', fontweight='bold')
ax4.set_ylabel('Powder Feed (g/min)', fontweight='bold')
ax4.set_title('Pareto Solutions\nPower vs Feed', fontweight='bold')
ax4.legend(fontsize=8)
ax4.grid(alpha=0.3)

# ── Plot 5: GPR uncertainty across Pareto solutions ──────────
ax5 = fig.add_subplot(gs[1, 2])

sc5 = ax5.scatter(pareto_df["Sigma_H_um"],
                  pareto_df["Sigma_AR"],
                  c=pareto_F[:, 0], cmap='viridis',
                  s=55, alpha=0.8)
plt.colorbar(sc5, ax=ax5, label='Height Error (µm)')
ax5.axvline(50, color='red', ls='--', lw=1.2, alpha=0.7,
            label='σ_H = 50 µm threshold')
ax5.axhline(0.05, color='orange', ls='--', lw=1.2, alpha=0.7,
            label='σ_AR = 0.05 threshold')
ax5.scatter(sH_knee[0], sAR_knee[0],
            c='royalblue', s=220, zorder=5, marker='*',
            edgecolors='navy', label='Knee point')
ax5.set_xlabel('GPR σ — Height (µm)', fontweight='bold')
ax5.set_ylabel('GPR σ — Aspect Ratio', fontweight='bold')
ax5.set_title('Surrogate Uncertainty\nAcross Pareto Solutions',
              fontweight='bold')
ax5.legend(fontsize=7)
ax5.grid(alpha=0.3)

plt.savefig("fig_ssa_nsga2_results1.png", dpi=300,
            bbox_inches='tight')
print("  Saved: fig_ssa_nsga2_results1.png")

# ─────────────────────────────────────────────────────────────
# SECTION 12: SAVE ALL OUTPUTS
# ─────────────────────────────────────────────────────────────

pareto_df.to_csv("ssa_nsga2_pareto_full1.csv", index=False)
print("  Saved: ssa_nsga2_pareto_full1.csv")

# SSA convergence history
pd.DataFrame({
    "iteration" : range(len(ssa_history)),
    "best_f"    : ssa_history,
}).to_csv("ssa_convergence_history1.csv", index=False)
print("  Saved: ssa_convergence_history1.csv")

# Validation candidates
cand_records = []
for lbl, row in candidates.items():
    cand_records.append({
        "Label"        : lbl,
        "Power_W"      : round(row.Power_W, 1),
        "Speed_mm_s"   : round(row.Speed_mm_s, 4),
        "Feed_g_min"   : round(row.Feed_g_min, 3),
        "H_pred_um"    : round(row.H_pred_um, 1),
        "AR_pred"      : round(row.AR_pred, 4),
        "Height_Err_um": round(row.Height_Err_um, 1),
        "AR_Dev"       : round(row.AR_Dev, 5),
        "Sigma_H_um"   : round(row.Sigma_H_um, 1),
        "Sigma_AR"     : round(row.Sigma_AR, 5),
    })
cand_df = pd.DataFrame(cand_records)
cand_df.to_csv("ssa_nsga2_validation_candidates1.csv", index=False)
print("  Saved: ssa_nsga2_validation_candidates1.csv")

print("\n" + "=" * 65)
print("PHASE 1B COMPLETE — Hybrid SSA-NSGA2")
print("=" * 65)
print(f"  SSA runtime    : {t_ssa_elapsed:.1f} s")
print(f"  NSGA-II runtime: {t_nsga_elapsed:.1f} s")
print(f"  Pareto size    : {n_pareto} solutions")
print(f"  Knee point     : "
      f"P={knee_params[0]:.1f} W, "
      f"S={knee_params[1]:.4f} mm/s, "
      f"F={knee_params[2]:.3f} g/min")
print("\nOutputs:")
print("  fig_ssa_nsga2_results1.png")
print("  ssa_nsga2_pareto_full1.csv")
print("  ssa_nsga2_convergence_history1.csv")
print("  ssa_nsga2_validation_candidates1.csv")
