# ============================================================
# 09_bayesian_optimisation.py
# Phase 2: Uncertainty-Aware Bayesian Optimisation
# LMD Bead Geometry — Active Learning via Expected Improvement
# ============================================================
#
# THEORETICAL CONTEXT:
# ─────────────────────────────────────────────────────────────
# Phase 1 (NSGA-II) searched what the GPR already knows.
# Phase 2 asks: "Where should the NEXT experiment go to
# maximally improve the model AND approach the target?"
#
# This is the Bayesian Optimisation (BO) loop:
#   1. Use current GPR as surrogate
#   2. Compute Expected Improvement (EI) across parameter space
#   3. Recommend the point with maximum EI as next experiment
#   4. (After running that experiment) retrain GPR with new point
#   5. Repeat until target is reached or budget exhausted
#
# EI balances exploration (high σ regions) and exploitation
# (regions where GPR predicts good performance). This is
# mathematically optimal for sequential experiment design
# under a Gaussian process prior [Mockus 1978].
# ============================================================

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')           # non-interactive backend — no plt.show() needed
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import norm
from scipy.optimize import differential_evolution
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# ─────────────────────────────────────────────
# SECTION 1: LOAD SURROGATE MODELS & TRAINING DATA
# ─────────────────────────────────────────────
# We need both the GPR models (for EI computation) and
# the original training data (to know the current best
# observed value f* for the EI formula).

print("=" * 65)
print("PHASE 2: Uncertainty-Aware Bayesian Optimisation")
print("LMD Bead Geometry — Active Learning via Expected Improvement")
print("=" * 65)

print("\nSTEP 1: Loading models and data")
print("-" * 45)

gpr_height = joblib.load("best_model_height.pkl")
gpr_aspect  = joblib.load("best_model_aspect.pkl")
scaler_X    = joblib.load("scaler_X.pkl")
scaler_H    = joblib.load("scaler_height.pkl")
scaler_A    = joblib.load("scaler_aspect.pkl")

# Load original dataset to establish f* (current best)
df = pd.read_csv("../data/processed/processed_data.csv")

# Reconstruct the 4-feature matrix exactly as in the notebook
X_raw  = df[["Power", "Speed", "Feed"]].values
heat   = X_raw[:, 0] / (X_raw[:, 1] + 1e-9)
X_all  = np.column_stack([X_raw, heat])

y_H    = df["height_um"].values       # bead height (µm)
y_AR   = df["aspect_ratio"].values    # aspect ratio

print(f"  Training samples   : {len(df)}")
print(f"  Height range       : {y_H.min():.1f} – {y_H.max():.1f} µm")
print(f"  Aspect ratio range : {y_AR.min():.3f} – {y_AR.max():.3f}")

# ─────────────────────────────────────────────
# SECTION 2: DEFINE TARGETS AND DESIGN SPACE
# ─────────────────────────────────────────────
# Phase 1 showed that H=400µm + AR=0.60 is infeasible in
# the current data region. We keep the same targets so that
# BO can actively search the unexplored high-feed regime
# where Phase 1 found high uncertainty.

H_target  = 1200.0    # µm
AR_target = 0.35     # dimensionless

# Parameter bounds — same as Phase 1
BOUNDS = [(1600.0, 2800.0),   # Power (W)
          (0.8,    1.6),       # Speed (mm/s)
          (2.0,    8.0)]       # Feed (g/min)

BOUNDS_LOWER = np.array([b[0] for b in BOUNDS])
BOUNDS_UPPER = np.array([b[1] for b in BOUNDS])

# Normalisation scales for objectives — use GPR output σ at
# training mean as a natural scale for each objective.
# This makes height error and AR deviation commensurable.
SCALE_H  = scaler_H.scale_[0]    # ≈ dataset std of height (µm)
SCALE_AR = scaler_A.scale_[0]    # ≈ dataset std of AR

print(f"\n  Target height      : {H_target:.1f} µm")
print(f"  Target aspect ratio: {AR_target:.3f}")
print(f"  Scale H (σ_data)   : {SCALE_H:.1f} µm")
print(f"  Scale AR (σ_data)  : {SCALE_AR:.4f}")

# ─────────────────────────────────────────────
# SECTION 3: SURROGATE PREDICTION WITH UNCERTAINTY
# ─────────────────────────────────────────────
# Identical to Phase 1 — included here for self-containment.

def predict(X_raw):
    """Predict height and AR with GPR uncertainty (σ)."""
    X_raw = np.atleast_2d(X_raw)
    heat  = X_raw[:, 0] / (X_raw[:, 1] + 1e-9)
    Xe    = np.column_stack([X_raw, heat])
    Xs    = scaler_X.transform(Xe)

    mu_H,  s_H  = gpr_height.predict(Xs, return_std=True)
    mu_AR, s_AR = gpr_aspect.predict(Xs,  return_std=True)

    H_orig  = scaler_H.inverse_transform(mu_H.reshape(-1,1)).ravel()
    AR_orig = scaler_A.inverse_transform(mu_AR.reshape(-1,1)).ravel()
    sH_orig  = s_H  * SCALE_H
    sAR_orig = s_AR * SCALE_AR

    return H_orig, AR_orig, sH_orig, sAR_orig

# ─────────────────────────────────────────────
# SECTION 4: COMPOSITE OBJECTIVE (SCALARISATION)
# ─────────────────────────────────────────────
# Convert two objectives into one scalar using the
# Augmented Tchebycheff function. This is the composite
# objective f(x) whose improvement we will compute EI for.
#
# f_composite = max(|ΔH|/scale_H, |ΔAR|/scale_AR)
#             + ρ * (|ΔH|/scale_H + |ΔAR|/scale_AR)
#
# ρ = 0.05 is a small regularisation term that prevents the
# function from ignoring one objective entirely (it penalises
# any imbalance between the two normalised errors).

RHO = 0.05   # Tchebycheff regularisation — standard value

def composite_objective(H_pred, AR_pred):
    """
    Compute scalarised objective value(s).
    Lower is better — zero means perfect match on both targets.
    """
    dH  = np.abs(H_pred  - H_target)  / SCALE_H
    dAR = np.abs(AR_pred - AR_target) / SCALE_AR
    return np.maximum(dH, dAR) + RHO * (dH + dAR)

# Compute f* = best (lowest) composite objective seen in training data
H_train, AR_train, _, _ = predict(X_raw)
f_train = composite_objective(H_train, AR_train)
f_star  = f_train.min()
best_idx = f_train.argmin()

print(f"\n  Current best composite objective f* : {f_star:.4f}")
print(f"  Best training point  : Power={X_raw[best_idx,0]:.1f}W, "
      f"Speed={X_raw[best_idx,1]:.3f} mm/s, Feed={X_raw[best_idx,2]:.2f} g/min")
print(f"  Predicted geometry   : H={H_train[best_idx]:.1f} µm, "
      f"AR={AR_train[best_idx]:.4f}")

# ─────────────────────────────────────────────
# SECTION 5: EXPECTED IMPROVEMENT ACQUISITION FUNCTION
# ─────────────────────────────────────────────
# EI(x) = E[max(f* - f(x), 0)]
#
# For a Gaussian posterior f(x) ~ N(μ, σ²):
#   EI(x) = (f* - μ(x)) * Φ(Z) + σ(x) * φ(Z)
#   where Z = (f* - μ(x)) / σ(x)
#
# We need σ of the COMPOSITE objective. We approximate this
# using uncertainty propagation:
#   σ_composite ≈ sqrt( (∂f/∂H)² σ_H² + (∂f/∂AR)² σ_AR² )
#
# For the Tchebycheff max function, the dominant term
# (whichever objective is larger) contributes its σ fully.
# We use a smooth approximation: σ_composite ≈ max(σ_H/scale_H, σ_AR/scale_AR).

def expected_improvement(X_candidate, f_star, xi=0.01):
    """
    Compute Expected Improvement at candidate points.

    Parameters
    ----------
    X_candidate : (n, 3) array of [Power, Speed, Feed]
    f_star      : float — current best composite objective
    xi          : float — jitter for exploration (default 0.01)
                  Higher xi = more exploration, less exploitation.
                  Typical range: 0.001 – 0.1

    Returns
    -------
    ei : (n,) array of EI values (higher = more promising)
    mu : (n,) GPR composite mean predictions
    sigma : (n,) GPR composite uncertainty estimates
    """
    H_pred, AR_pred, sH, sAR = predict(X_candidate)

    # Composite mean prediction
    mu = composite_objective(H_pred, AR_pred)

    # Composite uncertainty (dominant-term approximation)
    dH_norm  = np.abs(H_pred  - H_target) / SCALE_H
    dAR_norm = np.abs(AR_pred - AR_target) / SCALE_AR
    # sigma propagation: whichever objective dominates gets its uncertainty
    sigma = np.where(dH_norm > dAR_norm,
                     sH  / SCALE_H,
                     sAR / SCALE_AR)
    sigma = np.maximum(sigma, 1e-8)   # numerical stability

    # EI closed-form (Mockus 1978)
    # xi adds a small offset to f* to encourage more exploration
    improvement = f_star - mu - xi
    Z  = improvement / sigma
    ei = improvement * norm.cdf(Z) + sigma * norm.pdf(Z)
    ei = np.maximum(ei, 0.0)          # EI is non-negative by definition

    return ei, mu, sigma

# ─────────────────────────────────────────────
# SECTION 6: MAXIMISE EI — FIND NEXT EXPERIMENT
# ─────────────────────────────────────────────
# We maximise EI over the full 3D parameter space using
# Differential Evolution — a gradient-free global optimiser
# that handles multimodal acquisition functions well.
# (EI landscapes often have many local maxima.)
#
# We run BO for N_ITERATIONS virtual "rounds", where each
# round simulates adding a new experiment. In practice,
# you would run the FIRST recommended experiment in the lab,
# update the GPR with the real result, then call this script
# again with the new data point added.

N_ITERATIONS  = 10   # number of sequential BO recommendations
XI_SCHEDULE   = [0.1, 0.1, 0.05, 0.05, 0.02, 0.02, 0.01, 0.01, 0.005, 0.005]
# xi decreases over iterations: start exploratory, become exploitative

print("\n" + "=" * 65)
print("STEP 2: Sequential Expected Improvement Maximisation")
print(f"  Iterations : {N_ITERATIONS}")
print(f"  xi schedule: {XI_SCHEDULE}")
print("=" * 65)

# Store all BO recommendations
bo_results = []

# We work in a "virtual" mode — after each recommendation we
# add the GPR's predicted value as a "virtual observation" to
# track how f* would evolve if experiments were run sequentially.
# This lets us generate the full recommendation list NOW
# rather than waiting for real experiments.
virtual_X = X_raw.copy()
virtual_f = f_train.copy()
virtual_fstar = f_star

for i in range(N_ITERATIONS):
    xi = XI_SCHEDULE[i]

    # Maximise EI using Differential Evolution
    # We minimise -EI (scipy minimises by convention)
    def neg_ei(x):
        x = np.atleast_2d(x)
        ei, _, _ = expected_improvement(x, virtual_fstar, xi=xi)
        return -ei[0]

    result = differential_evolution(
        neg_ei,
        bounds=BOUNDS,
        seed=42 + i,
        maxiter=300,
        popsize=20,
        tol=1e-6,
        mutation=(0.5, 1.5),
        recombination=0.9,
        polish=True        # local refinement after global search
    )

    x_next  = result.x
    ei_val  = -result.fun

    # Evaluate GPR at recommended point
    H_next, AR_next, sH_next, sAR_next = predict(x_next.reshape(1, -1))
    f_next = composite_objective(H_next, AR_next)[0]

    # Update virtual best for next iteration
    if f_next < virtual_fstar:
        virtual_fstar = f_next

    bo_results.append({
        "Iteration"   : i + 1,
        "Power_W"     : x_next[0],
        "Speed_mm_s"  : x_next[1],
        "Feed_g_min"  : x_next[2],
        "EI"          : ei_val,
        "xi"          : xi,
        "H_pred_um"   : H_next[0],
        "AR_pred"     : AR_next[0],
        "Sigma_H_um"  : sH_next[0],
        "Sigma_AR"    : sAR_next[0],
        "Composite_f" : f_next,
        "f_star"      : virtual_fstar,
    })

    print(f"  Iter {i+1:2d} | EI={ei_val:.4f} | "
          f"P={x_next[0]:.1f}W S={x_next[1]:.3f} F={x_next[2]:.2f} | "
          f"H={H_next[0]:.1f}µm AR={AR_next[0]:.4f} | "
          f"σ_H={sH_next[0]:.1f} σ_AR={sAR_next[0]:.4f} | "
          f"f={f_next:.4f}")

bo_df = pd.DataFrame(bo_results)

# ─────────────────────────────────────────────
# SECTION 7: EXPLORATION vs EXPLOITATION ANALYSIS
# ─────────────────────────────────────────────
# Classify each BO recommendation as primarily exploitative
# (going to a region with good predicted performance) or
# exploratory (going to a high-uncertainty region).
#
# Rule: if σ_H > 50 µm OR σ_AR > 0.05 → exploratory (blind spot)
#       otherwise → exploitative (exploiting known good region)

bo_df["Type"] = np.where(
    (bo_df["Sigma_H_um"] > 50) | (bo_df["Sigma_AR"] > 0.05),
    "Exploratory", "Exploitative"
)

print("\n" + "─" * 65)
print("BO Recommendation Classification:")
print(bo_df[["Iteration","Type","Power_W","Speed_mm_s","Feed_g_min",
             "H_pred_um","AR_pred","Sigma_H_um","Composite_f"]].to_string(index=False))

# ─────────────────────────────────────────────
# SECTION 8: EI LANDSCAPE VISUALISATION
# ─────────────────────────────────────────────
# Generate a 2D slice of the EI landscape to understand
# where the acquisition function is directing the search.
# We slice at the Feed rate of the best BO recommendation
# (the most informative dimension from Phase 1).

print("\n" + "=" * 65)
print("STEP 3: Generating EI landscape and BO visualisations")
print("=" * 65)

fig = plt.figure(figsize=(20, 14))
fig.suptitle("Phase 2 — Uncertainty-Aware Bayesian Optimisation\n"
             "LMD Bead Geometry Active Learning",
             fontsize=13, fontweight='bold')

gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.38, wspace=0.35)

# ── Plot 1: EI Landscape — Power vs Feed (slice at best Speed) ──
ax1 = fig.add_subplot(gs[0, :2])

best_speed = bo_df.iloc[0]["Speed_mm_s"]
P_grid = np.linspace(BOUNDS_LOWER[0], BOUNDS_UPPER[0], 80)
F_grid = np.linspace(BOUNDS_LOWER[2], BOUNDS_UPPER[2], 80)
PP, FF  = np.meshgrid(P_grid, F_grid)
SS      = np.full_like(PP, best_speed)
grid_X  = np.column_stack([PP.ravel(), SS.ravel(), FF.ravel()])

ei_grid, mu_grid, sigma_grid = expected_improvement(
    grid_X, f_star, xi=XI_SCHEDULE[0]
)
EI_map = ei_grid.reshape(PP.shape)

cf = ax1.contourf(PP, FF, EI_map, levels=30, cmap='YlOrRd')
plt.colorbar(cf, ax=ax1, label='Expected Improvement (higher = more promising)')

# Overlay training data points
ax1.scatter(X_raw[:, 0], X_raw[:, 2],
            c='blue', s=40, alpha=0.6, zorder=3, label='Training data (N=89)')

# Overlay BO recommendations coloured by iteration
colours_iter = plt.cm.winter(np.linspace(0, 1, N_ITERATIONS))
for _, row in bo_df.iterrows():
    ax1.scatter(row.Power_W, row.Feed_g_min,
                color=colours_iter[int(row.Iteration)-1],
                s=150, zorder=5, marker='*',
                label=f"Iter {int(row.Iteration)}" if row.Iteration <= 3 else "")

# Mark Phase 1 high-uncertainty candidates
ax1.scatter(1600.0, 7.997, c='red', s=200, marker='X', zorder=6,
            label='Phase 1 high-σ candidate')

ax1.set_xlabel('Laser Power (W)', fontweight='bold')
ax1.set_ylabel('Powder Feed (g/min)', fontweight='bold')
ax1.set_title(f'EI Landscape — Power vs Feed\n'
              f'(Speed fixed at {best_speed:.3f} mm/s, xi={XI_SCHEDULE[0]})\n'
              f'Yellow/Red = high EI (next experiment here)',
              fontweight='bold')
ax1.legend(fontsize=7, loc='upper right')
ax1.grid(alpha=0.2)

# ── Plot 2: GPR Uncertainty Map — same slice ─────────────────
ax2 = fig.add_subplot(gs[0, 2])

H_map_grid, _, sH_map, sAR_map = predict(grid_X)
sH_2D = sH_map.reshape(PP.shape)

cf2 = ax2.contourf(PP, FF, sH_2D, levels=20, cmap='RdYlGn_r')
plt.colorbar(cf2, ax=ax2, label='GPR σ_Height (µm)')
ax2.contour(PP, FF, sH_2D, levels=[50], colors='red',
            linewidths=2, linestyles='--')  # reliability threshold
ax2.scatter(X_raw[:, 0], X_raw[:, 2],
            c='blue', s=30, alpha=0.5, zorder=3)
ax2.set_xlabel('Laser Power (W)', fontweight='bold')
ax2.set_ylabel('Powder Feed (g/min)', fontweight='bold')
ax2.set_title('GPR Uncertainty σ_H\nRed dashed = 50 µm threshold',
              fontweight='bold')
ax2.grid(alpha=0.2)

# ── Plot 3: BO Convergence — f* over iterations ───────────────
ax3 = fig.add_subplot(gs[1, 0])

ax3.plot(bo_df["Iteration"], bo_df["f_star"],
         'b-o', lw=2, ms=6, label='Best f* so far')
ax3.plot(bo_df["Iteration"], bo_df["Composite_f"],
         'r--s', lw=1.5, ms=5, alpha=0.7, label='f at recommended point')
ax3.axhline(y=f_star, color='gray', ls=':', lw=1.5,
            label=f'Phase 1 best f*={f_star:.3f}')
ax3.set_xlabel('BO Iteration', fontweight='bold')
ax3.set_ylabel('Composite Objective f (lower = better)', fontweight='bold')
ax3.set_title('BO Convergence\nf* = best composite objective seen',
              fontweight='bold')
ax3.legend(fontsize=8)
ax3.grid(alpha=0.3)

# ── Plot 4: EI value over iterations ─────────────────────────
ax4 = fig.add_subplot(gs[1, 1])

colours_type = bo_df["Type"].map({"Exploratory": "red",
                                   "Exploitative": "blue"})
ax4.bar(bo_df["Iteration"], bo_df["EI"],
        color=colours_type, alpha=0.7, edgecolor='black')
ax4.set_xlabel('BO Iteration', fontweight='bold')
ax4.set_ylabel('Expected Improvement', fontweight='bold')
ax4.set_title('EI per Iteration\nRed=Exploratory, Blue=Exploitative',
              fontweight='bold')

from matplotlib.patches import Patch
legend_el = [Patch(facecolor='red', label='Exploratory (high σ)'),
             Patch(facecolor='blue', label='Exploitative (low σ)')]
ax4.legend(handles=legend_el, fontsize=8)
ax4.grid(alpha=0.3, axis='y')

# ── Plot 5: Predicted geometry at BO recommendations ─────────
ax5 = fig.add_subplot(gs[1, 2])

sc5 = ax5.scatter(bo_df["H_pred_um"], bo_df["AR_pred"],
                  c=bo_df["Iteration"], cmap='winter',
                  s=120, zorder=4, edgecolors='black', linewidths=0.5)
plt.colorbar(sc5, ax=ax5, label='BO Iteration')
ax5.axvline(x=H_target,  color='blue', ls='--', lw=1.5, label=f'H target={H_target}µm')
ax5.axhline(y=AR_target, color='red',  ls='--', lw=1.5, label=f'AR target={AR_target}')
ax5.scatter(H_target, AR_target, c='gold', s=400, marker='*',
            zorder=6, label='Target point', edgecolors='black')

# Add iteration labels
for _, row in bo_df.iterrows():
    ax5.annotate(f"{int(row.Iteration)}",
                 (row.H_pred_um, row.AR_pred),
                 textcoords="offset points", xytext=(5, 5), fontsize=8)

ax5.set_xlabel('Predicted Height (µm)', fontweight='bold')
ax5.set_ylabel('Predicted Aspect Ratio', fontweight='bold')
ax5.set_title('BO Recommendations\nin Geometry Space',
              fontweight='bold')
ax5.legend(fontsize=8)
ax5.grid(alpha=0.3)

plt.savefig("fig_bayesian_optimisation.png", dpi=300, bbox_inches='tight')
print("  Saved: fig_bayesian_optimisation.png")

# ─────────────────────────────────────────────
# SECTION 9: SENSITIVITY ANALYSIS — xi SWEEP
# ─────────────────────────────────────────────
# xi is the exploration-exploitation trade-off parameter.
# Here we show how the FIRST recommendation changes with xi
# to help you choose the right value for your thesis.

print("\n" + "=" * 65)
print("STEP 4: xi Sensitivity Analysis (Exploration-Exploitation Trade-off)")
print("=" * 65)
print(f"{'xi':>8} {'Power':>8} {'Speed':>8} {'Feed':>8} "
      f"{'H_pred':>8} {'AR_pred':>8} {'σ_H':>7} {'EI':>10} {'Type':>14}")
print("-" * 85)

xi_values = [0.001, 0.01, 0.05, 0.1, 0.2, 0.5]
xi_rows   = []

for xi_test in xi_values:
    def neg_ei_xi(x):
        x = np.atleast_2d(x)
        ei, _, _ = expected_improvement(x, f_star, xi=xi_test)
        return -ei[0]

    res = differential_evolution(
        neg_ei_xi, bounds=BOUNDS, seed=42,
        maxiter=200, popsize=15, polish=True
    )
    x_xi = res.x
    ei_xi = -res.fun
    H_xi, AR_xi, sH_xi, _ = predict(x_xi.reshape(1, -1))
    t = "Exploratory" if sH_xi[0] > 50 else "Exploitative"

    print(f"  {xi_test:>6.3f} {x_xi[0]:>8.1f} {x_xi[1]:>8.3f} {x_xi[2]:>8.2f} "
          f"{H_xi[0]:>8.1f} {AR_xi[0]:>8.4f} {sH_xi[0]:>7.1f} {ei_xi:>10.6f} {t:>14}")

    xi_rows.append({"xi": xi_test, "Power_W": x_xi[0], "Speed_mm_s": x_xi[1],
                    "Feed_g_min": x_xi[2], "H_pred_um": H_xi[0],
                    "AR_pred": AR_xi[0], "Sigma_H_um": sH_xi[0],
                    "EI": ei_xi, "Type": t})

xi_df = pd.DataFrame(xi_rows)

# ─────────────────────────────────────────────
# SECTION 10: SAVE ALL OUTPUTS
# ─────────────────────────────────────────────

bo_df.to_csv("bo_recommendations.csv", index=False)
xi_df.to_csv("bo_xi_sensitivity.csv",  index=False)
print("\n  Saved: bo_recommendations.csv")
print("  Saved: bo_xi_sensitivity.csv")

print("\n" + "=" * 65)
print("PHASE 2 COMPLETE — SUMMARY")
print("=" * 65)
print(f"  BO recommendations generated   : {N_ITERATIONS}")
print(f"  Initial f* (Phase 1 best)       : {f_star:.4f}")
print(f"  Final   f* (after BO virtual)   : {bo_df.iloc[-1].f_star:.4f}")
f_improvement = (f_star - bo_df.iloc[-1].f_star) / f_star * 100
print(f"  Objective improvement (virtual) : {f_improvement:.1f}%")
print(f"\n  Priority experiments for the lab:")
print(f"  1. Run BO Iteration 1 first (highest EI)")
print(f"  2. Retrain GPR with new data point")
print(f"  3. Re-run this script for the updated recommendation")
print(f"  4. Repeat until target geometry is achieved or budget used")
print("\n  Outputs:")
print("    bo_recommendations.csv    — all 10 recommended experiments")
print("    bo_xi_sensitivity.csv     — xi exploration trade-off table")
print("    fig_bayesian_optimisation.png — full visualisation panel")
