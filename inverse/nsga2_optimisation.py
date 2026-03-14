# ============================================================
# 08_nsga2_optimisation.py
# Phase 1: GPR-Surrogate NSGA-II Multi-Objective Optimisation
# LMD Bead Geometry — Height Error + Aspect Ratio Deviation
# ============================================================

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.termination import get_termination
from pymoo.optimize import minimize
from pymoo.decomposition.asf import ASF
from pymoo.visualization.scatter import Scatter

# ─────────────────────────────────────────────
# SECTION 1: LOAD SAVED SURROGATE MODELS
# ─────────────────────────────────────────────
# These are the serialised GPR models from your notebook.
# The scalers are essential — the GPR was trained on
# standardised (mean=0, std=1) inputs and targets.
# Without them, predictions will be in wrong units.

print("=" * 60)
print("STEP 1: Loading saved GPR surrogate models")
print("=" * 60)

gpr_height = joblib.load("best_model_height.pkl")
gpr_aspect  = joblib.load("best_model_aspect.pkl")
scaler_X    = joblib.load("scaler_X.pkl")
scaler_H    = joblib.load("scaler_height.pkl")
scaler_A    = joblib.load("scaler_aspect.pkl")

print("  GPR Height    : loaded")
print("  GPR Aspect    : loaded")
print("  Scaler X      : loaded")
print("  Scaler Height : loaded")
print("  Scaler Aspect : loaded")

# ─────────────────────────────────────────────
# SECTION 2: DEFINE TARGET GEOMETRY
# ─────────────────────────────────────────────
# These are YOUR engineering targets.
# Change these to match the bead geometry your
# application requires.
#
# H_target  = desired bead height in micrometres
# AR_target = desired aspect ratio (Width/Height)
#
# The notebook showed: mean height ≈ 803 µm, mean AR ≈ 0.342
# A reasonable starting target is near the dataset mean.

H_target  = 1000.0   # µm  — adjust to your application
AR_target = 0.35    # W/H — adjust to your application

print(f"\nTarget bead height   : {H_target:.1f} µm")
print(f"Target aspect ratio  : {AR_target:.3f}")

# ─────────────────────────────────────────────
# SECTION 3: DEFINE THE SURROGATE PREDICTION FUNCTION
# ─────────────────────────────────────────────
# This function is the bridge between NSGA-II and your GPR.
# NSGA-II proposes [Power, Speed, Feed] combinations;
# this function asks the GPR "what geometry do you predict?"
# and returns it in original physical units.
#
# IMPORTANT: The GPR needs a 4-column input because your
# notebook added HeatInput = Power/Speed as a 4th feature.
# We must recreate this derived feature here too.

def predict_geometry(X_raw):
    """
    X_raw : np.ndarray of shape (n, 3)
            Columns: [Power (W), Speed (mm/s), Feed (g/min)]

    Returns
    -------
    H_pred  : np.ndarray (n,) — predicted bead height in µm
    AR_pred : np.ndarray (n,) — predicted aspect ratio
    """
    X_raw = np.atleast_2d(X_raw)

    # Recreate the HeatInput feature (Physics-informed)
    # This exactly mirrors Cell 3 in your notebook:
    #   heatinput = Power / (Speed + 1e-9)
    heat_input = X_raw[:, 0] / (X_raw[:, 1] + 1e-9)

    # Build 4-column feature matrix [Power, Speed, Feed, HeatInput]
    X_eng = np.column_stack([X_raw, heat_input])

    # Standardise using the saved scaler (mean=0, std=1)
    X_scaled = scaler_X.transform(X_eng)

    # GPR prediction — returns standardised outputs
    H_scaled  = gpr_height.predict(X_scaled)
    AR_scaled = gpr_aspect.predict(X_scaled)

    # Back-transform to original physical units
    H_pred  = scaler_H.inverse_transform(H_scaled.reshape(-1, 1)).ravel()
    AR_pred = scaler_A.inverse_transform(AR_scaled.reshape(-1, 1)).ravel()

    return H_pred, AR_pred


def predict_geometry_with_uncertainty(X_raw):
    """
    Same as predict_geometry but also returns GPR posterior σ.
    Used later for uncertainty-aware filtering of the Pareto front.
    """
    X_raw = np.atleast_2d(X_raw)
    heat_input = X_raw[:, 0] / (X_raw[:, 1] + 1e-9)
    X_eng    = np.column_stack([X_raw, heat_input])
    X_scaled = scaler_X.transform(X_eng)

    H_scaled,  sigma_H  = gpr_height.predict(X_scaled, return_std=True)
    AR_scaled, sigma_AR = gpr_aspect.predict(X_scaled,  return_std=True)

    H_pred  = scaler_H.inverse_transform(H_scaled.reshape(-1, 1)).ravel()
    AR_pred = scaler_A.inverse_transform(AR_scaled.reshape(-1, 1)).ravel()

    # Back-transform σ: since scaler is StandardScaler,
    # σ in original units ≈ σ_scaled × scaler.scale_
    sigma_H_orig  = sigma_H  * scaler_H.scale_
    sigma_AR_orig = sigma_AR * scaler_A.scale_

    return H_pred, AR_pred, sigma_H_orig, sigma_AR_orig

# ─────────────────────────────────────────────
# SECTION 4: DEFINE THE OPTIMISATION PROBLEM
# ─────────────────────────────────────────────
# This is the core class pymoo needs.
# We inherit from Problem and implement _evaluate().
#
# Decision variables (what NSGA-II controls):
#   x = Laser Power   (W)
#   x = Scan Speed    (mm/s)[1]
#   x = Powder Feed   (g/min)[2]
#
# Objectives (what we want to minimise simultaneously):
#   f = |H_predicted - H_target|      height error in µm
#   f[1] = |AR_predicted - AR_target|    aspect ratio deviation
#
# Bounds: set from your experimental design space.
# Your notebook showed Power 1620–2470W, Speed 0.9–1.5 mm/s,
# Feed 2–7.5 g/min. We use those ranges with small margins.

# ── Edit these bounds to match your actual experimental window ──
BOUNDS_LOWER = np.array([1600.0,  0.8,  2.0])  # [W, mm/s, g/min]
BOUNDS_UPPER = np.array([2800.0,  1.6,  8.0])  # [W, mm/s, g/min]
# ───────────────────────────────────────────────────────────────

class LMDProblem(Problem):
    """
    Two-objective optimisation problem for LMD bead geometry.
    Surrogate: GPR trained on 89 experimental samples.
    """

    def __init__(self):
        super().__init__(
            n_var=3,           # 3 decision variables
            n_obj=2,           # 2 objectives
            n_constr=0,        # no hard constraints (bounds handle this)
            xl=BOUNDS_LOWER,   # lower bounds
            xu=BOUNDS_UPPER,   # upper bounds
        )

    def _evaluate(self, X, out, *args, **kwargs):
        """
        X   : (pop_size, 3) array of candidate solutions
        out : dict — we write objectives into out["F"]

        pymoo calls this once per generation with the entire
        population, so we vectorise the GPR call for speed.
        """
        H_pred, AR_pred = predict_geometry(X)

        # Objective 1: height error (µm)
        f1 = np.abs(H_pred - H_target)

        # Objective 2: aspect ratio deviation (dimensionless)
        f2 = np.abs(AR_pred - AR_target)

        # pymoo expects shape (n_pop, n_obj)
        out["F"] = np.column_stack([f1, f2])

# ─────────────────────────────────────────────
# SECTION 5: CONFIGURE AND RUN NSGA-II
# ─────────────────────────────────────────────
# NSGA-II works like a biological evolutionary process:
#   1. Start with a random "population" of 100 parameter sets
#   2. Evaluate each against both objectives (via GPR call)
#   3. Rank them using non-dominated sorting:
#        Rank 1 = solutions no other solution beats on BOTH objectives
#        Rank 2 = solutions beaten only by Rank 1, etc.
#   4. Within each rank, prefer solutions with more "space" around
#      them (crowding distance) to maintain diversity
#   5. Select parents → crossover → mutation → new generation
#   6. Repeat for 300 generations
#
# pop_size=100 and n_gen=300 are standard starting values.
# For a 3D parameter space with N=89 training data,
# this is computationally trivial (< 2 minutes on CPU).

print("\n" + "=" * 60)
print("STEP 2: Running NSGA-II optimisation")
print(f"  Population size : 100")
print(f"  Generations     : 300")
print(f"  Objectives      : Height Error + AR Deviation")
print(f"  Design space    : Power {BOUNDS_LOWER[0]:.0f}–{BOUNDS_UPPER[0]:.0f}W, "
      f"Speed {BOUNDS_LOWER[1]:.1f}–{BOUNDS_UPPER[1]:.1f} mm/s, "
      f"Feed {BOUNDS_LOWER[2]:.1f}–{BOUNDS_UPPER[2]:.1f} g/min")
print("=" * 60)

problem    = LMDProblem()
algorithm  = NSGA2(
    pop_size=100,
    eliminate_duplicates=True   # prevents identical solutions
)
termination = get_termination("n_gen", 300)

result = minimize(
    problem,
    algorithm,
    termination,
    seed=42,           # reproducibility
    save_history=True, # allows convergence analysis later
    verbose=True       # prints progress every 10 generations
)

print(f"\n✓ Optimisation complete.")
print(f"  Pareto-optimal solutions found : {len(result.X)}")

# ─────────────────────────────────────────────
# SECTION 6: EXTRACT AND ENRICH PARETO RESULTS
# ─────────────────────────────────────────────
# result.X = Pareto-optimal process parameter sets (n_pareto, 3)
# result.F = Corresponding objective values (n_pareto, 2)
#
# We now re-evaluate with uncertainty to add GPR σ columns.
# This tells us WHICH Pareto solutions are in well-sampled
# regions (low σ = trustworthy) vs uncertain regions (high σ).

pareto_X = result.X    # shape: (n_pareto, 3) — [Power, Speed, Feed]
pareto_F = result.F    # shape: (n_pareto, 2) — [height_err, AR_dev]

H_pred, AR_pred, sigma_H, sigma_AR = predict_geometry_with_uncertainty(pareto_X)

# Assemble a results DataFrame — easy to inspect and export
pareto_df = pd.DataFrame({
    "Power_W"       : pareto_X[:, 0],
    "Speed_mm_s"    : pareto_X[:, 1],
    "Feed_g_min"    : pareto_X[:, 2],
    "H_pred_um"     : H_pred,
    "AR_pred"       : AR_pred,
    "Height_Err_um" : pareto_F[:, 0],
    "AR_Dev"        : pareto_F[:, 1],
    "Sigma_H_um"    : sigma_H,
    "Sigma_AR"      : sigma_AR,
    # Reliability score: low = confident prediction
    "Uncertainty_Score": sigma_H / scaler_H.scale_ + sigma_AR / scaler_A.scale_
})

# Sort by height error so best height solutions appear first
pareto_df = pareto_df.sort_values("Height_Err_um").reset_index(drop=True)

# ─────────────────────────────────────────────
# SECTION 7: KNEE-POINT SELECTION
# ─────────────────────────────────────────────
# The Pareto front is a curve of equally valid trade-offs.
# For physical experiments, you need to pick specific points.
# The "knee point" is the solution that offers the best
# simultaneous improvement on BOTH objectives — the natural
# choice when you have no strong preference for one over the other.
#
# ASF (Achievement Scalarisation Function) finds the knee by
# minimising the weighted max of normalised objectives.
# weights=[0.5, 0.5] means equal importance to both objectives.

weights  = np.array([0.5, 0.5])
decomp   = ASF()
knee_idx = decomp.do(pareto_F, 1 / weights).argmin()

knee_params = pareto_X[knee_idx]
knee_objs   = pareto_F[knee_idx]

print("\n" + "=" * 60)
print("STEP 3: Knee-Point (Best Trade-Off Solution)")
print("=" * 60)
print(f"  Laser Power   : {knee_params[0]:.1f} W")
print(f"  Scan Speed    : {knee_params[1]:.3f} mm/s")
print(f"  Powder Feed   : {knee_params[2]:.2f} g/min")
print(f"  → Predicted Height    : {H_pred[knee_idx]:.1f} µm  (target: {H_target:.1f})")
print(f"  → Predicted AR        : {AR_pred[knee_idx]:.4f}    (target: {AR_target:.4f})")
print(f"  → Height Error        : {knee_objs[0]:.1f} µm")
print(f"  → AR Deviation        : {knee_objs[1]:.4f}")
print(f"  → GPR σ_Height        : ±{sigma_H[knee_idx]:.1f} µm")
print(f"  → GPR σ_AR            : ±{sigma_AR[knee_idx]:.4f}")

# ─────────────────────────────────────────────
# SECTION 8: SELECT VALIDATION EXPERIMENT CANDIDATES
# ─────────────────────────────────────────────
# From the Pareto front, we select 7 candidate points
# covering different priorities:
#   1. Knee point (best trade-off)
#   2. Min height error (geometric precision priority)
#   3. Min AR deviation (bead shape priority)
#   4–5. High uncertainty (model blind spots — IMPORTANT)
#   6–7. Diverse interior Pareto points

'''
n_pareto = len(pareto_df)

candidates = {}

# 1. Knee point
candidates["Knee Point"]          = pareto_df.iloc[knee_idx]

# 2 & 3. Extremes of Pareto front
candidates["Min Height Error"]    = pareto_df.loc[pareto_df["Height_Err_um"].idxmin()]
candidates["Min AR Deviation"]    = pareto_df.loc[pareto_df["AR_Dev"].idxmin()]

# 4 & 5. Highest GPR uncertainty — these should be validated
#         because the model is least reliable there
top_uncertain = pareto_df.nlargest(2, "Uncertainty_Score")
candidates["High Uncertainty #1"] = top_uncertain.iloc[0]
candidates["High Uncertainty #2"] = top_uncertain.iloc[1]

# 6 & 7. Interior diversity: 25th and 75th percentile of height error
q25_idx = (pareto_df["Height_Err_um"] - pareto_df["Height_Err_um"].quantile(0.25)).abs().idxmin()
q75_idx = (pareto_df["Height_Err_um"] - pareto_df["Height_Err_um"].quantile(0.75)).abs().idxmin()
candidates["Interior Q25"]        = pareto_df.loc[q25_idx]
candidates["Interior Q75"]        = pareto_df.loc[q75_idx]




'''
# ── Candidate selection ──────────────────────────────────────
n_pareto = len(pareto_df)
candidates = {}

# 1. Knee point
candidates["Knee Point"]          = pareto_df.loc[knee_idx]

# 2 & 3. Extremes — use label-based loc throughout
min_h_idx  = pareto_df["Height_Err_um"].idxmin()
min_ar_idx = pareto_df["AR_Dev"].idxmin()
candidates["Min Height Error"]    = pareto_df.loc[min_h_idx]
candidates["Min AR Deviation"]    = pareto_df.loc[min_ar_idx]

# 4 & 5. Highest GPR uncertainty
top_uncertain = pareto_df.nlargest(2, "Uncertainty_Score")
candidates["High Uncertainty #1"] = top_uncertain.iloc[0]
candidates["High Uncertainty #2"] = top_uncertain.iloc[1]

# 6 & 7. Interior diversity — robust to small Pareto fronts
if n_pareto >= 4:
    sorted_idx = pareto_df["Height_Err_um"].argsort().values
    q25_pos    = max(1, int(n_pareto * 0.25))
    q75_pos    = min(n_pareto - 2, int(n_pareto * 0.75))
    candidates["Interior Q25"]    = pareto_df.iloc[sorted_idx[q25_pos]]
    candidates["Interior Q75"]    = pareto_df.iloc[sorted_idx[q75_pos]]
else:
    # Fallback if Pareto front is very small
    candidates["Interior Q25"]    = pareto_df.iloc[0]
    candidates["Interior Q75"]    = pareto_df.iloc[-1]


# Print validation candidate table
print("\n" + "=" * 60)
print("STEP 4: Recommended Validation Experiment Candidates")
print("=" * 60)
print(f"{'Label':<22} {'Power':>7} {'Speed':>7} {'Feed':>6} "
      f"{'H_pred':>8} {'AR_pred':>8} {'σ_H':>7} {'σ_AR':>7}")
print("-" * 80)
for label, row in candidates.items():
    print(f"{label:<22} {row.Power_W:>7.1f} {row.Speed_mm_s:>7.3f} "
          f"{row.Feed_g_min:>6.2f} {row.H_pred_um:>8.1f} {row.AR_pred:>8.4f} "
          f"{row.Sigma_H_um:>7.1f} {row.Sigma_AR:>7.4f}")

# ─────────────────────────────────────────────
# SECTION 9: VISUALISATION
# ─────────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 5: Generating visualisations")
print("=" * 60)

fig = plt.figure(figsize=(18, 14))
fig.suptitle("GPR-NSGA-II Multi-Objective Optimisation — LMD Bead Geometry",
             fontsize=14, fontweight='bold')

gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.35)

# ── Plot 1: Pareto Front (main result) ──────────────────────
ax1 = fig.add_subplot(gs[0, :2])   # spans 2 columns

# Colour the Pareto points by GPR uncertainty
sc = ax1.scatter(pareto_F[:, 0], pareto_F[:, 1],
                 c=pareto_df["Uncertainty_Score"], cmap='RdYlGn_r',
                 s=60, alpha=0.8, zorder=3, label='Pareto solutions')
plt.colorbar(sc, ax=ax1, label='GPR Uncertainty Score (lower = more reliable)')

# Highlight knee point
ax1.scatter(pareto_F[knee_idx, 0], pareto_F[knee_idx, 1],
            c='blue', s=180, zorder=5, marker='*', label=f'Knee point')

# Highlight the special candidates
colours = ['red', 'green', 'purple', 'orange', 'cyan', 'magenta']
for (label, row), col in zip(list(candidates.items())[1:], colours):
    ax1.scatter(row.Height_Err_um, row.AR_Dev,
                c=col, s=120, zorder=4, marker='D', label=label[:15])

ax1.set_xlabel('Height Error (µm)  ← minimise', fontweight='bold')
ax1.set_ylabel('Aspect Ratio Deviation  ← minimise', fontweight='bold')
ax1.set_title('Pareto Front — Colour = GPR Uncertainty\n'
              'Green = confident prediction,  Red = uncertain region',
              fontweight='bold')
ax1.legend(fontsize=7, loc='upper right')
ax1.grid(alpha=0.3)

# ── Plot 2: Pareto solutions in Power-Speed space ────────────
ax2 = fig.add_subplot(gs[0, 2])

sc2 = ax2.scatter(pareto_X[:, 0], pareto_X[:, 1],
                  c=pareto_F[:, 0], cmap='coolwarm',
                  s=50, alpha=0.8, zorder=3)
plt.colorbar(sc2, ax=ax2, label='Height Error (µm)')
ax2.scatter(knee_params[0], knee_params[1],
            c='blue', s=180, zorder=5, marker='*', label='Knee point')
ax2.set_xlabel('Laser Power (W)', fontweight='bold')
ax2.set_ylabel('Scan Speed (mm/s)', fontweight='bold')
ax2.set_title('Pareto Solutions\nPower vs Speed', fontweight='bold')
ax2.legend(fontsize=8)
ax2.grid(alpha=0.3)

# ── Plot 3: Pareto solutions in Power-Feed space ─────────────
ax3 = fig.add_subplot(gs[1, 0])

sc3 = ax3.scatter(pareto_X[:, 0], pareto_X[:, 2],
                  c=pareto_F[:, 1], cmap='coolwarm',
                  s=50, alpha=0.8, zorder=3)
plt.colorbar(sc3, ax=ax3, label='AR Deviation')
ax3.scatter(knee_params[0], knee_params[2],
            c='blue', s=180, zorder=5, marker='*', label='Knee point')
ax3.set_xlabel('Laser Power (W)', fontweight='bold')
ax3.set_ylabel('Powder Feed (g/min)', fontweight='bold')
ax3.set_title('Pareto Solutions\nPower vs Feed', fontweight='bold')
ax3.legend(fontsize=8)
ax3.grid(alpha=0.3)

# ── Plot 4: GPR uncertainty across Pareto solutions ──────────
ax4 = fig.add_subplot(gs[1, 1])

ax4.scatter(pareto_df["Sigma_H_um"], pareto_df["Sigma_AR"],
            c=pareto_F[:, 0], cmap='viridis', s=60, alpha=0.8)
ax4.scatter(sigma_H[knee_idx], sigma_AR[knee_idx],
            c='blue', s=180, zorder=5, marker='*', label='Knee point')
ax4.set_xlabel('GPR σ — Height (µm)', fontweight='bold')
ax4.set_ylabel('GPR σ — Aspect Ratio', fontweight='bold')
ax4.set_title('Model Uncertainty\nAcross Pareto Solutions', fontweight='bold')
ax4.legend(fontsize=8)
ax4.grid(alpha=0.3)
# Mark a reliability threshold (you can tune these)
ax4.axvline(x=50, color='red', ls='--', alpha=0.5, label='σ_H = 50 µm')
ax4.axhline(y=0.05, color='orange', ls='--', alpha=0.5, label='σ_AR = 0.05')

# ── Plot 5: Convergence (hypervolume proxy) ───────────────────
ax5 = fig.add_subplot(gs[1, 2])

# Extract CV R² history — use min objective sum as proxy
if result.history:
    gen_min_f1 = [np.min(gen.opt.get("F")[:, 0]) for gen in result.history]
    gen_min_f2 = [np.min(gen.opt.get("F")[:, 1]) for gen in result.history]
    gens = list(range(1, len(gen_min_f1) + 1))
    ax5.plot(gens, gen_min_f1, 'b-', lw=1.5, label='Min Height Error')
    ax5.plot(gens, gen_min_f2, 'r-', lw=1.5, label='Min AR Deviation')
    ax5.set_xlabel('Generation', fontweight='bold')
    ax5.set_ylabel('Best Objective Value', fontweight='bold')
    ax5.set_title('Convergence History', fontweight='bold')
    ax5.legend(fontsize=8)
    ax5.grid(alpha=0.3)

plt.savefig("fig_nsga2_pareto_results.png", dpi=300, bbox_inches='tight')
print("  Saved: fig_nsga2_pareto_results.png")

# ─────────────────────────────────────────────
# SECTION 10: SAVE RESULTS
# ─────────────────────────────────────────────

# Full Pareto front
pareto_df.to_csv("pareto_front_full.csv", index=False)
print("  Saved: pareto_front_full.csv")

# Validation candidates table
cand_records = []
for label, row in candidates.items():
    cand_records.append({
        "Label"        : label,
        "Power_W"      : round(row.Power_W, 1),
        "Speed_mm_s"   : round(row.Speed_mm_s, 4),
        "Feed_g_min"   : round(row.Feed_g_min, 3),
        "H_pred_um"    : round(row.H_pred_um, 1),
        "AR_pred"      : round(row.AR_pred, 4),
        "Height_Err_um": round(row.Height_Err_um, 1),
        "AR_Dev"       : round(row.AR_Dev, 4),
        "Sigma_H_um"   : round(row.Sigma_H_um, 1),
        "Sigma_AR"     : round(row.Sigma_AR, 4),
    })

cand_df = pd.DataFrame(cand_records)
cand_df.to_csv("validation_candidates.csv", index=False)
print("  Saved: validation_candidates.csv")

print("\n" + "=" * 60)
print("PHASE 1 COMPLETE")
print("=" * 60)
print("Next steps:")
print("  1. Review 'pareto_front_full.csv' — all Pareto solutions")
print("  2. Review 'validation_candidates.csv' — 7 experiment candidates")
print("  3. Prioritise LOW uncertainty candidates for first experiments")
print("  4. Include HIGH uncertainty candidates to reduce model blind spots")
print("  5. Proceed to Phase 2 (Bayesian Optimisation / EI) if N needs expanding")
