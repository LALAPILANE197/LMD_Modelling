# =============================================================================
# figures.py  —  Journal-grade validation figures for LMD MOGPR + NSGA-II study
# Run:  python figures.py
# Outputs: fig1_bar.png, fig2_scatter.png, fig3_ba.png, fig4_radar.png, fig5_heatmap.png
# Requires: numpy, pandas, matplotlib, seaborn  (pip install numpy pandas matplotlib seaborn)
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# 0.  GLOBAL STYLE  (Times New Roman + clean grid — classic journal look)
# ─────────────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":          "serif",
    "font.serif":           ["Times New Roman", "DejaVu Serif"],
    "font.size":            9,
    "axes.titlesize":       10,
    "axes.labelsize":       9,
    "xtick.labelsize":      8,
    "ytick.labelsize":      8,
    "legend.fontsize":      8.5,
    "legend.frameon":       False,
    "axes.spines.top":      False,
    "axes.spines.right":    False,
    "axes.linewidth":       0.8,
    "axes.grid":            True,
    "grid.linewidth":       0.4,
    "grid.color":           "#cccccc",
    "grid.linestyle":       "--",
    "lines.linewidth":      1.2,
    "figure.dpi":           150,     # screen preview
    "savefig.dpi":          600,     # journal-quality export
    "savefig.bbox":         "tight",
})

# ─────────────────────────────────────────────────────────────────────────────
# 1.  DATA
#     Replace these dictionaries with your own measurements if they change.
# ─────────────────────────────────────────────────────────────────────────────
targets = {
    "E1": {"H": 1000, "AR": 0.40, "D": 40.0},
    "E2": {"H":  440, "AR": 0.21, "D": 30.0},
    "E3": {"H": 1200, "AR": 0.46, "D": 35.0},
    "E4": {"H": 1600, "AR": 0.50, "D": 20.0},
}

wrought_raw = {
    "E1": [{"H": 819.674,  "AR": 0.290284, "D": 45.84},
           {"H": 856.678,  "AR": 0.306256, "D": 45.08}],
    "E2": [{"H": 295.951,  "AR": 0.171961, "D": 46.62},
           {"H": 256.260,  "AR": 0.170217, "D": 43.75}],
    "E3": [{"H": 1057.565, "AR": 0.374521, "D": 38.08},
           {"H":  979.534, "AR": 0.351734, "D": 39.66}],
    "E4": [{"H": 1231.936, "AR": 0.595913, "D": 18.82},
           {"H": 1296.882, "AR": 0.632720, "D": 18.05}],
}

cast_raw = {
    "E1": [{"H":  840.936, "AR": 0.272684, "D": 56.80},
           {"H":  888.255, "AR": 0.275859, "D": 56.02}],
    "E2": [{"H":  259.128, "AR": 0.103592, "D": 72.32},
           {"H":  253.787, "AR": 0.097163, "D": 72.73}],
    "E3": [{"H": 1004.699, "AR": 0.304957, "D": 54.55},
           {"H":  946.429, "AR": 0.279675, "D": 57.18}],
    "E4": [{"H": 1168.670, "AR": 0.430771, "D": 37.75},
           {"H": 1200.248, "AR": 0.418778, "D": 39.31}],
}

exps    = ["E1", "E2", "E3", "E4"]
metrics = [("H", "Bead height (µm)"), ("AR", "Aspect ratio (–)"), ("D", "Dilution (%)")]

# ─────────────────────────────────────────────────────────────────────────────
# 2.  HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────
def mn(raw, exp, resp):
    """Mean of duplicate measurements."""
    return np.mean([r[resp] for r in raw[exp]])

def sd(raw, exp, resp):
    """Std-dev of duplicate measurements (ddof=1)."""
    return np.std([r[resp] for r in raw[exp]], ddof=1)

def tgt(exp, resp):
    """NSGA-II / MOGPR target value."""
    return targets[exp][resp]

# ─────────────────────────────────────────────────────────────────────────────
# 3.  COLOURS  (colour-blind-friendly, print-safe)
# ─────────────────────────────────────────────────────────────────────────────
C_TGT = "#444444"   # dark grey  — hatched target bars
C_WR  = "#2166ac"   # blue       — Wrought 6xxx
C_CA  = "#d6604d"   # red-orange — Cast LM27

# ─────────────────────────────────────────────────────────────────────────────
# 4.  FIG 1  —  Grouped bar chart: Target vs Measured (H | AR | D)
# ─────────────────────────────────────────────────────────────────────────────
def plot_fig1():
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.4))
    fig.subplots_adjust(wspace=0.50, top=0.83)

    x, w = np.arange(len(exps)), 0.24

    for ax, (resp, ylabel) in zip(axes, metrics):
        t_vals = [tgt(e, resp)            for e in exps]
        w_mu   = [mn(wrought_raw, e, resp) for e in exps]
        w_sd_  = [sd(wrought_raw, e, resp) for e in exps]
        c_mu   = [mn(cast_raw, e, resp)    for e in exps]
        c_sd_  = [sd(cast_raw, e, resp)    for e in exps]

        # Target bars (hatched, no fill)
        ax.bar(x - w, t_vals, w, color="white", edgecolor=C_TGT,
               linewidth=1.1, hatch="///", zorder=3)
        # Wrought bars
        ax.bar(x, w_mu, w, yerr=w_sd_, color=C_WR, alpha=0.85,
               capsize=3, error_kw=dict(lw=0.9, ecolor="black"), zorder=3)
        # Cast bars
        ax.bar(x + w, c_mu, w, yerr=c_sd_, color=C_CA, alpha=0.85,
               capsize=3, error_kw=dict(lw=0.9, ecolor="black"), zorder=3)

        ax.set_ylabel(ylabel, labelpad=4)
        ax.set_xticks(x)
        ax.set_xticklabels(exps)   # <-- no "Experiment" x-axis label
        ax.yaxis.grid(True, zorder=0)
        ax.set_axisbelow(True)

    # Single legend centred above all panels
    p1 = mpatches.Patch(facecolor="white", edgecolor=C_TGT, hatch="///",
                         label="NSGA-II target")
    p2 = mpatches.Patch(facecolor=C_WR, edgecolor="none", alpha=0.85,
                         label="Wrought 6xxx")
    p3 = mpatches.Patch(facecolor=C_CA, edgecolor="none", alpha=0.85,
                         label="Cast LM27")
    fig.legend(handles=[p1, p2, p3], loc="upper center", ncol=3,
               bbox_to_anchor=(0.5, 1.00), frameon=False, fontsize=9)

    fig.savefig("fig1_bar.png")
    plt.close()
    print("  fig1_bar.png  saved.")


# ─────────────────────────────────────────────────────────────────────────────
# 5.  FIG 2  —  1:1 Agreement scatter (H | AR | D)
# ─────────────────────────────────────────────────────────────────────────────
def plot_fig2():
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.4))
    fig.subplots_adjust(wspace=0.46, top=0.83)

    for ax, (resp, label) in zip(axes, metrics):
        xs_w = [tgt(e, resp) for e in exps]
        ys_w = [mn(wrought_raw, e, resp) for e in exps]
        xs_c = [tgt(e, resp) for e in exps]
        ys_c = [mn(cast_raw, e, resp)    for e in exps]

        all_v = xs_w + ys_w + xs_c + ys_c
        lo = min(all_v) * 0.88
        hi = max(all_v) * 1.10
        g  = np.linspace(lo, hi, 100)

        # ±10 % shaded band
        ax.fill_between(g, 0.9 * g, 1.1 * g, color="#cccccc",
                         alpha=0.35, zorder=1, label="±10% band")
        # 1:1 line
        ax.plot(g, g, color="black", lw=1.2, ls="--", zorder=2, label="1:1 line")
        # Data points
        ax.scatter(xs_w, ys_w, marker="o", s=55,
                   facecolors="none", edgecolors=C_WR, linewidths=1.4,
                   zorder=4, label="Wrought 6xxx")
        ax.scatter(xs_c, ys_c, marker="s", s=55,
                   facecolors="none", edgecolors=C_CA, linewidths=1.4,
                   zorder=4, label="Cast LM27")

        # Experiment ID annotations (offset to avoid overlap)
        for i, e in enumerate(exps):
            ax.annotate(e, (xs_w[i], ys_w[i]),
                        textcoords="offset points", xytext=(5, 4),
                        fontsize=6.5, color=C_WR)
            ax.annotate(e, (xs_c[i], ys_c[i]),
                        textcoords="offset points", xytext=(5, 4),
                        fontsize=6.5, color=C_CA)

        ax.set_xlim(lo, hi);  ax.set_ylim(lo, hi)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(f"Target {label}", labelpad=4)
        ax.set_ylabel(f"Measured {label}", labelpad=4)
        ax.yaxis.grid(True, zorder=0);  ax.xaxis.grid(True, zorder=0)
        ax.set_axisbelow(True)

    # Shared legend above
    l1 = mlines.Line2D([], [], color="black", lw=1.2, ls="--",  label="1:1 line")
    l2 = mpatches.Patch(facecolor="#cccccc", alpha=0.55,         label="±10% band")
    l3 = mlines.Line2D([], [], marker="o", lw=0, mfc="none",
                        mec=C_WR, mew=1.4, ms=6,                 label="Wrought 6xxx")
    l4 = mlines.Line2D([], [], marker="s", lw=0, mfc="none",
                        mec=C_CA, mew=1.4, ms=6,                 label="Cast LM27")
    fig.legend(handles=[l1, l2, l3, l4], loc="upper center", ncol=4,
               bbox_to_anchor=(0.5, 1.00), frameon=False, fontsize=9)

    fig.savefig("fig2_scatter.png")
    plt.close()
    print("  fig2_scatter.png  saved.")


# ─────────────────────────────────────────────────────────────────────────────
# 6.  FIG 3  —  Bland-Altman (H | AR | D) + summary table
# ─────────────────────────────────────────────────────────────────────────────
def plot_fig3():
    # Build per-clad DataFrame
    rows = []
    for e in exps:
        for i in range(2):
            rows.append({"Exp": e, "Mat": "Wrought", **wrought_raw[e][i]})
            rows.append({"Exp": e, "Mat": "Cast",    **cast_raw[e][i]})
    df_clad = pd.DataFrame(rows)

    fig = plt.figure(figsize=(10, 5.0))
    gs  = gridspec.GridSpec(2, 3, figure=fig,
                            height_ratios=[3.2, 1.0],
                            hspace=0.10, wspace=0.52)
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    tax  = fig.add_subplot(gs[1, :])

    ba_stats = {}  # store (bias, loa_l, loa_u) per response x material

    for ax, (resp, label) in zip(axes, metrics):
        ba_stats[resp] = {}
        for mat, raw, col, sym in [("Wrought", wrought_raw, C_WR, "o"),
                                    ("Cast",    cast_raw,   C_CA, "s")]:
            sub = df_clad[df_clad["Mat"] == mat].copy()
            sub["tgt_v"] = sub["Exp"].map(lambda e: tgt(e, resp))
            sub["diff"]  = sub[resp] - sub["tgt_v"]
            sub["avg"]   = 0.5 * (sub[resp] + sub["tgt_v"])

            bias  = sub["diff"].mean()
            loa_u = bias + 1.96 * sub["diff"].std(ddof=1)
            loa_l = bias - 1.96 * sub["diff"].std(ddof=1)
            ba_stats[resp][mat] = (bias, loa_l, loa_u)

            xmin = sub["avg"].min() * 0.92
            xmax = sub["avg"].max() * 1.05

            ax.scatter(sub["avg"], sub["diff"],
                       marker=sym, s=50, facecolors="none",
                       edgecolors=col, linewidths=1.3, zorder=4)
            ax.hlines(bias,  xmin, xmax, colors=col, lw=1.3, ls="--", zorder=3)
            ax.hlines(loa_u, xmin, xmax, colors=col, lw=0.9, ls=":",  zorder=3)
            ax.hlines(loa_l, xmin, xmax, colors=col, lw=0.9, ls=":",  zorder=3)

        ax.axhline(0, color="black", lw=0.7, zorder=2)
        ax.set_xlabel(f"Mean — {label}", labelpad=3, fontsize=8.5)
        # Only the leftmost panel gets a y-axis label
        if label == "Bead height (µm)":
            ax.set_ylabel("Measured − target", labelpad=4, fontsize=8.5)
        ax.yaxis.grid(True, zorder=0);  ax.xaxis.grid(True, zorder=0)
        ax.set_axisbelow(True)

    # ── Summary statistics table (below the three panels) ────────────────────
    tax.axis("off")
    col_labs = ["",
                "Height\nBias", "Height\n±1.96σ LoA",
                "AR\nBias",     "AR\n±1.96σ LoA",
                "Dilution\nBias","Dilution\n±1.96σ LoA"]
    tab_data = []
    for mat_key, mat_label in [("Wrought", "Wrought 6xxx"),
                                ("Cast",   "Cast LM27")]:
        row = [mat_label]
        for resp in ["H", "AR", "D"]:
            b, lo_v, hi_v = ba_stats[resp][mat_key]
            row.append(f"{b:+.2f}")
            row.append(f"[{lo_v:+.2f}, {hi_v:+.2f}]")
        tab_data.append(row)

    tbl = tax.table(cellText=tab_data, colLabels=col_labs,
                    cellLoc="center", loc="center",
                    bbox=[0.0, -0.30, 1.0, 1.55])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7.8)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#cccccc")
        cell.set_facecolor("#eeeeee" if r == 0 else "white")

    # Shared legend above
    l3 = mlines.Line2D([], [], marker="o", lw=0, mfc="none",
                        mec=C_WR, mew=1.3, ms=6,       label="Wrought 6xxx")
    l4 = mlines.Line2D([], [], marker="s", lw=0, mfc="none",
                        mec=C_CA, mew=1.3, ms=6,       label="Cast LM27")
    l5 = mlines.Line2D([], [], color="gray", lw=1.3, ls="--", label="Bias (––)")
    l6 = mlines.Line2D([], [], color="gray", lw=0.9, ls=":",  label="±1.96σ LoA (···)")
    fig.legend(handles=[l3, l4, l5, l6], loc="upper center", ncol=4,
               bbox_to_anchor=(0.5, 1.01), frameon=False, fontsize=9)

    fig.savefig("fig3_ba.png")
    plt.close()
    print("  fig3_ba.png  saved.")


# ─────────────────────────────────────────────────────────────────────────────
# 7.  FIG 4  —  MAPE radar (Wrought vs Cast)
# ─────────────────────────────────────────────────────────────────────────────
def plot_fig4():
    resp_labels = ["Height", "Aspect ratio", "Dilution"]
    mape_wr, mape_ca = [], []
    for resp in ["H", "AR", "D"]:
        wr_errs = [abs(mn(wrought_raw, e, resp) - tgt(e, resp)) / tgt(e, resp) * 100
                   for e in exps]
        ca_errs = [abs(mn(cast_raw, e, resp)    - tgt(e, resp)) / tgt(e, resp) * 100
                   for e in exps]
        mape_wr.append(np.mean(wr_errs))
        mape_ca.append(np.mean(ca_errs))

    N      = len(resp_labels)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles_c = angles + [angles[0]]

    fig, ax = plt.subplots(figsize=(4.2, 4.2), subplot_kw=dict(polar=True))

    ax.plot(angles_c, mape_wr + [mape_wr[0]], color=C_WR, lw=1.8,
            marker="o", ms=6, label="Wrought 6xxx")
    ax.fill(angles_c, mape_wr + [mape_wr[0]], color=C_WR, alpha=0.15)
    ax.plot(angles_c, mape_ca + [mape_ca[0]], color=C_CA, lw=1.8,
            marker="s", ms=6, label="Cast LM27")
    ax.fill(angles_c, mape_ca + [mape_ca[0]], color=C_CA, alpha=0.15)

    ax.set_xticks(angles)
    ax.set_xticklabels(resp_labels, fontsize=9)
    max_v = max(max(mape_wr), max(mape_ca))
    ax.set_ylim(0, max_v * 1.30)
    r_ticks = [t for t in [10, 20, 30, 40, 50, 60, 70, 80, 90]
               if t < max_v * 1.30]
    ax.set_yticks(r_ticks)
    ax.set_yticklabels([f"{t}%" for t in r_ticks], fontsize=6.5)
    ax.grid(color="#bbbbbb", lw=0.5, ls=":")
    ax.spines["polar"].set_linewidth(0.6)

    # MAPE value annotations just outside each vertex
    offset = max_v * 0.12
    for i, angle in enumerate(angles):
        ax.text(angle, mape_wr[i] + offset,
                f"{mape_wr[i]:.1f}%", fontsize=7.5,
                ha="center", va="center", color=C_WR, fontweight="bold")
        ax.text(angle, mape_ca[i] + offset * 1.9,
                f"{mape_ca[i]:.1f}%", fontsize=7.5,
                ha="center", va="center", color=C_CA, fontweight="bold")

    ax.legend(loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.22),
              frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig("fig4_radar.png")
    plt.close()
    print("  fig4_radar.png  saved.")


# ─────────────────────────────────────────────────────────────────────────────
# 8.  FIG 5  —  CV heatmap (repeatability)
# ─────────────────────────────────────────────────────────────────────────────
def plot_fig5():
    row_labels_short = {
        ("Wrought", "H"):  "Wr – Height",
        ("Wrought", "AR"): "Wr – AR",
        ("Wrought", "D"):  "Wr – Dilution",
        ("Cast",    "H"):  "Cast – Height",
        ("Cast",    "AR"): "Cast – AR",
        ("Cast",    "D"):  "Cast – Dilution",
    }

    cv_rows = []
    for mat, raw in [("Wrought", wrought_raw), ("Cast", cast_raw)]:
        for resp in ["H", "AR", "D"]:
            row = {"row_lbl": row_labels_short[(mat, resp)]}
            for e in exps:
                mu = mn(raw, e, resp)
                s  = sd(raw, e, resp)
                row[e] = 100 * s / mu if mu else np.nan
            cv_rows.append(row)

    df_cv = pd.DataFrame(cv_rows).set_index("row_lbl")

    fig, ax = plt.subplots(figsize=(6.0, 3.5))
    fig.subplots_adjust(left=0.22, bottom=0.14, right=0.92, top=0.92)

    sns.heatmap(df_cv, annot=True, fmt=".2f", cmap="RdYlGn_r",
                vmin=0, vmax=12, linewidths=0.6, linecolor="white",
                annot_kws={"size": 9},
                cbar_kws={"label": "CV (%)", "shrink": 0.9, "pad": 0.02},
                ax=ax)
    ax.set_xlabel("Experiment", labelpad=6, fontsize=9)
    ax.set_ylabel("")
    ax.tick_params(axis="x", labelsize=9)
    ax.tick_params(axis="y", labelsize=9, labelrotation=0)
    ax.text(4.08, 6.5,
            "CV < 6 % → acceptable\nrepeatability (green)",
            fontsize=7.5, color="#444444", va="center")

    fig.savefig("fig5_heatmap.png")
    plt.close()
    print("  fig5_heatmap.png  saved.")


# ─────────────────────────────────────────────────────────────────────────────
# 9.  RUN ALL
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\nGenerating figures …")
    plot_fig1()
    plot_fig2()
    plot_fig3()
    plot_fig4()
    plot_fig5()
    print("\nDone — five PNG files saved in the current working directory.\n")