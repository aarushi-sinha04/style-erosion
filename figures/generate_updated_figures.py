"""
Clean Publication Figures — Final Version
Saves: fig1_tradeoff_new.png, fig2_accuracy_bars_new.png,
       fig3_asr_comparison_new.png, fig4_ablation_new.png
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

OUT = "figures"

# ── Data (updated values) ────────────────────────────────────────────────
MODEL_ACC = {
    "LogReg":        {"PAN22": 62.8, "Blog": 50.0, "Enron": 50.0, "avg": 54.3},
    "Base DANN":     {"PAN22": 53.2, "Blog": 55.8, "Enron": 78.8, "avg": 62.6},
    "Robust DANN":   {"PAN22": 54.4, "Blog": 52.8, "Enron": 74.0, "avg": 60.4},
    "PAN22 Siamese": {"PAN22": 97.0, "Blog": 52.1, "Enron": 56.8, "avg": 68.6},
    "CD Siamese":    {"PAN22": 98.2, "Blog": 66.5, "Enron": 77.2, "avg": 80.6},
    "Rob Siamese":   {"PAN22": 99.4, "Blog": 71.9, "Enron": 87.2, "avg": 86.2},
    "Ensemble":      {"PAN22": 98.0, "Blog": 64.4, "Enron": 76.8, "avg": 79.7},
    "BERT Siamese":  {"PAN22": 52.8, "Blog": 65.6, "Enron": 81.1, "avg": 66.48},
}

ASR = {
    "Rob Siamese":   80.2,
    "CD Siamese":    47.8,
    "PAN22 Siamese": 50.0,
    "Ensemble":      48.0,
    "Base DANN":     14.3,
    "Robust DANN":    7.7,
    "LogReg":        10.8,
    "BERT Siamese":   5.4,
}

MULTI_ASR = {
    #                       syn    bt     t5
    "Robust DANN":   (0.8,  13.8,   7.7),
    "Base DANN":     (0.7,  11.3,  14.3),
    "LogReg":        (None, None,  10.8),
    "BERT Siamese":  (6.8,  10.3,   5.4),
    "CD Siamese":    (0.0,   4.0,  47.8),
    "Ensemble":      (0.0,  None,  48.0),
    "PAN22 Siamese": (0.0,  12.0,  50.0),
    "Rob Siamese":   (0.5,  19.0,  80.2),
}

COLORS = {
    "LogReg":        "#95a5a6",
    "Base DANN":     "#3498db",
    "Robust DANN":   "#2980b9",
    "PAN22 Siamese": "#e74c3c",
    "CD Siamese":    "#e67e22",
    "Rob Siamese":   "#27ae60",
    "Ensemble":      "#9b59b6",
    "BERT Siamese":  "#f39c12",
}

plt.rcParams.update({
    "font.family":     "serif",
    "font.size":       12,
    "axes.labelsize":  13,
    "axes.titlesize":  15,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 10,
    "figure.dpi":      300,
    "savefig.dpi":     300,
    "savefig.bbox":    "tight",
})


# ══════════════════════════════════════════════════════════════════════════
# FIG 1 — Prettier Accuracy–Robustness Trade-off
# ══════════════════════════════════════════════════════════════════════════
def fig1_tradeoff_new():
    fig, ax = plt.subplots(figsize=(13, 8))

    # Quadrant shading (dividers: acc=70, asr=35)
    ax.fill_between([44, 70], [35, 35], [92, 92], color="#e74c3c", alpha=0.04)   # top-left
    ax.fill_between([70, 93], [35, 35], [92, 92], color="#e74c3c", alpha=0.08)   # top-right (vulnerable)
    ax.fill_between([44, 70], [-3, -3], [35, 35], color="#3498db", alpha=0.04)   # bottom-left
    ax.fill_between([70, 93], [-3, -3], [35, 35], color="#27ae60", alpha=0.08)   # bottom-right (ideal)

    ax.axhline(y=35, color="#bdc3c7", linestyle="--", linewidth=1, alpha=0.7)
    ax.axvline(x=70, color="#bdc3c7", linestyle="--", linewidth=1, alpha=0.7)

    # Quadrant labels — placed at corners, away from data points
    ax.text(45,  89, "Low Acc / High ASR",    ha="left",   fontsize=8.5, color="#c0392b", fontstyle="italic", alpha=0.65)
    ax.text(92,  89, "High Acc / High ASR",   ha="right",  fontsize=8.5, color="#c0392b", fontstyle="italic", alpha=0.65)
    ax.text(45,  -2, "Low Acc / Low ASR",     ha="left",   fontsize=8.5, color="#7f8c8d", fontstyle="italic", alpha=0.65)
    ax.text(92,  -2, "High Acc / Low ASR (Ideal)", ha="right", fontsize=8.5, color="#27ae60", fontweight="bold", fontstyle="italic", alpha=0.85)

    models = ["LogReg", "Base DANN", "Robust DANN", "PAN22 Siamese",
              "CD Siamese", "Rob Siamese", "Ensemble", "BERT Siamese"]

    feature_types = {
        "LogReg": "char", "PAN22 Siamese": "char",
        "CD Siamese": "char", "Rob Siamese": "char",
        "Base DANN": "multi", "Robust DANN": "multi",
        "Ensemble": "hybrid", "BERT Siamese": "contextual",
    }
    markers = {"char": "o", "multi": "s", "hybrid": "D", "contextual": "^"}
    sizes   = {"char": 200, "multi": 200, "hybrid": 220, "contextual": 200}

    accs = [MODEL_ACC[m]["avg"] for m in models]
    asrs = [ASR[m]              for m in models]

    for i, m in enumerate(models):
        ft = feature_types[m]
        ax.scatter(accs[i], asrs[i], c=COLORS[m],
                   marker=markers[ft], s=sizes[ft],
                   edgecolors="white", linewidths=1.2, zorder=5)

    # ── Label positions (data coords) — carefully spaced, no overlap ──
    label_xy = {
        "LogReg":        (46.0, 17.5),   # left of point
        "Robust DANN":   (52.5,  1.0),   # below-left
        "BERT Siamese":  (71.0, 12.0),   # right
        "Base DANN":     (67.5, 21.5),   # upper-right
        "PAN22 Siamese": (58.0, 56.5),   # upper-left
        "CD Siamese":    (76.0, 57.5),   # upper (CD & Ensemble separated by y)
        "Ensemble":      (84.5, 41.0),   # lower-right of CD
        "Rob Siamese":   (76.5, 72.5),   # left of point, below quadrant text
    }

    for m in models:
        xi, yi = MODEL_ACC[m]["avg"], ASR[m]
        xt, yt = label_xy[m]
        ax.annotate(
            m,
            xy=(xi, yi), xytext=(xt, yt),
            fontsize=9.5, fontweight="bold", color="#2c3e50",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.85),
            arrowprops=dict(arrowstyle="-", color="#aab7b8", lw=0.9),
            zorder=6,
        )

    # Legend
    legend_elements = [
        plt.Line2D([0],[0], marker="o", color="w", markerfacecolor="#555", markersize=9, label="Character n-grams"),
        plt.Line2D([0],[0], marker="s", color="w", markerfacecolor="#555", markersize=9, label="Multi-view syntactic"),
        plt.Line2D([0],[0], marker="D", color="w", markerfacecolor="#555", markersize=9, label="Hybrid ensemble"),
        plt.Line2D([0],[0], marker="^", color="w", markerfacecolor="#555", markersize=9, label="Contextual (BERT)"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", framealpha=0.92,
              edgecolor="#ddd", fontsize=9.5)

    ax.set_xlabel("Average Cross-Domain Accuracy (%)", fontweight="bold", labelpad=8)
    ax.set_ylabel("T5 Paraphrase Attack Success Rate (%) — lower is better", fontweight="bold", labelpad=8)
    ax.set_title("Accuracy–Robustness Trade-off by Feature Type", fontweight="bold", pad=14)
    ax.set_xlim(44, 93)
    ax.set_ylim(-3, 92)
    ax.grid(True, alpha=0.15, linewidth=0.7)
    ax.set_facecolor("#fdfdfd")
    fig.patch.set_facecolor("white")

    plt.tight_layout()
    plt.savefig(f"{OUT}/fig1_tradeoff_new.png")
    plt.close()
    print("  fig1_tradeoff_new.png")


# ══════════════════════════════════════════════════════════════════════════
# FIG 2 — Cross-Domain Accuracy Grouped Bar Chart
# ══════════════════════════════════════════════════════════════════════════
def fig2_accuracy_bars_new():
    fig, ax = plt.subplots(figsize=(14, 6.5))

    models  = ["LogReg", "BERT Siamese", "Base DANN", "Robust DANN",
               "PAN22 Siamese", "CD Siamese", "Rob Siamese", "Ensemble"]
    domains = ["PAN22", "Blog", "Enron"]

    data = {m: [MODEL_ACC[m][d] for d in domains] for m in models}

    x       = np.arange(len(domains))
    n       = len(models)
    width   = 0.10
    offsets = np.arange(n) - (n - 1) / 2.0

    for i, m in enumerate(models):
        ax.bar(x + offsets[i] * width, data[m], width,
               label=m, color=COLORS[m], edgecolor="white", linewidth=0.5)

    # Label only bars >= 85% to avoid clutter
    for i, m in enumerate(models):
        for j, v in enumerate(data[m]):
            if v >= 85:
                ax.text(x[j] + offsets[i] * width, v + 0.8, f"{v:.0f}",
                        ha="center", va="bottom", fontsize=7, fontweight="bold",
                        color="#2c3e50")

    ax.set_xlabel("Domain", fontweight="bold", labelpad=8)
    ax.set_ylabel("Accuracy (%)", fontweight="bold", labelpad=8)
    ax.set_title("Cross-Domain Authorship Verification Accuracy", fontweight="bold", pad=14)
    ax.set_xticks(x)
    ax.set_xticklabels(domains, fontweight="bold")
    ax.set_ylim(0, 112)
    ax.axhline(y=50, color="red", linestyle=":", alpha=0.35, linewidth=1, label="Random baseline")
    ax.legend(loc="upper right", ncol=2, framealpha=0.92, edgecolor="#ddd", fontsize=8.5)
    ax.grid(axis="y", alpha=0.18, linewidth=0.7)
    ax.set_facecolor("#fdfdfd")
    fig.patch.set_facecolor("white")

    plt.tight_layout()
    plt.savefig(f"{OUT}/fig2_accuracy_bars_new.png")
    plt.close()
    print("  fig2_accuracy_bars_new.png")


# ══════════════════════════════════════════════════════════════════════════
# FIG 3 — Multi-Attack ASR Heatmap
# ══════════════════════════════════════════════════════════════════════════
def fig3_asr_comparison_new():
    fig, ax = plt.subplots(figsize=(9, 6.5))

    models_order  = ["Robust DANN", "Base DANN", "LogReg", "BERT Siamese",
                     "CD Siamese", "Ensemble", "PAN22 Siamese", "Rob Siamese"]
    attack_labels = ["Synonym\nReplacement", "Back-\nTranslation", "T5\nParaphrase"]

    # y-tick labels include feature type in italics via two-line label
    ytick_labels = [
        "Robust DANN\n(syntactic)",
        "Base DANN\n(syntactic)",
        "LogReg\n(char 3-gram)",
        "BERT Siamese\n(contextual)",
        "CD Siamese\n(char 4-gram)",
        "Ensemble\n(hybrid)",
        "PAN22 Siamese\n(char 4-gram)",
        "Rob Siamese\n(char 4-gram)",
    ]

    matrix = np.array([list(MULTI_ASR[m]) for m in models_order], dtype=float)

    cmap = LinearSegmentedColormap.from_list("robustness", ["#27ae60", "#f1c40f", "#e74c3c"])
    im = ax.imshow(matrix, cmap=cmap, aspect="auto", vmin=0, vmax=85)

    for i in range(len(models_order)):
        for j in range(len(attack_labels)):
            val = matrix[i, j]
            if np.isnan(val):
                ax.text(j, i, "—", ha="center", va="center",
                        fontsize=13, color="gray", fontstyle="italic")
            else:
                color = "white" if val > 38 else "black"
                ax.text(j, i, f"{val:.1f}%", ha="center", va="center",
                        fontsize=11, fontweight="bold", color=color)

    ax.set_xticks(np.arange(len(attack_labels)))
    ax.set_xticklabels(attack_labels, fontweight="bold", fontsize=11)
    ax.set_yticks(np.arange(len(models_order)))
    ax.set_yticklabels(ytick_labels, fontsize=9.5)
    ax.set_title("Attack Success Rate (%) by Model and Attack Type",
                 fontweight="bold", pad=14)

    cbar = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("ASR (%) — lower is better", fontweight="bold", fontsize=10)

    plt.tight_layout()
    plt.savefig(f"{OUT}/fig3_asr_comparison_new.png")
    plt.close()
    print("  fig3_asr_comparison_new.png")


# ══════════════════════════════════════════════════════════════════════════
# FIG 4 — Siamese Ablation Progression
# ══════════════════════════════════════════════════════════════════════════
def fig4_ablation_new():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    stages     = ["PAN22\nSiamese", "Cross-Domain\nSiamese", "Robust\nSiamese"]
    accs_pan   = [97.0, 98.2, 99.4]
    accs_blog  = [52.1, 66.5, 71.9]
    accs_enron = [56.8, 77.2, 87.2]
    asrs       = [50.0, 47.8, 80.2]
    x = np.arange(len(stages))

    # Left: accuracy progression
    ax1.plot(x, accs_pan,   "o-", color="#e74c3c", lw=2.5, ms=9, label="PAN22", zorder=4)
    ax1.plot(x, accs_blog,  "s-", color="#3498db", lw=2.5, ms=9, label="Blog",  zorder=4)
    ax1.plot(x, accs_enron, "D-", color="#27ae60", lw=2.5, ms=9, label="Enron", zorder=4)

    # Value labels — offset to avoid overlap
    pan_offsets  = [(0, 2.5), (0, 2.5), (0, 2.5)]
    blog_offsets = [(0, 2.5), (0, 2.5), (0, 2.5)]
    enr_offsets  = [(0, -6),  (0, 2.5), (0, 2.5)]   # stage 0: below to avoid blog

    for i in range(3):
        ax1.annotate(f"{accs_pan[i]}%",   (x[i], accs_pan[i]),
                     xytext=(x[i]+pan_offsets[i][0], accs_pan[i]+pan_offsets[i][1]),
                     ha="center", fontsize=8.5, color="#e74c3c", fontweight="bold")
        ax1.annotate(f"{accs_blog[i]}%",  (x[i], accs_blog[i]),
                     xytext=(x[i]+blog_offsets[i][0], accs_blog[i]+blog_offsets[i][1]),
                     ha="center", fontsize=8.5, color="#3498db", fontweight="bold")
        ax1.annotate(f"{accs_enron[i]}%", (x[i], accs_enron[i]),
                     xytext=(x[i]+enr_offsets[i][0], accs_enron[i]+enr_offsets[i][1]),
                     ha="center", fontsize=8.5, color="#27ae60", fontweight="bold")

    # Blog +14.4 pp arrow
    ax1.annotate("", xy=(1, 66.5), xytext=(0, 52.1),
                 arrowprops=dict(arrowstyle="->", color="#3498db", lw=1.4, ls="--"))
    ax1.text(0.5, 56, "+14.4 pp", ha="center", fontsize=8, color="#3498db")

    ax1.set_xticks(x)
    ax1.set_xticklabels(stages, fontweight="bold")
    ax1.set_ylabel("Accuracy (%)", fontweight="bold")
    ax1.set_title("(a) Accuracy Progression", fontweight="bold")
    ax1.set_ylim(40, 110)
    ax1.legend(loc="lower right", framealpha=0.92, edgecolor="#ddd")
    ax1.grid(alpha=0.18)
    ax1.set_facecolor("#fdfdfd")

    # Right: ASR bars
    bar_colors = ["#e74c3c", "#e67e22", "#27ae60"]
    ax2.bar(x, asrs, color=bar_colors, edgecolor="white", width=0.5, zorder=3)
    for i, v in enumerate(asrs):
        ax2.text(i, v + 2, f"{v:.0f}%", ha="center", fontweight="bold", fontsize=13, color="#2c3e50")

    ax2.annotate("Adversarial training\nparadox: +32.4 pp",
                 xy=(2, 80.2), xytext=(0.7, 87),
                 arrowprops=dict(arrowstyle="->", color="red", lw=1.5),
                 fontsize=9, color="red", fontweight="bold", ha="center")

    ax2.set_xticks(x)
    ax2.set_xticklabels(stages, fontweight="bold")
    ax2.set_ylabel("Attack Success Rate (%)", fontweight="bold")
    ax2.set_title("(b) Robustness — T5 Paraphrase ASR", fontweight="bold")
    ax2.set_ylim(0, 100)
    ax2.axhline(y=35, color="red", linestyle="--", alpha=0.35, linewidth=1)
    ax2.grid(axis="y", alpha=0.18)
    ax2.set_facecolor("#fdfdfd")

    plt.suptitle("Siamese Ablation: Single-Domain → Cross-Domain → Adversarial Training",
                 fontweight="bold", fontsize=13)
    fig.patch.set_facecolor("white")
    plt.tight_layout()
    plt.savefig(f"{OUT}/fig4_ablation_new.png")
    plt.close()
    print("  fig4_ablation_new.png")


if __name__ == "__main__":
    print("Generating clean publication figures...\n")
    fig1_tradeoff_new()
    fig2_accuracy_bars_new()
    fig3_asr_comparison_new()
    fig4_ablation_new()
    print(f"\nDone. Saved to {OUT}/")
