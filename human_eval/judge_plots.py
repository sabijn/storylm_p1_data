"""
Distribution plots for LLM-as-a-judge ratings.
pip install matplotlib seaborn pandas
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

DATA_DIR    = Path(__file__).parent
JSON_CSV    = DATA_DIR / "llm_judge_results_concat.csv"
STORIES_CSV = DATA_DIR / "questions_cleaned_and_sampled.csv"
RESULTS_DIR = DATA_DIR / "analysis_results"

CRITERIA = [
    "grammaticality",
    "coherence",
    "originality",
    "creativity",
    "complexity",
    "likeability",
    "humanlikeness",
]

SOURCE_MAP = {"chiscor": "real", "sftw": "generated"}
PALETTE    = {"real": "#4C72B0", "generated": "#DD8452"}


def load_data() -> pd.DataFrame:
    stories = pd.read_csv(STORIES_CSV, usecols=["id", "source"])
    stories["id"] = stories["id"].astype(str)
    stories["group"] = stories["source"].map(SOURCE_MAP)

    df = pd.read_csv(JSON_CSV)
    df["id"] = df["id"].astype(str)
    df[CRITERIA] = df[CRITERIA].apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=CRITERIA)

    return df.merge(stories[["id", "group"]], on="id")


def plot_distributions(df: pd.DataFrame) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)

    long = df.melt(
        id_vars=["id", "judge", "group"],
        value_vars=CRITERIA,
        var_name="criterion",
        value_name="score",
    )

    handles = [
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=PALETTE[g], markersize=9, label=g.capitalize())
        for g in ["real", "generated"]
    ]

    for criterion in CRITERIA:
        data = long[long["criterion"] == criterion].dropna(subset=["score"])

        fig, ax = plt.subplots(figsize=(5, 5))

        # Strip plot first so box elements are drawn on top of points
        sns.stripplot(
            data=data, x="group", y="score", hue="group",
            palette=PALETTE, ax=ax,
            order=["real", "generated"],
            hue_order=["real", "generated"],
            size=3, alpha=0.35, jitter=True,
            legend=False,
        )

        # Box plot on top: shows median (thick line) + IQR box + whiskers
        sns.boxplot(
            data=data, x="group", y="score", hue="group",
            palette=PALETTE, ax=ax,
            order=["real", "generated"],
            hue_order=["real", "generated"],
            width=0.45, linewidth=1.8,
            showfliers=False,
            boxprops=dict(alpha=0.25),
            medianprops=dict(color="black", linewidth=2.5),
            whiskerprops=dict(linewidth=1.4),
            capprops=dict(linewidth=1.4),
            legend=False,
        )

        ax.set_title(criterion.capitalize(), fontsize=12, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("Score")
        ax.set_yticks([1, 2, 3, 4, 5])
        ax.set_ylim(0.5, 5.5)
        ax.tick_params(axis="x", labelsize=10)
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(handles=handles, fontsize=10, frameon=False)

        plt.tight_layout()

        out_path = RESULTS_DIR / f"distribution_{criterion}.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {out_path}")
        plt.show()
        plt.close(fig)


def main() -> None:
    df = load_data()
    plot_distributions(df)


if __name__ == "__main__":
    main()
