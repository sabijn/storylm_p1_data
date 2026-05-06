"""
Statistical analysis of LLM-as-a-judge ratings for Dutch children's stories.

pip install numpy pandas scipy pingouin krippendorff statsmodels
"""

import argparse
from pathlib import Path

import krippendorff
import numpy as np
import pandas as pd
import pingouin as pg
from scipy import stats
from statsmodels.stats.multitest import multipletests

# ── Configuration ──────────────────────────────────────────────────────────────

DATA_DIR = Path(__file__).parent

STORIES_CSV  = DATA_DIR / "questions_cleaned_and_sampled.csv"
JSON_CSV     = DATA_DIR / "llm_judge_results_concat.csv"
LOGPROB_CSV  = DATA_DIR / "llm_judge_logprobs.csv"
RESULTS_DIR  = DATA_DIR / "analysis_results"

CRITERIA = [
    "grammaticality",
    "coherence",
    "originality",
    "creativity",
    "complexity",
    "likeability",
    "humanlikeness",
]

# Map the two source labels to human-readable group names
SOURCE_MAP = {"chiscor": "real", "sftw": "generated"}


# ── Data loading ───────────────────────────────────────────────────────────────

def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    stories = pd.read_csv(STORIES_CSV, usecols=["id", "source"])
    stories["id"] = stories["id"].astype(str)
    stories["group"] = stories["source"].map(SOURCE_MAP)

    json_df = pd.read_csv(JSON_CSV)
    json_df["id"] = json_df["id"].astype(str)
    json_df[CRITERIA] = json_df[CRITERIA].apply(pd.to_numeric, errors="coerce")

    logprob_df = None
    if LOGPROB_CSV.exists():
        logprob_df = pd.read_csv(LOGPROB_CSV)
        logprob_df["id"] = logprob_df["id"].astype(str)
        for col in ["greedy_score", "expected_score", "entropy"] + [f"p{t}" for t in range(1, 6)]:
            logprob_df[col] = pd.to_numeric(logprob_df[col], errors="coerce")

    return stories, json_df, logprob_df


# ── 1. Descriptive statistics ──────────────────────────────────────────────────

def descriptive_stats(json_df: pd.DataFrame, logprob_df: pd.DataFrame | None,
                      stories: pd.DataFrame) -> pd.DataFrame:
    """
    Median + IQR for discrete scores; mean ± SD for continuous expected scores.
    Reported overall and split by source group.
    """
    df = json_df.merge(stories[["id", "group"]], on="id")
    groups = ["overall"] + sorted(df["group"].dropna().unique().tolist())

    rows = []
    for criterion in CRITERIA:
        row = {"criterion": criterion}
        for grp in groups:
            sub = df[criterion] if grp == "overall" else df.loc[df["group"] == grp, criterion]
            sub = sub.dropna()
            q25, q75 = sub.quantile(0.25), sub.quantile(0.75)
            row[f"{grp}_median"] = round(sub.median(), 2)
            row[f"{grp}_IQR"]    = f"{q25:.1f}–{q75:.1f}"

        if logprob_df is not None:
            lp = logprob_df[logprob_df["criterion"] == criterion]["expected_score"].dropna()
            row["expected_mean"] = round(lp.mean(), 3)
            row["expected_SD"]   = round(lp.std(), 3)

        rows.append(row)

    return pd.DataFrame(rows).set_index("criterion")


# ── 2. Inter-rater reliability ─────────────────────────────────────────────────

def interrater_reliability(json_df: pd.DataFrame,
                           logprob_df: pd.DataFrame | None) -> pd.DataFrame:
    """
    Krippendorff's α (ordinal) on discrete 1–5 scores.
    ICC(2,k) on continuous expected scores (if available).
    """
    rows = []
    for criterion in CRITERIA:
        row = {"criterion": criterion}

        # Krippendorff α — matrix is raters × items
        pivot = json_df.pivot(index="id", columns="judge", values=criterion)
        matrix = pivot.T.to_numpy(dtype=float)  # shape: (n_raters, n_stories)
        try:
            alpha = krippendorff.alpha(
                reliability_data=matrix,
                level_of_measurement="ordinal",
            )
            row["Krippendorff α (ordinal)"] = round(alpha, 3)
        except Exception as exc:
            row["Krippendorff α (ordinal)"] = f"ERR: {exc}"

        # ICC(2,k) on expected scores
        row["ICC(2,k)"] = "–"
        if logprob_df is not None:
            lp = (
                logprob_df[logprob_df["criterion"] == criterion]
                [["id", "judge", "expected_score"]]
                .dropna()
            )
            if len(lp) >= 6:  # need at least 2 targets × 3 raters
                try:
                    icc = pg.intraclass_corr(
                        data=lp,
                        targets="id",
                        raters="judge",
                        ratings="expected_score",
                    )
                    val = icc.set_index("Type").loc["ICC2k", "ICC"]
                    row["ICC(2,k)"] = round(val, 3)
                except Exception as exc:
                    row["ICC(2,k)"] = f"ERR: {exc}"

        rows.append(row)

    return pd.DataFrame(rows).set_index("criterion")


# ── 3. Real vs. generated ──────────────────────────────────────────────────────

def real_vs_generated(json_df: pd.DataFrame, stories: pd.DataFrame) -> pd.DataFrame:
    """
    Mann-Whitney U + rank-biserial r per criterion.
    Benjamini-Hochberg correction applied across the seven tests.
    """
    df = json_df.merge(stories[["id", "group"]], on="id")
    real = df[df["group"] == "real"]
    gen  = df[df["group"] == "generated"]

    rows = []
    p_values = []

    for criterion in CRITERIA:
        x = real[criterion].dropna().values
        y = gen[criterion].dropna().values
        u_stat, p_val = stats.mannwhitneyu(x, y, alternative="two-sided")
        n1, n2 = len(x), len(y)
        r_rb = 2 * u_stat / (n1 * n2) - 1  # rank-biserial correlation
        row = {
            "criterion": criterion,
            "real median":      round(float(np.median(x)), 2),
            "generated median": round(float(np.median(y)), 2),
            "U":    u_stat,
            "p":    round(p_val, 4),
            "r_rb": round(r_rb, 3),
        }
        rows.append(row)
        p_values.append(p_val)

    # BH correction
    _, p_adj, _, _ = multipletests(p_values, method="fdr_bh")
    for row, p in zip(rows, p_adj):
        row["p_adj (BH)"] = round(p, 4)

    return pd.DataFrame(rows).set_index("criterion")


# ── 4. JSON vs. logprobs consistency ──────────────────────────────────────────

def json_vs_logprobs(json_df: pd.DataFrame,
                     logprob_df: pd.DataFrame | None) -> pd.DataFrame | None:
    """
    Spearman ρ between the discrete JSON score and the continuous expected score,
    and between the greedy score and the expected score per criterion.
    """
    if logprob_df is None:
        print("Logprobs not available — skipping consistency analysis.")
        return None

    json_long = json_df.melt(
        id_vars=["id", "judge"],
        value_vars=CRITERIA,
        var_name="criterion",
        value_name="discrete_score",
    )

    merged = json_long.merge(
        logprob_df[["id", "judge", "criterion", "greedy_score", "expected_score"]],
        on=["id", "judge", "criterion"],
    )

    rows = []
    for criterion in CRITERIA:
        sub = merged[merged["criterion"] == criterion].dropna(
            subset=["discrete_score", "greedy_score", "expected_score"]
        )
        r1, p1 = stats.spearmanr(sub["discrete_score"], sub["expected_score"])
        r2, p2 = stats.spearmanr(sub["greedy_score"],   sub["expected_score"])
        rows.append({
            "criterion":                   criterion,
            "ρ discrete↔expected":         round(r1, 3),
            "p discrete↔expected":         round(p1, 4),
            "ρ greedy↔expected":           round(r2, 3),
            "p greedy↔expected":           round(p2, 4),
        })

    return pd.DataFrame(rows).set_index("criterion")


# ── 5. Entropy summary ─────────────────────────────────────────────────────────

def entropy_summary(logprob_df: pd.DataFrame | None) -> pd.DataFrame | None:
    """Mean entropy per criterion — high entropy = uncertain judge."""
    if logprob_df is None:
        return None

    agg = (
        logprob_df.groupby("criterion")["entropy"]
        .agg(mean="mean", sd="std")
        .round(3)
        .reindex(CRITERIA)
    )
    agg.columns = ["mean entropy", "SD entropy"]
    return agg


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Statistical analysis of LLM judge ratings.")
    parser.add_argument(
        "--logprobs",
        action="store_true",
        help="Include logprob-based statistics (sections 4 and 5, and ICC in section 2).",
    )
    args = parser.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)
    stories, json_df, logprob_df = load_data()

    if not args.logprobs:
        logprob_df = None

    sep = "\n" + "─" * 60 + "\n"

    print(sep + "1. DESCRIPTIVE STATISTICS")
    desc = descriptive_stats(json_df, logprob_df, stories)
    print(desc.to_string())
    desc.to_csv(RESULTS_DIR / "1_descriptive.csv")

    print(sep + "2. INTER-RATER RELIABILITY")
    rel = interrater_reliability(json_df, logprob_df)
    print(rel.to_string())
    rel.to_csv(RESULTS_DIR / "2_interrater.csv")

    print(sep + "3. REAL vs. GENERATED  (Mann-Whitney U, BH-corrected)")
    rvg = real_vs_generated(json_df, stories)
    print(rvg.to_string())
    rvg.to_csv(RESULTS_DIR / "3_real_vs_generated.csv")

    if args.logprobs:
        print(sep + "4. JSON vs. LOGPROBS CONSISTENCY  (Spearman ρ)")
        jvl = json_vs_logprobs(json_df, logprob_df)
        if jvl is not None:
            print(jvl.to_string())
            jvl.to_csv(RESULTS_DIR / "4_json_vs_logprobs.csv")

        print(sep + "5. ENTROPY SUMMARY")
        ent = entropy_summary(logprob_df)
        if ent is not None:
            print(ent.to_string())
            ent.to_csv(RESULTS_DIR / "5_entropy.csv")

    print(f"\nAll tables saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
