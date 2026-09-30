"""Compare LLM-as-judge scores against human rater scores.

Aligns the two sources at the (report_id, criterion) level using the
survey-slot -> report_id mapping in config.yaml, then reports:
  1. Overall Report x Criterion correlation (Pearson + Spearman, n = 10 reports x 9 criteria)
  2. Per-criterion correlation (n = 10 reports each)
  3. Template vs GPT-4 correlation, computed separately within each system
     (n = 5 reports x 9 criteria each) - does LLM-human agreement hold up
     the same way for template-generated and GPT-4-generated reports, or is
     it driven by one system?

Human score per (report_id, criterion) = mean across the raters who scored
that report (each report is rated by all 4 raters of whichever survey it
appears in, i.e. n=4 per report here, not 2 - each survey's 4 raters all
score all 5 of that survey's reports).

Usage:
    python compare_with_humans.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import pearsonr, spearmanr

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent


def load_config() -> dict:
    with open(SCRIPT_DIR / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def preprocess_value(value):
    try:
        return int(str(value).split(" :")[0])
    except ValueError:
        return value


def load_survey_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df[df["submitdate. Date de soumission"].notna()]
    cols = [
        c for c in df.columns
        if c.startswith("R") and "[comment]" not in c and "Time" not in c and "Q10" not in c
    ]
    extracted = df.loc[:, cols].copy()
    for c in extracted.columns:
        extracted[c] = extracted[c].apply(preprocess_value)
    extracted.rename(columns=lambda x: x.split(".")[0], inplace=True)
    return extracted


def human_scores_long(config: dict) -> pd.DataFrame:
    """Long-format: one row per (report_id, criterion, rater), from both surveys."""
    criteria_ids = [c["id"] for c in config["criteria"]]
    records = []

    for survey_name, survey_cfg in config["surveys"].items():
        csv_path = REPO_ROOT / survey_cfg["csv"]
        df = load_survey_csv(csv_path)
        slots = survey_cfg["slots"]  # e.g. {"R1": "T1", "R2": "G2", ...}

        for slot, report_id in slots.items():
            for qid in criteria_ids:
                col = f"{slot}{qid}"
                if col not in df.columns:
                    continue
                for rater_idx, val in enumerate(df[col].values):
                    records.append({
                        "report_id": report_id,
                        "criterion": qid,
                        "survey": survey_name,
                        "rater_idx": rater_idx,
                        "human_score": val,
                    })

    return pd.DataFrame(records)


def human_scores_mean(config: dict) -> pd.DataFrame:
    long_df = human_scores_long(config)
    return (
        long_df.groupby(["report_id", "criterion"])["human_score"]
        .mean()
        .reset_index()
    )


def load_llm_scores(output_dir: Path) -> pd.DataFrame:
    """Loads the pointwise LLM scores (one row per model/report/criterion/run)
    written by run_judge.py, and averages over runs (n_repeats). The
    free-text `comment` (when n_repeats=1) is carried through unaggregated
    for manual auditing - with >1 repeats, comments are joined with " | "."""
    csv_path = output_dir / "llm_scores_pointwise.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path} not found — run `python run_judge.py` first."
        )
    df = pd.read_csv(csv_path)
    # Exclude "deepseek-chat": DeepSeek's API now silently redirects this model
    # id to "deepseek-flash", so data logged under this name may span two
    # different underlying models depending on when it was scored (see
    # config.yaml / FINDINGS.md). Kept in sync with eval_results.ipynb and
    # llm_judge_iaa.py, which apply the same exclusion.
    df = df[df["model"] != "deepseek-chat"]
    has_comment = "comment" in df.columns

    group_cols = ["model", "report_id", "system", "criterion"]
    scores = df.groupby(group_cols)["score"].mean().rename("llm_score")

    if has_comment:
        comments = (
            df.groupby(group_cols)["comment"]
            .apply(lambda s: " | ".join(str(c) for c in s.dropna()))
            .rename("llm_comment")
        )
        return pd.concat([scores, comments], axis=1).reset_index()

    return scores.reset_index()


def safe_corr(sub: pd.DataFrame):
    """Pearson + Spearman, returning NaNs when the sample is degenerate
    (n<3, or either side is constant so correlation is undefined)."""
    if len(sub) < 3 or sub["human_score"].std() == 0 or sub["llm_score"].std() == 0:
        return np.nan, np.nan, np.nan, np.nan
    r_p, p_p = pearsonr(sub["human_score"], sub["llm_score"])
    r_s, p_s = spearmanr(sub["human_score"], sub["llm_score"])
    return r_p, p_p, r_s, p_s


def main():
    config = load_config()
    criteria_ids = [c["id"] for c in config["criteria"]]
    criteria_names = {c["id"]: c["name"] for c in config["criteria"]}
    output_dir = REPO_ROOT / config["output_dir"]

    human_df = human_scores_mean(config)
    llm_df = load_llm_scores(output_dir)

    models = llm_df["model"].unique()
    all_results = []

    for model in models:
        model_llm = llm_df[llm_df["model"] == model]
        merged = model_llm.merge(human_df, on=["report_id", "criterion"], how="inner")

        if merged.empty:
            print(f"[{model}] No overlapping (report_id, criterion) rows — check config mapping.")
            continue

        merged.to_csv(output_dir / f"merged_scores_{model}.csv", index=False)

        # ── 1. Overall Report x Criterion correlation ──────────────────────
        r_pearson, p_pearson, r_spearman, p_spearman = safe_corr(merged)
        mae = (merged["human_score"] - merged["llm_score"]).abs().mean()

        print(f"\n=== {model}: Overall Report x Criterion (n={len(merged)}) ===")
        print(f"  Pearson  r = {r_pearson:.3f} (p={p_pearson:.3f})")
        print(f"  Spearman r = {r_spearman:.3f} (p={p_spearman:.3f})")
        print(f"  MAE = {mae:.3f}")

        all_results.append({
            "model": model, "level": "overall", "group": "ALL", "criterion": "ALL",
            "n": len(merged), "pearson_r": r_pearson, "pearson_p": p_pearson,
            "spearman_r": r_spearman, "spearman_p": p_spearman, "mae": mae,
        })

        # ── 2. Per-criterion correlation ────────────────────────────────────
        print(f"\n=== {model}: Per-Criterion (n=10 reports each) ===")
        print(f"{'Criterion':<22} {'Pearson r':>10} {'Spearman r':>11} {'MAE':>6}")
        for qid in criteria_ids:
            sub = merged[merged["criterion"] == qid]
            r_p, p_p, r_s, p_s = safe_corr(sub)
            mae_q = (sub["human_score"] - sub["llm_score"]).abs().mean()

            print(f"{criteria_names[qid]:<22} {r_p:>10.3f} {r_s:>11.3f} {mae_q:>6.3f}")

            all_results.append({
                "model": model, "level": "per_criterion", "group": "ALL",
                "criterion": criteria_names[qid],
                "n": len(sub), "pearson_r": r_p, "pearson_p": p_p,
                "spearman_r": r_s, "spearman_p": p_s, "mae": mae_q,
            })

        # ── 3. Template vs GPT-4 correlation (split by generation system) ──
        # Same overall Report x Criterion correlation, computed separately
        # within each system (n=5 reports x 9 criteria = 45 each), to check
        # whether LLM-human agreement holds for both systems or is driven by
        # one of them.
        print(f"\n=== {model}: Template vs GPT-4 (n=45 reports x criteria each) ===")
        print(f"{'System':<10} {'n':>4} {'Pearson r':>10} {'Spearman r':>11} {'MAE':>6}")
        for system in ["Template", "GPT-4"]:
            sub = merged[merged["system"] == system]
            r_p, p_p, r_s, p_s = safe_corr(sub)
            mae_sys = (sub["human_score"] - sub["llm_score"]).abs().mean()

            print(f"{system:<10} {len(sub):>4} {r_p:>10.3f} {r_s:>11.3f} {mae_sys:>6.3f}")

            all_results.append({
                "model": model, "level": "system", "group": system, "criterion": "ALL",
                "n": len(sub), "pearson_r": r_p, "pearson_p": p_p,
                "spearman_r": r_s, "spearman_p": p_s, "mae": mae_sys,
            })

        # Also: does the LLM rank Template vs GPT-4 the same direction as
        # humans, per criterion? (mean score by system, both sources)
        print(f"\n=== {model}: Template vs GPT-4 mean score by criterion (direction check) ===")
        print(f"{'Criterion':<22} {'Human T':>8} {'Human G':>8} {'LLM T':>7} {'LLM G':>7} {'Human dir':>10} {'LLM dir':>8}")
        for qid in criteria_ids:
            sub = merged[merged["criterion"] == qid]
            h_t = sub[sub["system"] == "Template"]["human_score"].mean()
            h_g = sub[sub["system"] == "GPT-4"]["human_score"].mean()
            l_t = sub[sub["system"] == "Template"]["llm_score"].mean()
            l_g = sub[sub["system"] == "GPT-4"]["llm_score"].mean()
            human_dir = "Template" if h_t > h_g else ("GPT-4" if h_g > h_t else "tie")
            llm_dir = "Template" if l_t > l_g else ("GPT-4" if l_g > l_t else "tie")
            agree = "✓" if human_dir == llm_dir else "✗"

            print(f"{criteria_names[qid]:<22} {h_t:>8.2f} {h_g:>8.2f} {l_t:>7.2f} {l_g:>7.2f} "
                  f"{human_dir:>10} {llm_dir:>8} {agree}")

            all_results.append({
                "model": model, "level": "system_direction", "group": criteria_names[qid],
                "criterion": criteria_names[qid], "n": len(sub),
                "pearson_r": np.nan, "pearson_p": np.nan,
                "spearman_r": np.nan, "spearman_p": np.nan, "mae": np.nan,
                "human_template_mean": h_t, "human_gpt4_mean": h_g,
                "llm_template_mean": l_t, "llm_gpt4_mean": l_g,
                "human_direction": human_dir, "llm_direction": llm_dir,
                "directions_agree": human_dir == llm_dir,
            })

    results_df = pd.DataFrame(all_results)
    out_path = output_dir / "correlation_results.csv"
    results_df.to_csv(out_path, index=False)
    print(f"\nSaved correlation results -> {out_path}")


if __name__ == "__main__":
    main()
