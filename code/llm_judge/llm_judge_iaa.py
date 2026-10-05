"""Inter-judge agreement (IAA) among LLM judges, compared against human IAA.

Motivation: alongside LLM-as-judge, it's worth checking whether independent
LLM judges agree with EACH OTHER more than human raters agree with each
other. Two readings of a high LLM-LLM agreement result are possible and
worth distinguishing in the writeup:
  - LLMs converging on a genuine, checkable signal humans under-weight, or
  - LLMs sharing training biases that produce spuriously high agreement
    without tracking the actual construct (the same risk flagged in the
    paper's own text: "higher LLM agreement on stylistic dimensions would
    suggest shared training biases rather than genuine consensus").

This script treats each configured judge model as one "rater" (same role
human raters played in eval_results.ipynb) and computes Krippendorff's alpha
(ordinal) across judges, overall and per criterion, using the same
methodology as the human IAA analysis - so the two numbers are directly
comparable.

Requires >=3 judge models to have been scored (run_judge.py, one row per
model/report/criterion in output/llm_scores_pointwise.csv) for a
meaningful multi-rater alpha (2 raters alone gives a valid but
less-informative alpha).

Usage:
    python llm_judge_iaa.py
"""
from __future__ import annotations

from pathlib import Path

import krippendorff
import numpy as np
import pandas as pd
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent


def load_config() -> dict:
    with open(SCRIPT_DIR / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_pointwise_scores(output_dir: Path) -> pd.DataFrame:
    csv_path = output_dir / "llm_scores_pointwise.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"{csv_path} not found — run `python run_judge.py` first.")
    df = pd.read_csv(csv_path)
    # Exclude "deepseek-chat": DeepSeek's API now silently redirects this model
    # id to "deepseek-flash", so data logged under this name may span two
    # different underlying models depending on when it was scored (see
    # config.yaml / FINDINGS.md). Kept in sync with eval_results.ipynb and
    # compare_with_humans.py, which apply the same exclusion.
    df = df[df["model"] != "deepseek-chat"]
    # mean across runs (n_repeats) per (model, report_id, criterion)
    return (
        df.groupby(["model", "report_id", "criterion"])["score"]
        .mean()
        .reset_index()
    )


def alpha_for_criterion(df: pd.DataFrame, criterion: str, report_ids: list[str]) -> float:
    """Build a (n_judges x n_reports) reliability matrix for one criterion
    and compute Krippendorff's alpha (ordinal) — same call signature as
    eval_results.ipynb's compute_alpha_criterion, so results are comparable."""
    sub = df[df["criterion"] == criterion]
    models = sorted(sub["model"].unique())
    matrix = np.full((len(models), len(report_ids)), np.nan)
    for i, model in enumerate(models):
        model_scores = sub[sub["model"] == model].set_index("report_id")["score"]
        for j, rid in enumerate(report_ids):
            if rid in model_scores.index:
                matrix[i, j] = model_scores.loc[rid]
    try:
        return krippendorff.alpha(reliability_data=matrix, level_of_measurement="ordinal")
    except (AssertionError, ValueError):
        return np.nan


def main():
    config = load_config()
    output_dir = REPO_ROOT / config["output_dir"]
    criteria_ids = [c["id"] for c in config["criteria"]]
    criteria_names = {c["id"]: c["name"] for c in config["criteria"]}
    report_ids = [r["report_id"] for r in config["reports"]]

    df = load_pointwise_scores(output_dir)
    n_judges = df["model"].nunique()
    print(f"Judge models found: {sorted(df['model'].unique())}")
    if n_judges < 3:
        print(
            f"\nWARNING: only {n_judges} judge(s) scored so far. Krippendorff's "
            f"alpha across judges is computable with 2, but is much more "
            f"informative with >=3 independent judges — enable more models "
            f"in config.yaml and re-run run_judge.py first for a robust result."
        )

    rows = []
    for qid in criteria_ids:
        alpha = alpha_for_criterion(df, qid, report_ids)
        rows.append({"criterion": criteria_names[qid], "llm_judge_alpha": alpha})

    models = sorted(df["model"].unique())
    result_df = pd.DataFrame(rows).set_index("criterion")

    # Primary summary: mean of the 9 per-criterion alphas, not a single alpha
    # from one pooled 9x10 matrix stacking all criteria together. The two
    # differ sharply on this data (mean-of-9 ~0.10 vs. pooled ~0.44) because
    # pooling estimates one expected-disagreement term D_e jointly across all
    # 9 criteria's marginal score distributions, so a criterion with unusually
    # skewed scale usage can dominate D_e and distort alpha for criteria that
    # individually show no such skew. Mean-of-9 is also the only one directly
    # comparable to human alpha, which is itself a mean across 9 within-
    # criterion matrices (human raters cannot be pooled into one matrix
    # across the disconnected V1/V2 survey design). See paper.tex Appendix
    # "Statistical Tests" for the full argument.
    mean_of_9_alpha = np.nanmean(result_df["llm_judge_alpha"].values)

    all_items = [(qid, rid) for qid in criteria_ids for rid in report_ids]
    matrix = np.full((len(models), len(all_items)), np.nan)
    for i, model in enumerate(models):
        model_df = df[df["model"] == model].set_index(["criterion", "report_id"])["score"]
        for j, item in enumerate(all_items):
            if item in model_df.index:
                matrix[i, j] = model_df.loc[item]
    try:
        pooled_alpha = krippendorff.alpha(reliability_data=matrix, level_of_measurement="ordinal")
    except (AssertionError, ValueError):
        pooled_alpha = np.nan

    print(f"\n=== LLM-judge inter-rater agreement (Krippendorff's alpha, ordinal) ===")
    print(f"Judges treated as raters: {models}")
    print(f"\nMean across {len(criteria_ids)} criteria (primary):  alpha = {mean_of_9_alpha:.3f}")
    print(f"Pooled {len(criteria_ids)}x{len(report_ids)} matrix (for reference): alpha = {pooled_alpha:.3f}\n")
    print(result_df.round(3).to_string())

    out_path = output_dir / "llm_judge_iaa.csv"
    result_df["mean_of_9_alpha"] = mean_of_9_alpha
    result_df["pooled_alpha"] = pooled_alpha
    result_df.to_csv(out_path)
    print(f"\nSaved -> {out_path}")
    print(
        "\nCompare against human Cross-survey alpha per criterion from "
        "eval_results.ipynb (IAA_per_criterion_barchart) to check whether "
        "LLM judges agree with each other more, less, or about the same as "
        "human raters do — and whether that agreement lines up with the "
        "a priori Tier classification or not."
    )


if __name__ == "__main__":
    main()
