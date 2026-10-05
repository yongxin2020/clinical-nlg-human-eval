"""LLM-as-judge scoring of the 10 clinical NLG reports on the same 9 criteria
used in the human evaluation questionnaire.

Pointwise scoring: one API call per (report, criterion) pair — the model
sees only one question and its full 1-5 anchor definitions at a time, the
same way a human rater answers the LimeSurvey questionnaire one question at
a time. This avoids score collapse (the model anchoring on a single value
across all 9 dimensions when asked to score them jointly).

Usage:
    cd code/llm_judge
    pip install -r requirements.txt
    cp .env.example .env   # then fill in your real key(s)
    python run_judge.py

Reads config.yaml for model list, criteria (with Likert anchors), and report
paths. Writes one row per (model, report_id, criterion, run_index) to
output/llm_scores_pointwise.csv (including a free-text `comment` field the
model must justify its score with, mirroring the human questionnaire's
[comment] fields - useful for auditing whether the model is reasoning about
report-specific content or defaulting to generic boilerplate), plus raw
model responses to output/raw_responses/ for auditing.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

import requests
import yaml
from dotenv import load_dotenv

from prompt_template import SYSTEM_PROMPT, build_user_prompt

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent

load_dotenv(SCRIPT_DIR / ".env")


def load_config() -> dict:
    with open(SCRIPT_DIR / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


class ChatClient:
    """Minimal OpenAI-compatible chat completions client (plain HTTP).

    Works for DeepSeek, OpenAI, and any other OpenAI-compatible
    /chat/completions endpoint (incl. image content parts) without
    depending on the `openai` SDK (whose recent versions need a newer
    typing-extensions than some Python 3.9 setups can install).
    """

    def __init__(self, api_key: str, base_url: str):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")

    def chat(self, model: str, messages: list[dict], temperature: float = 0.0,
             timeout: int = 120, supports_temperature: bool = True) -> str:
        payload = {"model": model, "messages": messages}
        if supports_temperature:
            payload["temperature"] = temperature
        resp = requests.post(
            f"{self.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]


def make_client(model_cfg: dict) -> ChatClient:
    api_key = os.environ.get(model_cfg["env_key"])
    if not api_key:
        raise RuntimeError(
            f"Missing {model_cfg['env_key']} in code/llm_judge/.env "
            f"(see .env.example)."
        )
    return ChatClient(api_key=api_key, base_url=model_cfg["base_url"])


def extract_json_score(raw_text: str) -> tuple[int, str]:
    """Pull {"score": n, "comment": "..."} out of a model response, tolerating
    stray text or markdown code fences around it. `comment` defaults to ""
    for backward compatibility with older raw responses that lack it."""
    text = raw_text.strip()
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in response: {raw_text[:200]!r}")
    obj = json.loads(match.group(0))

    if "score" not in obj:
        raise ValueError(f"Missing 'score' key in model response: {obj}")
    val = int(round(float(obj["score"])))
    if not 1 <= val <= 5:
        raise ValueError(f"score out of range [1,5]: {val}")
    comment = str(obj.get("comment", ""))
    return val, comment


def build_user_message(report: dict, criterion: dict) -> dict:
    report_path = REPO_ROOT / report["path"]
    report_text = report_path.read_text(encoding="utf-8")
    return {"role": "user", "content": build_user_prompt(report_text, criterion)}


def score_one(
    client: ChatClient,
    model_cfg: dict,
    report: dict,
    criterion: dict,
    max_retries: int = 6,
) -> tuple[int, str, str]:
    user_message = build_user_message(report, criterion)

    # Reasoning models can genuinely take a couple of minutes per call, but
    # if a single request hangs far longer than that, fail it and retry
    # rather than block indefinitely - a stalled TCP connection with no
    # server-side progress does not raise on its own before `timeout`.
    request_timeout = 180 if "reasoner" in model_cfg["name"] else 60

    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            raw_text = client.chat(
                model=model_cfg["name"],
                temperature=model_cfg.get("temperature", 0.0),
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    user_message,
                ],
                timeout=request_timeout,
                supports_temperature=model_cfg.get("supports_temperature", True),
            )
            score, comment = extract_json_score(raw_text)
            return score, comment, raw_text
        except Exception as e:  # malformed JSON, timeout, transient error, etc.
            last_err = e
            # OpenRouter free-tier models share a rate-limited upstream pool
            # (429 "upstream_provider_shared_pool") that needs much longer
            # backoff than a normal transient error to actually clear.
            is_rate_limit = "429" in str(e)
            wait = min(60, 10 * attempt) if is_rate_limit else 2 ** attempt
            print(f"      retry {attempt}/{max_retries} after error: {e} (sleeping {wait}s)")
            time.sleep(wait)
    raise RuntimeError(f"Failed after {max_retries} retries: {last_err}")


def run_model_incremental(model_cfg: dict, criteria: list, reports: list,
                           raw_dir: Path, out_csv: Path) -> None:
    """Score one model, saving after every single call and skipping
    (report, criterion) pairs whose raw response already exists on disk.
    For models slow enough (minutes per call) that a batch run risks losing
    hours of progress if interrupted partway; resuming just re-runs main()."""
    import pandas as pd

    client = make_client(model_cfg)
    for report in reports:
        print(f"  {report['report_id']} ({report['system']}, {report['session']})", flush=True)
        for criterion in criteria:
            raw_path = raw_dir / f"{model_cfg['name']}_{report['report_id']}_{criterion['id']}_run1.txt"
            if raw_path.exists():
                print(f"    {criterion['id']} already done, skipping", flush=True)
                continue

            print(f"    {criterion['id']} ({criterion['name']})...", flush=True)
            score, comment, raw_text = score_one(client, model_cfg, report, criterion)
            raw_path.write_text(raw_text, encoding="utf-8")

            row = pd.DataFrame([{
                "model": model_cfg["name"],
                "report_id": report["report_id"],
                "system": report["system"],
                "session": report["session"],
                "criterion": criterion["id"],
                "run": 1,
                "score": score,
                "comment": comment,
            }])
            if out_csv.exists():
                existing = pd.read_csv(out_csv)
                existing = existing[
                    ~((existing["model"] == model_cfg["name"])
                      & (existing["report_id"] == report["report_id"])
                      & (existing["criterion"] == criterion["id"]))
                ]
                df = pd.concat([existing, row], ignore_index=True)
            else:
                df = row
            df.to_csv(out_csv, index=False)
            print(f"      score={score} (saved)", flush=True)


def main():
    config = load_config()
    criteria = config["criteria"]
    n_repeats = config.get("n_repeats", 1)
    output_dir = REPO_ROOT / config["output_dir"]
    raw_dir = output_dir / "raw_responses"
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    enabled_models = [m for m in config["models"] if m.get("enabled", True)]
    if not enabled_models:
        print("No enabled models in config.yaml — nothing to do.")
        sys.exit(1)

    reports = config["reports"]
    import pandas as pd
    out_csv = output_dir / "llm_scores_pointwise.csv"

    for model_cfg in enabled_models:
        print(f"=== Model: {model_cfg['name']} ===")

        # Models slow enough that a multi-hour batch risks losing all
        # progress if interrupted (set `incremental_save: true` in
        # config.yaml) save after every call and skip completed pairs on
        # resume, instead of accumulating in memory for one save at the end.
        if model_cfg.get("incremental_save", False):
            run_model_incremental(model_cfg, criteria, reports, raw_dir, out_csv)
            continue

        client = make_client(model_cfg)
        model_rows = []
        for report in reports:
            print(f"  {report['report_id']} ({report['system']}, {report['session']})")

            for criterion in criteria:
                for run_idx in range(1, n_repeats + 1):
                    print(f"    {criterion['id']} ({criterion['name']}) "
                          f"run {run_idx}/{n_repeats}...")
                    score, comment, raw_text = score_one(
                        client, model_cfg, report, criterion
                    )

                    raw_path = raw_dir / (
                        f"{model_cfg['name']}_{report['report_id']}_"
                        f"{criterion['id']}_run{run_idx}.txt"
                    )
                    raw_path.write_text(raw_text, encoding="utf-8")

                    model_rows.append({
                        "model": model_cfg["name"],
                        "report_id": report["report_id"],
                        "system": report["system"],
                        "session": report["session"],
                        "criterion": criterion["id"],
                        "run": run_idx,
                        "score": score,
                        "comment": comment,
                    })

        # Save immediately after each model finishes, so a slow/failing
        # later model (e.g. a rate-limited free-tier one) never risks
        # losing an earlier model's completed results.
        new_df = pd.DataFrame(model_rows)
        if out_csv.exists():
            existing = pd.read_csv(out_csv)
            existing = existing[existing["model"] != model_cfg["name"]]
            df = pd.concat([existing, new_df], ignore_index=True)
        else:
            df = new_df
        df.to_csv(out_csv, index=False)
        print(f"  Saved {len(new_df)} rows for {model_cfg['name']} -> {out_csv} "
              f"({len(df)} total rows)")

    print(f"\nDone.")


if __name__ == "__main__":
    main()
