"""Standalone runner for gpt-6-luna only, since it's extremely slow
(~4 min/call observed, so ~6h for the full 90-call batch) — kept separate
from run_judge.py's main loop so its slowness never blocks other judges
sharing the same config.yaml `enabled` flags.

Saves incrementally (appends after every single call, not just per-model),
so progress is never lost even if this process is killed partway.

Usage:
    python run_gpt6luna_standalone.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from run_judge import load_config, make_client, score_one, REPO_ROOT

MODEL_CFG = {
    "name": "gpt-6-luna",
    "provider": "openai",
    "env_key": "OPENAI_API_KEY",
    "base_url": "https://api.openai.com/v1",
    "input_mode": "text",
    "supports_temperature": False,
}


def main():
    config = load_config()
    criteria = config["criteria"]
    reports = config["reports"]
    output_dir = REPO_ROOT / config["output_dir"]
    raw_dir = output_dir / "raw_responses"
    raw_dir.mkdir(parents=True, exist_ok=True)
    out_csv = output_dir / "llm_scores_pointwise.csv"

    client = make_client(MODEL_CFG)

    for report in reports:
        print(f"{report['report_id']} ({report['system']}, {report['session']})", flush=True)
        for criterion in criteria:
            raw_path = raw_dir / f"gpt-6-luna_{report['report_id']}_{criterion['id']}_run1.txt"
            if raw_path.exists():
                print(f"  {criterion['id']} already done, skipping", flush=True)
                continue

            print(f"  {criterion['id']} ({criterion['name']})...", flush=True)
            score, comment, raw_text = score_one(client, MODEL_CFG, report, criterion, output_dir)
            raw_path.write_text(raw_text, encoding="utf-8")

            row = pd.DataFrame([{
                "model": "gpt-6-luna",
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
                    ~((existing["model"] == "gpt-6-luna")
                      & (existing["report_id"] == report["report_id"])
                      & (existing["criterion"] == criterion["id"]))
                ]
                df = pd.concat([existing, row], ignore_index=True)
            else:
                df = row
            df.to_csv(out_csv, index=False)
            print(f"    score={score} (saved)", flush=True)

    print("\nDone with gpt-6-luna.", flush=True)


if __name__ == "__main__":
    main()
