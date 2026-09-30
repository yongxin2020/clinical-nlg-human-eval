# Expert Disagreement as Signal: Evaluating Clinical NLG Without Reference Reports

This repository contains the code, anonymized data, and generated outputs
supporting the human evaluation reported in the paper above.

In low-resource clinical NLG settings, no gold reference reports exist and
expert disagreement is expected. This project treats disagreement as a
structured evaluation signal rather than a failure, and operationalizes three
design principles in practice:
- dimension separation,
- evaluator stratification,
- disagreement transparency.

Case study context: cognitive remediation report generation evaluated by eight
expert raters (certified speech therapists and final-year students), with
analysis of inter-annotator agreement (IAA), profile-linked scoring direction,
and criterion-level reliability for LLM-as-judge applicability.

## Contents

```
.
├── code/
│   ├── eval_results.ipynb   # IAA analysis: Krippendorff's alpha (overall + per-criterion),
│   │                        # Therapist-vs-Student comparison (Mann-Whitney U / rank-biserial r),
│   │                        # pooled and split by report-generation system, LLM-as-judge validation
│   ├── Fig/                 # Pre-generated figures (also reproduced when notebook is run)
│   └── llm_judge/           # LLM-as-judge experiment: 5 LLM judges score the same 10 reports
│                            # on the same 9 French criteria as the human survey
├── data/
│   ├── results-survey728958_anonymized.csv   # Survey V1 raw responses (4 raters x 5 reports)
│   ├── results-survey589241_anonymized.csv   # Survey V2 raw responses (4 raters x 5 reports)
│   ├── questionnaire.md                      # Questionnaire wording (French original + English)
│   └── reports/
│       ├── template/    # 5 template-generated reports (Markdown)
│       └── gpt4/        # 5 GPT-4-generated reports (Markdown)
```

## Evaluation protocol

Eight expert evaluators (4 professional speech-language pathologists / speech
therapists, 4 advanced students) rated 5 session reports each (a mix of
Template- and GPT-4-generated reports), scoring 9 criteria per report on a
1–5 Likert scale, split across two survey instances (V1, V2), 4 raters each.
See `data/questionnaire.md` for the full question wording.

## Reproducing the analysis

**Dependencies**: `pip install pandas numpy matplotlib seaborn scipy scikit-learn krippendorff`
(additional dependencies for the LLM-as-judge experiment below are in
`code/llm_judge/requirements.txt`)

Open `code/eval_results.ipynb` in Jupyter and run top to bottom. The notebook:
1. Loads and preprocesses the two anonymized survey CSVs.
2. Computes inter-annotator agreement (Krippendorff's alpha, ordinal) overall
   and per criterion, for the full panel and by evaluator group.
3. Runs Mann-Whitney U tests (with rank-biserial correlation as effect size)
   comparing Therapist vs. Student scoring patterns, both pooled and split by
   report-generation system (Template vs. GPT-4).
4. Loads LLM-as-judge scores (from `code/llm_judge/`) and compares LLM-human
   agreement against human IAA, and LLM-LLM agreement against human-human
   agreement.
5. Saves all generated figures to `code/Fig/`.

Reproducing the LLM-as-judge scoring itself requires API keys; see
`code/llm_judge/` for setup.

## Privacy & anonymization

- All rater identities have been replaced with randomly generated tokens.
- IP addresses, timestamps, and other identifying survey metadata have been
  removed from the CSV exports.
- Participant identifiers in the generated reports (e.g. `M01E`) are
  study-internal codes, not real names, and cannot be traced back to any
  individual outside the original research team.
