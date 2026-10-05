"""French prompt templates for the LLM-as-judge evaluation.

Pointwise scoring: one API call scores ONE criterion for ONE report. This
mirrors how a human rater answers the LimeSurvey questionnaire one question
at a time, and avoids the score collapse seen when all 9 criteria are asked
in a single call (the model tends to anchor on one value, e.g. 4, across
every dimension when they're all presented together).

Each criterion's question wording and its 1-5 anchor labels are taken
verbatim from the actual LimeSurvey export (data/questionnaire.md +
config.yaml `anchors`), so the LLM sees exactly the same scale definition a
human rater saw.
"""

SYSTEM_PROMPT = """Vous êtes un·e orthophoniste expérimenté·e qui évalue des rapports de séance \
de remédiation cognitive générés automatiquement. Ces rapports résument une séance \
au cours de laquelle un·e patient·e a réalisé des exercices cognitifs.

On vous soumet un rapport et UNE seule question d'évaluation à la fois, avec une \
échelle de Likert à 5 points dont chaque point est défini précisément. Notez le \
rapport en respectant strictement les définitions de l'échelle fournie.

Répondez UNIQUEMENT avec un objet JSON valide, sans texte avant ou après, au format :
{"score": <note entière de 1 à 5>, "comment": "<commentaire libre bref, en français, \
justifiant la note et signalant tout élément spécifique du rapport ayant motivé votre \
choix>"}

Le champ "comment" doit être concret et spécifique au rapport (ex. éléments précis \
présents ou absents), pas une formule générique. N'incluez aucun texte en dehors de \
cet objet JSON."""

USER_PROMPT_TEMPLATE_TEXT = """Voici le rapport de séance à évaluer :

---
{report_text}
---

Question : {question}

Échelle de notation :
{anchor_block}

Répondez uniquement avec l'objet JSON {{"score": ..., "comment": ...}}."""


def build_anchor_block(anchors: dict) -> str:
    lines = [f"{point} = {label}" for point, label in sorted(anchors.items())]
    return "\n".join(lines)


def build_user_prompt(report_text: str, criterion: dict) -> str:
    return USER_PROMPT_TEMPLATE_TEXT.format(
        report_text=report_text.strip(),
        question=criterion["question"],
        anchor_block=build_anchor_block(criterion["anchors"]),
    )
