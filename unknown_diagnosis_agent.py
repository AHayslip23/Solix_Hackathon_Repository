"""
Unknown-Diagnosis Review Agent
==========================

Investigates why some meningitis cases have no recorded diagnosis (Diagnosis = Unknown).

For each Unknown case it:
  1. Scores how bacterial-like vs. viral-like the case looks, using a model trained only on
     cases with a known diagnosis (spinal fluid labs, CRP, age, gender).
  2. Checks each lab on its own: does this value look bacterial, viral, or neither?
  3. Assigns likely reasons the diagnosis may be missing:
       - Mixed lab signals      some labs point bacterial, others viral
       - Lab/test conflict      labs point one way, pathogen test the other
       - Atypical values        a lab falls outside the usual range of BOTH groups
       - Borderline profile     the overall picture sits between the two groups
       - Clear profile          labs strongly match one group (diagnosis may simply not
                                have been recorded)
  4. Builds a review queue, least certain cases first, for human review.

A Claude agent sits on top and turns the findings into plain-language case summaries.

Usage (in a notebook):

    from unknown_diagnosis_agent import UnknownDiagnosisReviewer, UnknownDiagnosisAgent

    reviewer = UnknownDiagnosisReviewer(meningitis_clean)
    reviewer.print_overview()                      # works without an API key
    reviewer.review_queue().head(10)               # DataFrame of all Unknown cases
    print(reviewer.template_summary(reviewer.assess_case(12)))
    reviewer.export_queue("unknown_review_queue.csv")

    agent = UnknownDiagnosisAgent(reviewer)        # needs ANTHROPIC_API_KEY
    print(agent.ask("Why do these cases lack a diagnosis? Give me the main patterns."))
    print(agent.ask("Walk me through the three cases that most need review."))
"""

import json

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_FEATURES = ["Glucose_Level", "CRP_Level", "WBC_Count", "Protein_Level",
                  "Age", "Gender_Female", "Gender_Unknown"]

# Labs assessed one at a time. Direction notes are the expected bacterial pattern.
LABS = {
    "Glucose_Level": ("spinal fluid glucose", "mg/dL", "low in bacterial cases"),
    "CRP_Level":     ("CRP (inflammation marker)", "mg/L", "high in bacterial cases"),
    "WBC_Count":     ("spinal fluid white cell count", "cells/µL", "high in bacterial cases"),
    "Protein_Level": ("spinal fluid protein", "mg/dL", "high in bacterial cases"),
}

DIAGNOSIS_COLS = {"Bacterial": "Diagnosis_Bacterial", "Viral": "Diagnosis_Viral",
                  "Unknown": "Diagnosis_Unknown"}

LEAN_THRESHOLD = np.log(3)   # a lab "leans" one way if it's 3x more typical of that group
REASON_ORDER = ["Lab/test conflict", "Atypical values", "Mixed lab signals",
                "Borderline profile", "Clear profile"]

MODEL_NAME = "claude-sonnet-5-5"


def _diagnosis_label(df):
    """Rebuild a single Diagnosis column from one-hot columns if needed."""
    if "Diagnosis" in df.columns:
        return df["Diagnosis"].astype(str)
    cols = [c for c in DIAGNOSIS_COLS.values() if c in df.columns]
    return df[cols].astype(int).idxmax(axis=1).str.replace("Diagnosis_", "", regex=False)


# ---------------------------------------------------------------------------
# Analysis layer
# ---------------------------------------------------------------------------

class UnknownDiagnosisReviewer:
    """Scores and categorizes every Unknown-diagnosis case."""

    def __init__(self, df, random_state=42):
        self.df = df.copy()
        self.df["_Diagnosis"] = _diagnosis_label(self.df)
        self.known = self.df[self.df["_Diagnosis"].isin(["Bacterial", "Viral"])]
        self.unknown = self.df[self.df["_Diagnosis"] == "Unknown"]

        # --- Bacterial-vs-viral model, trained on known cases only ---
        X = self.known[MODEL_FEATURES].astype("float64")
        y = (self.known["_Diagnosis"] == "Bacterial").astype(int)
        self.model = RandomForestClassifier(n_estimators=500, min_samples_leaf=20,
                                            random_state=random_state, n_jobs=-1)
        cv = StratifiedKFold(5, shuffle=True, random_state=random_state)
        self.model_cv_auc = float(cross_val_score(self.model, X, y, cv=cv,
                                                  scoring="roc_auc").mean())
        self.model.fit(X, y)

        # --- Per-lab reference distributions for each known group ---
        self.kde, self.ranges, self.medians = {}, {}, {}
        for lab in LABS:
            self.kde[lab], self.ranges[lab], self.medians[lab] = {}, {}, {}
            for grp in ["Bacterial", "Viral"]:
                vals = self.known.loc[self.known["_Diagnosis"] == grp, lab].astype(float)
                self.kde[lab][grp] = gaussian_kde(vals)
                self.ranges[lab][grp] = (float(vals.quantile(0.05)), float(vals.quantile(0.95)))
                self.medians[lab][grp] = float(vals.median())

        # --- Score every Unknown case once ---
        Xu = self.unknown[MODEL_FEATURES].astype("float64")
        self.unknown_probs = pd.Series(self.model.predict_proba(Xu)[:, 1], index=self.unknown.index)
        self._assessments = {i: self._assess(i) for i in self.unknown.index}

        # --- Baseline: how often the same flags fire for cases WITH a diagnosis ---
        # (out-of-fold probabilities, so known cases aren't scored by a model that saw them)
        known_probs = cross_val_predict(self.model, X, y, cv=cv, method="predict_proba")[:, 1]
        self.baseline_rates = {
            "Atypical values": self._share(self.known, lambda lean, atyp: any(atyp)),
            "Mixed lab signals": self._share(
                self.known, lambda lean, atyp: "bacterial" in lean and "viral" in lean),
            "Borderline profile": round(float(((known_probs > 0.3) & (known_probs < 0.7)).mean()), 2),
        }

    def _share(self, frame, rule):
        hits = []
        for _, row in frame.iterrows():
            res = [self._lab_lean(lab, float(row[lab])) for lab in LABS]
            hits.append(rule([r[0] for r in res], [r[1] for r in res]))
        return round(float(np.mean(hits)), 2)

    # ---- per-case logic ----------------------------------------------------

    def _lab_lean(self, lab, value):
        lb = float(self.kde[lab]["Bacterial"](value)[0]) + 1e-12
        lv = float(self.kde[lab]["Viral"](value)[0]) + 1e-12
        log_ratio = np.log(lb / lv)
        if log_ratio > LEAN_THRESHOLD:
            lean = "bacterial"
        elif log_ratio < -LEAN_THRESHOLD:
            lean = "viral"
        else:
            lean = "neutral"
        (b_lo, b_hi), (v_lo, v_hi) = self.ranges[lab]["Bacterial"], self.ranges[lab]["Viral"]
        atypical = not (b_lo <= value <= b_hi or v_lo <= value <= v_hi)
        return lean, atypical

    def _assess(self, idx):
        row = self.unknown.loc[idx]
        prob = float(self.unknown_probs.loc[idx])
        pathogen = bool(row["Pathogen_Present"]) if "Pathogen_Present" in row else None

        labs = []
        for lab, (name, unit, note) in LABS.items():
            value = float(row[lab])
            lean, atypical = self._lab_lean(lab, value)
            labs.append({
                "lab": name, "value": f"{value:,.0f} {unit}", "leans": lean,
                "outside_both_groups_usual_range": atypical,
                "typical_bacterial": f"{self.medians[lab]['Bacterial']:,.0f}",
                "typical_viral": f"{self.medians[lab]['Viral']:,.0f}",
                "pattern": note,
            })

        leans = [l["leans"] for l in labs]
        reasons = []
        if pathogen is not None and ((prob >= 0.7 and not pathogen) or (prob <= 0.3 and pathogen)):
            reasons.append("Lab/test conflict")
        if any(l["outside_both_groups_usual_range"] for l in labs):
            reasons.append("Atypical values")
        if "bacterial" in leans and "viral" in leans:
            reasons.append("Mixed lab signals")
        if 0.3 < prob < 0.7:
            reasons.append("Borderline profile")
        if not reasons and (prob >= 0.85 or prob <= 0.15):
            reasons.append("Clear profile")
        if not reasons:
            reasons.append("Borderline profile")
        reasons.sort(key=REASON_ORDER.index)

        lean_overall = "bacterial" if prob >= 0.5 else "viral"
        certainty = abs(prob - 0.5) * 2          # 0 = coin flip, 1 = certain
        return {
            "case_id": int(idx),
            "labs_look_more": lean_overall,
            "probability_bacterial": round(prob, 3),
            "certainty": round(certainty, 3),
            "pathogen_detected": pathogen,
            "risk_level": self._risk(row),
            "outcome": row.get("Outcome", None),
            "primary_reason": reasons[0],
            "all_reasons": reasons,
            "labs": labs,
            "labs_leaning_bacterial": sum(l == "bacterial" for l in leans),
            "labs_leaning_viral": sum(l == "viral" for l in leans),
            "needs_review": reasons[0] != "Clear profile",
        }

    @staticmethod
    def _risk(row):
        if "Risk_Level" in row.index:
            return row["Risk_Level"]
        cols = [c for c in row.index if c.startswith("Risk_Level_")]
        for c in cols:
            if bool(row[c]):
                return c.replace("Risk_Level_", "")
        return None

    # ---- public tools --------------------------------------------------------

    def assess_case(self, case_id):
        if case_id not in self._assessments:
            ids = sorted(self._assessments)
            return {"error": f"Case {case_id} is not an Unknown-diagnosis case. "
                             f"There are {len(ids)} such cases, e.g. {ids[:8]}."}
        return self._assessments[case_id]

    def review_queue(self, reason=None, limit=None):
        """All Unknown cases as a DataFrame, least certain first."""
        rows = [{"case_id": a["case_id"], "primary_reason": a["primary_reason"],
                 "labs_look_more": a["labs_look_more"],
                 "probability_bacterial": a["probability_bacterial"],
                 "certainty": a["certainty"], "pathogen_detected": a["pathogen_detected"],
                 "labs_bacterial/viral": f"{a['labs_leaning_bacterial']}/{a['labs_leaning_viral']}",
                 "unusual_labs": ", ".join(f"{l['lab']} {l['value']}" for l in a["labs"]
                                           if l["outside_both_groups_usual_range"]),
                 "all_reasons": "; ".join(a["all_reasons"]), "needs_review": a["needs_review"]}
                for a in self._assessments.values()]
        q = pd.DataFrame(rows).sort_values(["needs_review", "certainty"], ascending=[False, True])
        if reason:
            q = q[q["all_reasons"].str.contains(reason, case=False)]
        return q.head(limit) if limit else q.reset_index(drop=True)

    def cohort_overview(self):
        """How Unknown cases compare with Bacterial and Viral, plus reason counts."""
        labs = list(LABS)
        med = self.df.groupby("_Diagnosis")[labs].median().round(0)
        overview = {
            "counts": self.df["_Diagnosis"].value_counts().to_dict(),
            "median_labs": {g: {LABS[l][0]: f"{v:,.0f} {LABS[l][1]}" for l, v in r.items()}
                            for g, r in med.iterrows()},
            "share_with_pathogen_detected": self.df.groupby("_Diagnosis")["Pathogen_Present"]
                                                .mean().round(2).to_dict(),
            "share_glucose_above_70": self.df.groupby("_Diagnosis")["Glucose_Level"]
                                          .apply(lambda s: round(float((s > 70).mean()), 2)).to_dict(),
            "model_quality_on_known_cases_roc_auc": round(self.model_cv_auc, 3),
        }
        if "Outcome" in self.df.columns:
            overview["death_rate"] = self.df.groupby("_Diagnosis")["Outcome"].apply(
                lambda s: round(float((s == "Deceased").mean()), 3)).to_dict()
        q = self.review_queue()
        overview["primary_reason_counts"] = q["primary_reason"].value_counts().to_dict()
        overview["any_reason_counts"] = {r: int(q["all_reasons"].str.contains(r).sum())
                                         for r in REASON_ORDER}
        n = len(q)
        overview["flag_rate_unknown_vs_diagnosed_cases"] = {
            r: {"unknown_cases": round(overview["any_reason_counts"][r] / n, 2),
                "cases_with_a_diagnosis": self.baseline_rates[r]}
            for r in self.baseline_rates}
        overview["labs_look_more"] = q["labs_look_more"].value_counts().to_dict()
        overview["needs_review"] = int(q["needs_review"].sum())
        return overview

    def compare_lab(self, lab):
        """Distribution of one lab across Bacterial, Viral, and Unknown cases."""
        key = next((k for k, v in LABS.items() if lab in (k, v[0])), None)
        if key is None:
            return {"error": f"Unknown lab '{lab}'. Options: {list(LABS)}"}
        desc = self.df.groupby("_Diagnosis")[key].describe(percentiles=[.1, .25, .5, .75, .9])
        return {"lab": LABS[key][0], "unit": LABS[key][1], "bacterial_pattern": LABS[key][2],
                "by_group": desc.round(0).to_dict(orient="index")}

    # ---- outputs without the API --------------------------------------------

    def print_overview(self):
        o = self.cohort_overview()
        print(f"Unknown cases: {o['counts'].get('Unknown', 0)} "
              f"(model ROC-AUC on known cases: {o['model_quality_on_known_cases_roc_auc']})")
        print(f"Labs look more bacterial / viral: {o['labs_look_more']}")
        print(f"Cases needing review: {o['needs_review']}\n")
        print("Primary reason:")
        for r, n in o["primary_reason_counts"].items():
            print(f"  {r:<20} {n}")
        print("\nAny reason (cases can have several):")
        for r, n in o["any_reason_counts"].items():
            print(f"  {r:<20} {n}")
        print("\nHow often each flag fires (Unknown vs. cases with a diagnosis):")
        for r, d in o["flag_rate_unknown_vs_diagnosed_cases"].items():
            print(f"  {r:<20} {d['unknown_cases']:.0%} vs. {d['cases_with_a_diagnosis']:.0%}")

    @staticmethod
    def template_summary(a):
        if "error" in a:
            return a["error"]
        b = [l["lab"] for l in a["labs"] if l["leans"] == "bacterial"]
        v = [l["lab"] for l in a["labs"] if l["leans"] == "viral"]
        odd = [l["lab"] for l in a["labs"] if l["outside_both_groups_usual_range"]]
        text = (f"Case {a['case_id']}: labs look more {a['labs_look_more']} "
                f"({a['probability_bacterial']:.0%} bacterial). Primary reason: {a['primary_reason']}.")
        if b:
            text += f" Bacterial-leaning: {', '.join(b)}."
        if v:
            text += f" Viral-leaning: {', '.join(v)}."
        if odd:
            text += f" Unusual for both groups: {', '.join(odd)}."
        if a["pathogen_detected"] is not None:
            text += f" Pathogen {'detected' if a['pathogen_detected'] else 'not detected'}."
        text += " Recommend clinician review." if a["needs_review"] else " Profile is clear."
        return text

    def export_queue(self, path="unknown_review_queue.csv"):
        q = self.review_queue()
        q["summary"] = [self.template_summary(self._assessments[i]) for i in q["case_id"]]
        q.to_csv(path, index=False)
        return path


# ---------------------------------------------------------------------------
# Claude agent layer
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You help a data analyst understand why some meningitis cases in a dataset \
have no recorded diagnosis (Diagnosis = Unknown), and which of them most need a human to review.

Always use the tools to get facts; never invent numbers or case details.

How to reason:
- Each Unknown case has a model score for how bacterial-like vs. viral-like its labs look. The \
model was trained only on cases with a known diagnosis, using spinal fluid glucose, protein, \
white cell count, CRP, age, and gender. It did NOT use the pathogen test.
- Each lab is also judged on its own. Bacterial meningitis classically shows LOW spinal fluid \
glucose and HIGH protein, white cells, and CRP; viral cases are usually milder.
- Reason categories: Mixed lab signals, Lab/test conflict, Atypical values, Borderline \
profile, Clear profile. A case can have several.
- Always put counts in context: cohort_overview gives how often each flag fires for cases \
WITH a diagnosis. A flag only matters if it is much more common among Unknown cases.

When explaining, offer plausible reasons as hypotheses, never as facts. Reasonable clinical \
possibilities include: antibiotics given before the spinal tap (can make bacterial labs look \
milder and tests come back negative), culture-negative bacterial infection, early-stage \
illness, high blood sugar raising spinal fluid glucose, a non-infectious cause, or a diagnosis \
that was reached but never recorded. Say clearly that the dataset has no information on \
antibiotics, blood glucose, or timing, so these cannot be confirmed.

Rules:
- Never state a case's true diagnosis. Say "the labs look more bacterial", not "this is bacterial".
- Recommend clinician review for uncertain cases; you support review, you do not replace it.
- Write for a non-technical reader: plain words, short paragraphs, no raw model internals.
- This is a teaching dataset with some unrealistic values; never give treatment advice.
- A spinal fluid glucose of 0 was kept on purpose: in severe bacterial infection, glucose can \
fall below what a lab can detect. Treat it as "undetectable", not as a data error.
- For questions covering many cases, use compact tables rather than long paragraphs per case. \
If there are more than about 15 cases, summarize patterns and show the most important ones, \
and offer to continue with the rest.
- review_queue includes an "unusual_labs" column naming which labs are outside the usual range."""

TOOLS = [
    {"name": "cohort_overview",
     "description": "Compare Unknown-diagnosis cases with Bacterial and Viral cases (median labs, "
                    "pathogen detection, death rate) and count how many Unknown cases fall into "
                    "each reason category. Start here for general 'why' questions.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "review_queue",
     "description": "List Unknown cases ordered from least to most certain (cases needing review "
                    "first). Optionally filter to one reason category.",
     "input_schema": {"type": "object", "properties": {
         "reason": {"type": "string", "description": "Optional reason category to filter by.",
                    "enum": REASON_ORDER},
         "limit": {"type": "integer", "description": "How many cases to return (default 10)."}}}},
    {"name": "assess_case",
     "description": "Full assessment of one Unknown case: bacterial-vs-viral score, how each lab "
                    "leans, values outside the usual range, pathogen result, and reasons.",
     "input_schema": {"type": "object", "properties": {
         "case_id": {"type": "integer", "description": "The case's row index."}},
         "required": ["case_id"]}},
    {"name": "compare_lab",
     "description": "Distribution of one lab across Bacterial, Viral, and Unknown cases.",
     "input_schema": {"type": "object", "properties": {
         "lab": {"type": "string", "enum": list(LABS)}}, "required": ["lab"]}},
]


class UnknownDiagnosisAgent:
    """Claude agent that investigates Unknown-diagnosis cases using the reviewer's tools."""

    def __init__(self, reviewer, model=MODEL_NAME, max_tokens=8000, max_steps=12,
                 max_continuations=2):
        import anthropic  # imported here so the analysis layer works without the SDK
        self.client = anthropic.Anthropic()   # reads ANTHROPIC_API_KEY from the environment
        self.reviewer = reviewer
        self.model = model
        self.max_tokens = max_tokens
        self.max_steps = max_steps
        self.max_continuations = max_continuations
        self.messages = []
        self.last_stop_reason = None      # for checking why the last reply ended
        self.last_output_tokens = None

    def _run_tool(self, name, args):
        r = self.reviewer
        try:
            if name == "cohort_overview":
                result = r.cohort_overview()
            elif name == "review_queue":
                q = r.review_queue(reason=args.get("reason"), limit=args.get("limit", 10))
                result = q.to_dict(orient="records")
            elif name == "assess_case":
                result = r.assess_case(int(args["case_id"]))
            elif name == "compare_lab":
                result = r.compare_lab(args["lab"])
            else:
                result = {"error": f"Unknown tool {name}"}
        except Exception as exc:
            result = {"error": f"{type(exc).__name__}: {exc}"}
        return json.dumps(result, default=str)

    def ask(self, question, verbose=False):
        """Ask a question. Long answers that hit the length cap are continued automatically."""
        self.messages.append({"role": "user", "content": question})
        parts, continuations = [], 0
        for _ in range(self.max_steps):
            response = self.client.messages.create(
                model=self.model, max_tokens=self.max_tokens, system=SYSTEM_PROMPT,
                tools=TOOLS, messages=self.messages)
            self.last_stop_reason = response.stop_reason
            self.last_output_tokens = getattr(response.usage, "output_tokens", None)
            text = "".join(b.text for b in response.content if b.type == "text")

            if response.stop_reason == "max_tokens":
                # The reply was cut off by the length cap. Keep what was written (dropping any
                # half-written tool call) and ask Claude to pick up where it stopped.
                kept = [b for b in response.content if b.type != "tool_use"]
                parts.append(text)
                if verbose:
                    print(f"[length cap reached after {self.last_output_tokens} tokens; continuing]")
                if continuations >= self.max_continuations or not kept:
                    return "".join(parts) + "\n\n[Answer cut off: length limit reached. " \
                                            "Ask a narrower question or raise max_tokens.]"
                self.messages.append({"role": "assistant", "content": kept})
                self.messages.append({"role": "user", "content":
                                      "Continue exactly where you stopped. Do not repeat anything."})
                continuations += 1
                continue

            self.messages.append({"role": "assistant", "content": response.content})
            if response.stop_reason != "tool_use":
                parts.append(text)
                return "".join(parts)

            results = []
            for block in response.content:
                if block.type == "tool_use":
                    if verbose:
                        print(f"[tool] {block.name}({block.input})")
                    results.append({"type": "tool_result", "tool_use_id": block.id,
                                    "content": self._run_tool(block.name, block.input)})
            self.messages.append({"role": "user", "content": results})
        return "".join(parts) + "\n\n[Stopped: the agent reached its step limit.]"

    def reset(self):
        self.messages = []
