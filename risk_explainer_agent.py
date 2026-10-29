"""
Risk Level Explanation Agent
============================

Trains a classifier to predict a patient's meningitis Risk_Level (High / Moderate / Low),
uses SHAP to work out which factors drove each prediction, and lets a Claude agent turn
those numbers into plain-language explanations for a non-technical audience.

Usage (in a notebook):

    from risk_explainer_agent import RiskExplainer, RiskExplainerAgent

    explainer = RiskExplainer(meningitis_clean)       # trains + evaluates the model
    explainer.print_evaluation()

    # Works without an API key: rule-based explanation
    print(explainer.template_explanation(explainer.explain_patient(42)))

    # With an API key (export ANTHROPIC_API_KEY=...): the full agent
    agent = RiskExplainerAgent(explainer)
    print(agent.ask("Why was patient 42 rated high risk?"))
    print(agent.ask("What would have to change for them to be low risk?"))
"""

import json

import numpy as np
import pandas as pd
import shap
from scipy.stats import percentileofscore
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

RISK_FEATURES = [
    # Spinal fluid labs: the main drivers
    "CRP_Level", "Glucose_Level", "Protein_Level", "WBC_Count",
    # Blood labs
    "WBC_Blood_Count", "Platelets",
    # Demographics
    "Age", "Gender_Male", "Gender_Female", "Gender_Unknown",
    # Test results / clinical judgment
    "Pathogen_Present", "Diagnosis_Viral", "Diagnosis_Bacterial", "Diagnosis_Unknown",
]

TARGET = "Risk_Level"

# One-hot groups are combined into a single factor in explanations
GROUPS = {
    "Gender": ["Gender_Male", "Gender_Female", "Gender_Unknown"],
    "Diagnosis": ["Diagnosis_Viral", "Diagnosis_Bacterial", "Diagnosis_Unknown"],
}

PLAIN_NAMES = {
    "CRP_Level": "CRP (a blood marker of inflammation)",
    "Glucose_Level": "spinal fluid glucose (sugar)",
    "Protein_Level": "spinal fluid protein",
    "WBC_Count": "white blood cell count in the spinal fluid",
    "WBC_Blood_Count": "white blood cell count in the blood",
    "Platelets": "platelet count",
    "Age": "age",
    "Gender": "gender",
    "Pathogen_Present": "whether a pathogen (germ) was detected",
    "Diagnosis": "the diagnosis (bacterial, viral, or unknown)",
}

UNITS = {
    "CRP_Level": "mg/L", "Glucose_Level": "mg/dL", "Protein_Level": "mg/dL",
    "WBC_Count": "cells/µL", "WBC_Blood_Count": "cells/µL", "Platelets": "per µL",
    "Age": "years",
}

MODEL_NAME = "claude-sonnet-5-5"


# ---------------------------------------------------------------------------
# Model + SHAP layer
# ---------------------------------------------------------------------------

class RiskExplainer:
    """Trains the risk model and produces structured explanations for any patient."""

    def __init__(self, df, features=RISK_FEATURES, target=TARGET, test_size=0.2, random_state=42):
        self.features = list(features)
        self.df = df.copy()

        # Booleans and nullable ints -> float, which sklearn and SHAP handle cleanly
        X = self.df[self.features].astype("float64")
        y = self.df[target].astype(str)
        if X.isna().any().any():
            raise ValueError(f"Features still contain missing values: "
                             f"{X.isna().sum()[lambda s: s > 0].to_dict()}")
        self.X, self.y = X, y

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=random_state)

        self.model = RandomForestClassifier(
            n_estimators=400, min_samples_leaf=3, class_weight="balanced",
            random_state=random_state, n_jobs=-1)
        self.model.fit(self.X_train, self.y_train)
        self.classes = list(self.model.classes_)

        self.shap_explainer = shap.TreeExplainer(self.model)

        # Reference values from the training data, for "high / low / typical" wording
        self.medians = self.X_train.median()
        self.q25 = self.X_train.quantile(0.25)
        self.q75 = self.X_train.quantile(0.75)

    # ---- evaluation -------------------------------------------------------

    def evaluation(self):
        pred = self.model.predict(self.X_test)
        return {
            "balanced_accuracy": round(balanced_accuracy_score(self.y_test, pred), 3),
            "report": classification_report(self.y_test, pred, output_dict=True, zero_division=0),
            "confusion_matrix": pd.DataFrame(
                confusion_matrix(self.y_test, pred, labels=self.classes),
                index=[f"true: {c}" for c in self.classes],
                columns=[f"pred: {c}" for c in self.classes]),
        }

    def print_evaluation(self):
        pred = self.model.predict(self.X_test)
        print(f"Balanced accuracy (test set): {balanced_accuracy_score(self.y_test, pred):.3f}\n")
        print(classification_report(self.y_test, pred, zero_division=0))
        print(self.evaluation()["confusion_matrix"])

    # ---- SHAP helpers -----------------------------------------------------

    def _shap_matrix(self, X):
        """SHAP values as an array of shape (rows, features, classes)."""
        values = self.shap_explainer.shap_values(X)
        if isinstance(values, list):                      # older shap: list per class
            values = np.stack(values, axis=-1)
        return np.asarray(values)

    def _group_contributions(self, shap_row, x_row):
        """Combine one-hot dummies into single factors. Returns {factor: (shap, description)}."""
        out = {}
        grouped = {c for cols in GROUPS.values() for c in cols}
        for name, cols in GROUPS.items():
            cols = [c for c in cols if c in self.features]
            if not cols:
                continue
            idx = [self.features.index(c) for c in cols]
            active = [c.split("_", 1)[1] for c in cols if x_row[c] == 1]
            out[name] = (float(shap_row[idx].sum()), active[0] if active else "not recorded")
        for i, col in enumerate(self.features):
            if col not in grouped:
                out[col] = (float(shap_row[i]), self._describe_value(col, x_row[col]))
        return out

    def _describe_value(self, col, value):
        if col == "Pathogen_Present":
            return "detected" if value == 1 else "not detected"
        unit = UNITS.get(col, "")
        pct = percentileofscore(self.X_train[col], value, kind="mean")
        if pct < 50:
            where = f"lower than {100 - pct:.0f}% of patients"
        else:
            where = f"higher than {pct:.0f}% of patients"
        return f"{value:,.0f} {unit}, {where} (median {self.medians[col]:,.0f})".replace("  ", " ")

    @staticmethod
    def _confidence(prob):
        if prob >= 0.7:
            return "high"
        if prob >= 0.5:
            return "moderate"
        return "low"

    # ---- public tools -----------------------------------------------------

    def explain_patient(self, patient_id, top_n=4, overrides=None):
        """Structured explanation for one patient (by DataFrame index label).

        overrides: optional {feature: new_value} for what-if questions.
        """
        if patient_id not in self.X.index:
            return {"error": f"No patient with index {patient_id}. "
                             f"Valid range is {self.X.index.min()}-{self.X.index.max()}."}
        x = self.X.loc[[patient_id]].copy()
        if overrides:
            for k, v in overrides.items():
                if k not in self.features:
                    return {"error": f"Unknown feature '{k}'. Options: {self.features}"}
                x[k] = float(v)

        probs = self.model.predict_proba(x)[0]
        pred_idx = int(np.argmax(probs))
        predicted = self.classes[pred_idx]

        shap_row = self._shap_matrix(x)[0, :, pred_idx]
        contribs = self._group_contributions(shap_row, x.iloc[0])
        ranked = sorted(contribs.items(), key=lambda kv: abs(kv[1][0]), reverse=True)

        def fmt(items):
            return [{"factor": PLAIN_NAMES.get(k, k), "feature": k, "value": desc,
                     "effect": round(s, 3)} for k, (s, desc) in items]

        pushed_toward = [kv for kv in ranked if kv[1][0] > 0][:top_n]
        pushed_away = [kv for kv in ranked if kv[1][0] < 0][:2]

        result = {
            "patient_id": int(patient_id),
            "predicted_risk": predicted,
            "probabilities": {c: round(float(p), 3) for c, p in zip(self.classes, probs)},
            "confidence": self._confidence(probs[pred_idx]),
            "factors_pushing_toward_prediction": fmt(pushed_toward),
            "factors_pushing_away": fmt(pushed_away),
        }
        if overrides:
            result["what_if_changes"] = overrides
        else:
            result["recorded_risk_level"] = self.y.loc[patient_id]
            result["in_training_data"] = bool(patient_id in self.X_train.index)
        return result

    def global_drivers(self, sample=400, random_state=0):
        """Average importance of each factor across patients, per risk level."""
        Xs = self.X_test if len(self.X_test) <= sample else self.X_test.sample(sample, random_state=random_state)
        sv = np.abs(self._shap_matrix(Xs))                 # rows x features x classes
        out = {}
        for k, cls in enumerate(self.classes):
            mean_abs = pd.Series(sv[:, :, k].mean(axis=0), index=self.features)
            for name, cols in GROUPS.items():
                cols = [c for c in cols if c in mean_abs.index]
                if cols:
                    mean_abs[name] = mean_abs[cols].sum()
                    mean_abs = mean_abs.drop(cols)
            mean_abs = mean_abs.sort_values(ascending=False)
            out[cls] = [{"factor": PLAIN_NAMES.get(f, f), "importance": round(float(v), 3)}
                        for f, v in mean_abs.head(6).items()]
        return out

    def typical_values(self):
        """Median lab values and group shares for each recorded risk level."""
        numeric = [c for c in self.features if c in UNITS]
        med = self.X.groupby(self.y)[numeric].median().round(0)
        shares = self.X.groupby(self.y)[["Pathogen_Present", "Diagnosis_Bacterial", "Diagnosis_Viral"]].mean().round(2)
        return {
            "median_values": {r: {PLAIN_NAMES[c]: f"{v:,.0f} {UNITS[c]}" for c, v in row.items()}
                              for r, row in med.iterrows()},
            "share_with": {r: {"pathogen detected": row["Pathogen_Present"],
                               "bacterial diagnosis": row["Diagnosis_Bacterial"],
                               "viral diagnosis": row["Diagnosis_Viral"]}
                           for r, row in shares.iterrows()},
            "counts": self.y.value_counts().to_dict(),
        }

    def count_patients(self, group_by="Diagnosis", use="recorded"):
        """Exact patient counts by risk level, split by another variable.

        group_by: Diagnosis, Pathogen_Present, Gender, Outcome, or Age_Group.
        use: "recorded" (risk level in the data) or "predicted" (model's prediction).
        """
        df = self.df.loc[self.X.index]
        if group_by in GROUPS:
            cols = [c for c in GROUPS[group_by] if c in df.columns]
            groups = df[cols].astype(int).idxmax(axis=1).str.replace(group_by + "_", "", regex=False)
        elif group_by == "Pathogen_Present":
            groups = df["Pathogen_Present"].map({True: "detected", False: "not detected",
                                                 1: "detected", 0: "not detected"})
        elif group_by in df.columns:
            groups = df[group_by].astype(str)
        else:
            return {"error": f"Can't group by '{group_by}'. Options: Diagnosis, Pathogen_Present, "
                             f"Gender, Outcome, Age_Group."}
        risk = self.y if use == "recorded" else pd.Series(self.model.predict(self.X), index=self.X.index)
        table = pd.crosstab(risk, groups, margins=True, margins_name="Total")
        return {"risk_level_source": use, "grouped_by": group_by,
                "counts": {r: {k: int(v) for k, v in row.items()} for r, row in table.iterrows()}}

    def test_patient_ids(self):
        """Patients the model never saw in training; best for honest demos."""
        return sorted(int(i) for i in self.X_test.index)

    # ---- no-API fallback ----------------------------------------------------

    @staticmethod
    def template_explanation(result):
        """A plain-language explanation built from rules, for use without the API."""
        if "error" in result:
            return result["error"]
        top = result["factors_pushing_toward_prediction"]
        reasons = ", ".join(f"{f['factor']} ({f['value']})" for f in top[:3])
        text = (f"Patient {result['patient_id']} is predicted {result['predicted_risk']} "
                f"({result['confidence']} confidence, "
                f"{result['probabilities'][result['predicted_risk']]:.0%}). "
                f"The main reasons: {reasons}.")
        if result["factors_pushing_away"]:
            f = result["factors_pushing_away"][0]
            text += f" Pointing the other way: {f['factor']} ({f['value']})."
        rec = result.get("recorded_risk_level")
        if rec and rec != result["predicted_risk"]:
            text += f" Note: the recorded risk level was {rec}, so the model disagrees with the record."
        return text


# ---------------------------------------------------------------------------
# Claude agent layer
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You explain a machine-learning model's meningitis risk-level predictions \
to a non-technical audience (for example, a hospital administrator or a student).

Use the tools to get the facts; never invent numbers. Then explain in plain language:
- Lead with the predicted risk level and how confident the model is.
- Name the 2-3 factors that mattered most, saying whether each value was high, low, or \
typical compared with other patients. Avoid jargon; if you use a medical term, explain it briefly.
- Mention a factor that pointed the other way when there is one.
- If confidence is low, or the prediction disagrees with the recorded risk level, say so plainly.
- Keep answers short: one short paragraph for a single patient, unless asked for more.

Important context:
- "Risk level" in this dataset reflects how severe and how likely bacterial a case looks. \
It does not measure the chance of death; do not describe it that way.
- The model finds patterns; it does not show cause and effect. Say "the model associates", \
not "this causes".
- This is a teaching dataset with some unrealistic values. These explanations describe the \
model, not real medical advice; never recommend treatment.
- Effects are SHAP values: positive pushes toward the predicted level, negative away from it. \
Do not quote raw SHAP numbers to the reader; describe them as stronger or weaker influences.
- Present factors in the order the tool returns them (strongest first); don't reorder them.
- If a patient's result says in_training_data is true, the model saw this patient while \
learning, so its confidence may be higher than it would be for a new patient. Mention this briefly.
- For any question about how many patients fall in a group, use count_patients for exact \
numbers; never estimate counts from percentages.
- For questions covering many patients, use compact tables and keep it concise."""

TOOLS = [
    {
        "name": "explain_patient",
        "description": "Predict one patient's risk level and return the factors that drove the "
                       "prediction, with each value described as high, low, or typical. Use for "
                       "any question about a specific patient.",
        "input_schema": {
            "type": "object",
            "properties": {
                "patient_id": {"type": "integer", "description": "The patient's row index."},
            },
            "required": ["patient_id"],
        },
    },
    {
        "name": "what_if",
        "description": "Re-run the prediction for a patient with some values changed, to answer "
                       "'what would change their risk level?'. Feature names: " + ", ".join(RISK_FEATURES)
                       + ". Binary features take 0 or 1; when changing Diagnosis or Gender, set "
                       "the old category to 0 and the new one to 1.",
        "input_schema": {
            "type": "object",
            "properties": {
                "patient_id": {"type": "integer"},
                "changes": {"type": "object", "description": "Map of feature name to new value.",
                            "additionalProperties": {"type": "number"}},
            },
            "required": ["patient_id", "changes"],
        },
    },
    {
        "name": "global_drivers",
        "description": "The factors that matter most overall for each risk level, across all "
                       "patients. Use for general questions like 'what drives high risk?'.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "count_patients",
        "description": "Exact patient counts by risk level, split by another variable "
                       "(Diagnosis, Pathogen_Present, Gender, Outcome, or Age_Group). Use for any "
                       "'how many' question. Set use='predicted' for the model's predictions "
                       "instead of the recorded risk levels.",
        "input_schema": {
            "type": "object",
            "properties": {
                "group_by": {"type": "string",
                             "enum": ["Diagnosis", "Pathogen_Present", "Gender", "Outcome", "Age_Group"]},
                "use": {"type": "string", "enum": ["recorded", "predicted"]},
            },
            "required": ["group_by"],
        },
    },
    {
        "name": "typical_values",
        "description": "Typical (median) lab values and the share of patients with a detected "
                       "pathogen or bacterial diagnosis, for each recorded risk level.",
        "input_schema": {"type": "object", "properties": {}},
    },
]


class RiskExplainerAgent:
    """A Claude agent that answers questions by calling the RiskExplainer's tools."""

    def __init__(self, explainer, model=MODEL_NAME, max_tokens=8000, max_steps=12,
                 max_continuations=2):
        import anthropic  # imported here so the model layer works without the SDK
        self.client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment
        self.explainer = explainer
        self.model = model
        self.max_tokens = max_tokens
        self.max_steps = max_steps
        self.max_continuations = max_continuations
        self.messages = []  # conversation history, so follow-up questions work
        self.last_stop_reason = None
        self.last_output_tokens = None

    def _run_tool(self, name, args):
        try:
            if name == "explain_patient":
                result = self.explainer.explain_patient(int(args["patient_id"]))
            elif name == "what_if":
                result = self.explainer.explain_patient(int(args["patient_id"]),
                                                        overrides=args.get("changes", {}))
            elif name == "global_drivers":
                result = self.explainer.global_drivers()
            elif name == "typical_values":
                result = self.explainer.typical_values()
            elif name == "count_patients":
                result = self.explainer.count_patients(args.get("group_by", "Diagnosis"),
                                                       args.get("use", "recorded"))
            else:
                result = {"error": f"Unknown tool {name}"}
        except Exception as exc:  # report errors back to the model instead of crashing
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

            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    if verbose:
                        print(f"[tool] {block.name}({block.input})")
                    tool_results.append({"type": "tool_result", "tool_use_id": block.id,
                                         "content": self._run_tool(block.name, block.input)})
            self.messages.append({"role": "user", "content": tool_results})
        return "".join(parts) + "\n\n[Stopped: the agent reached its step limit.]"

    def reset(self):
        self.messages = []
