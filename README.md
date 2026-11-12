# Solix_Hackathon_Repository
Here is the notebook and agents that I built for the Solix hackathon.

## Overview
Using a messy meningitis patient dataset (1,200 patients), I built an end-to-end workflow:

1. Cleaned the data and investigated why values were missing.
2. Tested whether lab results differ by diagnosis and pathogen status.
3. Built two classification models: one predicts whether a pathogen was present, the other whether a case is bacterial or viral.
4. Built two Claude-powered AI agents:
   - one explains each patient's risk level in plain language
   - one helps clinicians review the 136 cases with an unknown diagnosis

## Repository Contents

| File | Description |
|---|---|
| `Solix_Hackathon_Notebook.ipynb` | Main notebook: cleaning, hypothesis testing, EDA, classification models and the agent demos |
| `mening_missing.csv` | Raw dataset (1,200 patients, 14 columns) |
| `meningitis_cleaned_data.csv` | Cleaned dataset (1,182 patients, 21 columns) |
| `risk_explainer_agent.py` | Risk Level Explanation Agent |
| `unknown_diagnosis_agent.py` | Unknown Diagnosis Review Agent |
| `unknown_review_queue.csv` | Ranked review queue of the 136 unknown-diagnosis cases, exported by the review agent |

## 1. Data Cleaning
- **Removed PII:** dropped `Patient_ID`.
- **Removed impossible values:**
  - ages over 110 and spinal fluid protein of 0 were set to missing
  - a glucose of 0 was kept for bacterial cases, where glucose can be undetectable, but set to missing for viral cases, where glucose is usually normal
- **Added `CRP_Level_Below_Detection`:** flags the 75 patients whose CRP is exactly 0.
- **Investigated the missing data:**
  - missing `Pathogen_Present` (8) and `Outcome` (10) values each sat in one consecutive block of records
  - compared missing rows against every other variable (Mann-Whitney U for numeric columns, chi-square for categorical) and found no significant association after Bonferroni correction, consistent with the values being missing at random
- **Handled the missing values:**
  - dropped the rows missing `Pathogen_Present` or `Outcome`, because both are target variables and targets shouldn't be imputed
  - filled numeric gaps with each column's median, keeping an `Age_imputed` flag
  - filled missing `Gender` and `Diagnosis` with "Unknown"
- **Engineered features:**
  - `Age_Group`, from Infant (<1) through Older adult (65+)
  - `Is_Minor`
  - one-hot encoded `Gender` and `Diagnosis`
  - converted `Pathogen_Present` to boolean and fixed integer types

## 2. Hypothesis Testing
Mann-Whitney U tests on four lab values (glucose, CRP, spinal fluid WBC and protein), with a Bonferroni-corrected α = 0.0125:

| Comparison | Result |
|---|---|
| Bacterial vs. viral | All four labs differ significantly (p < 1e-85). Bacterial cases have lower glucose and higher CRP, WBC and protein. |
| Pathogen detected vs. not | All four differ significantly (p < 1e-71), in the same direction. |
| Unknown vs. known diagnosis | All four differ significantly. Unknown cases show higher glucose, CRP, WBC and protein than known cases. |

## 3. Exploratory Data Analysis
- **Correlations:** Spearman correlation heatmap of the numeric variables.
- **Distributions:** distribution plots of each lab value by diagnosis and pathogen status.
- **Deaths by age group:** adults 25–64 account for 62.8% of deaths.
- **Pathogen detection rates:** broken down by age group and by gender.
- **Other charts:** recoveries by age group, and platelets vs. hemoglobin by pathogen status.

## 4. Classification Models
Both models use the same features: glucose, CRP, spinal fluid WBC, protein, age and gender. Each used an 80/20 stratified train/test split and 5-fold stratified cross-validation. I compared Logistic Regression, Decision Tree, KNN, SVM, Random Forest, Histogram Gradient Boosting and XGBoost.

### Model A: Was a pathogen present?
- **Best model:** Random Forest, CV ROC AUC **0.956**. Tuning `min_samples_leaf` and `max_depth` didn't beat the base model.
- **Decision threshold:** lowered to **0.40** to favor catching true positives.
- **Test set results:** ROC AUC **0.945**, recall for detected pathogens **0.96**, accuracy 0.85.
- **Top features:** glucose (0.32), CRP (0.27), WBC (0.20), protein (0.15).

### Model B: Bacterial or viral? (known diagnoses only, 1,046 patients)
- **Best model:** XGBoost, CV ROC AUC **0.975**, tuned with `RandomizedSearchCV`.
- **Decision threshold:** lowered to **0.35** to prioritize catching bacterial cases, which are the more dangerous ones.
- **Test set results:** ROC AUC **0.980**, bacterial recall **0.94**, accuracy 0.91.
- **Top features:** CRP (0.32), WBC (0.22), glucose (0.22), protein (0.14).

## 5. AI Agents
Both agents use the Anthropic API (`claude-sonnet-5-5`) with tool use. The model runs the calculations, and Claude turns the results into plain-language answers for a non-technical audience. Each one also has a rule-based mode that works without an API key.

### Risk Level Explanation Agent (`risk_explainer_agent.py`)
- **Model:** a Random Forest predicts each patient's `Risk_Level` (High / Moderate / Low), with a balanced accuracy of **0.86** on the test set.
- **Explanations:** uses **SHAP** to find which factors drove each prediction, comparing the patient's values against the rest of the dataset.
- **Tools available to the agent:**
  - `explain_patient`: why a patient got their risk level
  - `what_if`: what would need to change for a different risk level
  - `global_drivers`: which factors matter most across all patients
  - `count_patients`: patient counts by group
  - `typical_values`: typical lab values by risk level
- **Example questions:** "Why was patient 42 rated Low Risk?", "What is the breakdown of High Risk patients by diagnosis?"

### Unknown Diagnosis Review Agent (`unknown_diagnosis_agent.py`)
- **Purpose:** investigates why **136 cases** have an unknown diagnosis and ranks which ones most need a clinician's review.
- **Approach:** trains a bacterial-vs-viral model on the known cases (CV ROC AUC **0.975**) and scores each unknown case.
- **What it checks for each unknown case:**
  - **Lab/test conflict:** the labs point one way but the pathogen test points the other
  - **Atypical values:** a lab value outside the usual range for both bacterial and viral cases
  - **Mixed lab signals:** some labs look bacterial and others look viral
  - **Borderline profile:** the model is close to 50/50
  - **Clear profile:** everything points one way
- **Results:**
  - 72 unknown cases look more bacterial and 64 look more viral
  - 125 of the 136 are flagged for review
  - "Atypical values" is the most common primary reason (81 cases)
- **Output:** exports the ranked queue to `unknown_review_queue.csv`.
- **Tools available to the agent:** `cohort_overview`, `review_queue`, `assess_case`, `compare_lab`.

## Running the Notebook
1. Install the dependencies: `pandas`, `numpy`, `scipy`, `scikit-learn`, `xgboost`, `shap`, `matplotlib`, `seaborn`, `python-dotenv` and `anthropic`.
2. To use the Claude agents, add `ANTHROPIC_API_KEY=...` to a `.env` file in the project folder. The rule-based explanations work without a key.
3. Open `Solix_Hackathon_Notebook.ipynb` and run the cells in order.

## Tools
Python, pandas, NumPy, SciPy, scikit-learn, XGBoost, SHAP, matplotlib, seaborn, Anthropic API (Claude), Jupyter Notebook
