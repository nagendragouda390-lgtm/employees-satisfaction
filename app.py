import json
import pickle
from pathlib import Path

import pandas as pd
from flask import Flask, render_template, request, redirect, url_for

BASE_DIR = Path(__file__).resolve().parent

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Load trained pipeline (preprocessing + model bundled together) and metadata
# ---------------------------------------------------------------------------
with open(BASE_DIR / "models" / "model.pkl", "rb") as f:
    MODEL = pickle.load(f)

with open(BASE_DIR / "models" / "meta.json", "r") as f:
    META = json.load(f)

FEATURE_ORDER = META["feature_order"]
CAT_COLS = set(META["cat_cols"])
NUM_COLS = set(META["num_cols"])

# Human friendly labels + dropdown "options" for every field on the form.
# Rating-scale numeric columns (1-4 / 1-5) get a select with descriptive
# labels instead of a bare number box -- this is the "label column" option
# the form uses so users pick a meaningful option rather than a raw number.
RATING_SCALE_LABELS = {
    1: "1 - Low",
    2: "2 - Medium",
    3: "3 - High",
    4: "4 - Very High",
}
EDUCATION_LABELS = {
    1: "1 - Below College",
    2: "2 - College",
    3: "3 - Bachelor",
    4: "4 - Master",
    5: "5 - Doctor",
}
PERFORMANCE_LABELS = {
    3: "3 - Excellent",
    4: "4 - Outstanding",
}

RATING_FIELDS = {
    "EnvironmentSatisfaction": RATING_SCALE_LABELS,
    "JobInvolvement": RATING_SCALE_LABELS,
    "JobSatisfaction": RATING_SCALE_LABELS,
    "RelationshipSatisfaction": RATING_SCALE_LABELS,
    "WorkLifeBalance": RATING_SCALE_LABELS,
    "Education": EDUCATION_LABELS,
    "PerformanceRating": PERFORMANCE_LABELS,
}

FIELD_LABELS = {
    "Age": "Age",
    "BusinessTravel": "Business Travel Frequency",
    "DailyRate": "Daily Rate ($)",
    "Department": "Department",
    "DistanceFromHome": "Distance From Home (km)",
    "Education": "Education Level",
    "EducationField": "Field of Education",
    "EnvironmentSatisfaction": "Environment Satisfaction",
    "Gender": "Gender",
    "HourlyRate": "Hourly Rate ($)",
    "JobInvolvement": "Job Involvement",
    "JobLevel": "Job Level",
    "JobRole": "Job Role",
    "JobSatisfaction": "Job Satisfaction",
    "MaritalStatus": "Marital Status",
    "MonthlyIncome": "Monthly Income ($)",
    "MonthlyRate": "Monthly Rate ($)",
    "NumCompaniesWorked": "Number of Companies Worked",
    "OverTime": "Works Overtime",
    "PercentSalaryHike": "Percent Salary Hike (%)",
    "PerformanceRating": "Performance Rating",
    "RelationshipSatisfaction": "Relationship Satisfaction",
    "StockOptionLevel": "Stock Option Level",
    "TotalWorkingYears": "Total Working Years",
    "TrainingTimesLastYear": "Trainings Last Year",
    "WorkLifeBalance": "Work-Life Balance",
    "YearsAtCompany": "Years At Company",
    "YearsInCurrentRole": "Years In Current Role",
    "YearsSinceLastPromotion": "Years Since Last Promotion",
    "YearsWithCurrManager": "Years With Current Manager",
}


def build_form_fields():
    """Builds an ordered list of field descriptors used to render index.html."""
    fields = []
    for col in FEATURE_ORDER:
        label = FIELD_LABELS.get(col, col)
        if col in CAT_COLS:
            fields.append({
                "name": col,
                "label": label,
                "type": "select",
                "options": [{"value": v, "label": v} for v in META["cat_options"][col]],
            })
        elif col in RATING_FIELDS:
            labels = RATING_FIELDS[col]
            fields.append({
                "name": col,
                "label": label,
                "type": "select",
                "options": [{"value": k, "label": v} for k, v in labels.items()],
            })
        else:
            lo, hi, default = META["num_ranges"][col]
            fields.append({
                "name": col,
                "label": label,
                "type": "number",
                "min": lo,
                "max": hi,
                "default": default,
            })
    return fields


FORM_FIELDS = build_form_fields()


@app.route("/")
def index():
    return render_template("index.html", fields=FORM_FIELDS)


@app.route("/predict", methods=["POST"])
def predict():
    row = {}
    for col in FEATURE_ORDER:
        raw = request.form.get(col, "")
        if col in CAT_COLS:
            row[col] = raw
        else:
            try:
                row[col] = float(raw)
            except ValueError:
                row[col] = 0.0

    X = pd.DataFrame([row], columns=FEATURE_ORDER)

    proba = float(MODEL.predict_proba(X)[0][1])
    prediction = "Yes" if proba >= 0.5 else "No"
    risk_pct = round(proba * 100, 1)

    if risk_pct >= 60:
        risk_level = "High"
    elif risk_pct >= 30:
        risk_level = "Moderate"
    else:
        risk_level = "Low"

    return render_template(
        "result.html",
        prediction=prediction,
        risk_pct=risk_pct,
        risk_level=risk_level,
        employee=row,
        field_labels=FIELD_LABELS,
    )


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
