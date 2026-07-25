import json
import pickle
import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd
from flask import Flask, render_template, request, redirect, url_for

BASE_DIR = Path(__file__).resolve().parent
DB_DIR = BASE_DIR / "data"
DB_DIR.mkdir(exist_ok=True)
DB_PATH = DB_DIR / "app.db"

app = Flask(__name__)


# ---------------------------------------------------------------------------
# Tiny SQLite layer for visit counting + prediction history.
#
# NOTE: On Render's free web service plan, the filesystem is ephemeral --
# app.db will reset whenever the service redeploys or restarts (it does NOT
# reset on every request, just on deploys). That's fine for a demo/portfolio
# project. For counts/history that must survive redeploys, swap this for a
# hosted database (e.g. Render's free Postgres) and point the DB calls there.
# ---------------------------------------------------------------------------

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS visits (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            visited_at TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            predicted_at TEXT NOT NULL,
            prediction TEXT NOT NULL,
            risk_pct REAL NOT NULL,
            risk_level TEXT NOT NULL,
            profile_json TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()


def log_visit():
    conn = get_db()
    conn.execute("INSERT INTO visits (visited_at) VALUES (?)", (datetime.utcnow().isoformat(),))
    conn.commit()
    conn.close()


def get_visit_count():
    conn = get_db()
    count = conn.execute("SELECT COUNT(*) AS c FROM visits").fetchone()["c"]
    conn.close()
    return count


def log_prediction(prediction, risk_pct, risk_level, profile):
    conn = get_db()
    conn.execute(
        "INSERT INTO predictions (predicted_at, prediction, risk_pct, risk_level, profile_json) "
        "VALUES (?, ?, ?, ?, ?)",
        (datetime.utcnow().isoformat(), prediction, risk_pct, risk_level, json.dumps(profile)),
    )
    conn.commit()
    conn.close()


def get_recent_predictions(limit=50):
    conn = get_db()
    rows = conn.execute(
        "SELECT * FROM predictions ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return rows


def get_prediction_count():
    conn = get_db()
    count = conn.execute("SELECT COUNT(*) AS c FROM predictions").fetchone()["c"]
    conn.close()
    return count


init_db()

# ---------------------------------------------------------------------------
# Load trained pipeline (preprocessing + model bundled together) and metadata
# ---------------------------------------------------------------------------
with open(BASE_DIR / "model" / "model.pkl", "rb") as f:
    MODEL = pickle.load(f)

with open(BASE_DIR / "model" / "meta.json", "r") as f:
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
    log_visit()
    return render_template("index.html", fields=FORM_FIELDS, visit_count=get_visit_count())


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

    log_prediction(prediction, risk_pct, risk_level, row)

    return render_template(
        "result.html",
        prediction=prediction,
        risk_pct=risk_pct,
        risk_level=risk_level,
        employee=row,
        field_labels=FIELD_LABELS,
    )


@app.route("/history")
def history():
    rows = get_recent_predictions(limit=50)
    records = []
    for r in rows:
        records.append({
            "predicted_at": r["predicted_at"],
            "prediction": r["prediction"],
            "risk_pct": r["risk_pct"],
            "risk_level": r["risk_level"],
        })
    return render_template(
        "history.html",
        records=records,
        prediction_count=get_prediction_count(),
        visit_count=get_visit_count(),
    )


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
