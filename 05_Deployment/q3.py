import pickle
from pathlib import Path

from model import normalize_record

# Load the frozen pipeline (DictVectorizer + LogisticRegression)
with Path("pipeline.bin").open("rb") as f:
    pipeline = pickle.load(f)

lead = {
    "lead_source": "paid_ads",
    "industry": "technology",
    "employment_status": "employed",
    "location": "north_america",
    "number_of_courses_viewed": 2,
    "annual_income": 79276.0,
    "interaction_count": 4,
    "lead_score": 0.41,
}

# Same missing-value policy as training and the API (this lead is complete, so no change)
record = normalize_record(lead)

# The pipeline vectorizes the dict itself, so we pass a list with one record
probability = pipeline.predict_proba([record])[0, 1]

print(f"conversion probability: {probability:.6f}")
print(f"rounded to 3 decimals : {probability:.3f}")
