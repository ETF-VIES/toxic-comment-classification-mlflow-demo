from fastapi import FastAPI, Request
from pydantic import BaseModel
import mlflow.sklearn
import joblib

app = FastAPI()

# Replace this with your actual run ID from MLflow
RUN_ID = "f01bae019772482a98e68ccd8f366589"
MODEL_ID = "m-cebe2487dda4438b8f78577270ec83ea"

model = mlflow.sklearn.load_model(f"mlartifacts/0/models/{MODEL_ID}/artifacts")
vectorizer = joblib.load(f"mlartifacts/0/{RUN_ID}/artifacts/vectorizer.pkl")
class InputText(BaseModel):
    text: str

@app.post("/predict")
def predict(input: InputText):
    vec = vectorizer.transform([input.text])
    pred = model.predict(vec)[0]
    return {"prediction": int(pred), "label": "toxic" if pred else "not toxic"}
