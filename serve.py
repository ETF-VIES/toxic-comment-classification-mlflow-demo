from fastapi import FastAPI, Request
from pydantic import BaseModel
import mlflow.sklearn
import joblib

app = FastAPI()

# Replace this with your actual run ID from MLflow
RUN_ID = "e5bd9b0902704c7d8a06a89544f0c8d2"
MODEL_ID = "m-9fdc8ba945e04a9387aaa4f64438a8ad"

model = mlflow.sklearn.load_model(f"mlartifacts/0/models/{MODEL_ID}/artifacts")
vectorizer = joblib.load(f"mlartifacts/0/{RUN_ID}/artifacts/vectorizer.pkl")
class InputText(BaseModel):
    text: str

@app.post("/predict")
def predict(input: InputText):
    vec = vectorizer.transform([input.text])
    pred = model.predict(vec)[0]
    return {"prediction": int(pred), "label": "toxic" if pred else "not toxic"}
