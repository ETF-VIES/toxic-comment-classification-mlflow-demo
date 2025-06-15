from fastapi import FastAPI, Request
from pydantic import BaseModel
import mlflow.sklearn
import joblib

app = FastAPI()

# # Replace this with your actual run ID from MLflow
RUN_ID = "78a7485c9b394b3fa100764f5e0eb1e0"
model = mlflow.sklearn.load_model(f"mlartifacts/0/{RUN_ID}/artifacts/model")
vectorizer = joblib.load(f"mlartifacts/0/{RUN_ID}/artifacts/vectorizer.pkl")

class InputText(BaseModel):
    text: str

@app.post("/predict")
def predict(input: InputText):
    vec = vectorizer.transform([input.text])
    pred = model.predict(vec)[0]
    return {"prediction": int(pred), "label": "toxic" if pred else "not toxic"}
