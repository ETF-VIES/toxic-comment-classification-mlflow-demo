from fastapi import FastAPI, Request
from pydantic import BaseModel
import mlflow.sklearn
import joblib

app = FastAPI()

# Replace this with your actual run ID from MLflow
RUN_ID = "86cea025a5a24930876db9dc4730c50d"
MODEL_ID = "m-963fae582da4404e825ee99ceb4c4411"

model = mlflow.sklearn.load_model(f"mlartifacts/0/models/{MODEL_ID}/artifacts")
vectorizer = joblib.load(f"mlartifacts/0/{RUN_ID}/artifacts/vectorizer.pkl")
class InputText(BaseModel):
    text: str

@app.post("/predict")
def predict(input: InputText):
    vec = vectorizer.transform([input.text])
    pred = model.predict(vec)[0]
    return {"prediction": int(pred), "label": "toxic" if pred else "not toxic"}
