# toxic-comment-classification-mlflow-demo

## MLFlow demo

```sh
python3 -m venv venv
source venv/bin/activate

pip install --upgrade pip
pip install pandas scikit-learn mlflow fastapi uvicorn requests

export MLFLOW_TRACKING_URI=http://localhost:5000

python3 train.py

mlflow ui

uvicorn serve:app --reload


curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Screw you"}'
```
