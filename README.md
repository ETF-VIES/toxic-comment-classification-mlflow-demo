# toxic-comment-classification-mlflow-demo

## MLFlow demo

```sh
python3 -m venv venv
source venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

mlflow ui
```

```sh
source venv/bin/activate
export MLFLOW_TRACKING_URI=http://localhost:5000
python3 train.py
```

```sh
vim serve.py
source venv/bin/activate
uvicorn serve:app --reload
```

```sh
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Screw you"}'
```
