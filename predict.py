import sys
import requests

if len(sys.argv) != 2:
    print("Usage: python predict.py "your comment here"")
    exit(1)

text = sys.argv[1]
response = requests.post("http://localhost:8000/predict", json={"text": text})

if response.status_code == 200:
    result = response.json()
    print(f"Text: {text}")
    print(f"Prediction: {result['label']} (raw={result['prediction']})")
else:
    print("Error:", response.text)
