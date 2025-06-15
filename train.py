import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn
import joblib

# Load data
df = pd.read_csv("train.csv")

# Prepare data for binary classification: label as 1 if any toxicity column is 1, else 0
toxicity_cols = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
df["label"] = (df[toxicity_cols].sum(axis=1) > 0).astype(int)
X = df["comment_text"]
y = df["label"]
# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

# Vectorize
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train model
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# Predict & evaluate
y_pred = model.predict(X_test_vec)
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc:.2f}")

# Log with MLflow
with mlflow.start_run() as run:
    mlflow.log_param("vectorizer", "TfidfVectorizer")
    mlflow.log_param("model", "LogisticRegression")
    mlflow.log_metric("accuracy", acc)

    mlflow.sklearn.log_model(model, "model")
    joblib.dump(vectorizer, "vectorizer.pkl")
    mlflow.log_artifact("vectorizer.pkl")

    print("Run ID:", run.info.run_id)
