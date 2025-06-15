FROM python:3.10-slim
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
WORKDIR /app
COPY . /app/
RUN pip install --upgrade pip && \ 
    pip install pandas scikit-learn mlflow fastapi uvicorn requests
CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8000"]
