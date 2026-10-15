# Day 4 MLOps: Modular FastAPI + Docker

This example is intended for classroom training only. It uses **synthetic** delivery data and is not a real delivery risk service.

Open **Day4_Modular_Student_Lab_Guide.md** and follow Labs 1–10 in order.

Quick start from the `mlops-day4/` directory (Windows Git Bash):

```bash
python -m venv .venv
source .venv/Scripts/activate
python -m pip install -r requirements.txt
python -m scripts.create_dataset
python -m scripts.validate_data
python -m scripts.train_model
python -m pytest tests/ -v
python -m uvicorn app:app --reload
```

Visit http://127.0.0.1:8000/docs; stop the development server before launching Docker.

```bash
docker build -t delivery-api:1.0 .
docker run -d --name delivery-container -p 127.0.0.1:8000:8000 delivery-api:1.0
```

This package includes a saved model for the introductory API demo. Regenerate it using `python -m scripts.train_model`. Never load untrusted Joblib files.
