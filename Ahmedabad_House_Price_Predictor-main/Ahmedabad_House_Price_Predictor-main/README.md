
````markdown
# 🏠 Ahmedabad House Price Predictor

This is my Ahmedabad house price predictor. I built it so anyone can quickly predict house prices using a simple web app. 💻⚡

- Frontend: Streamlit
- Backend: FastAPI
- ML Model: XGBoost
- Pipeline: Prefect + joblib
- Deployment (Optional): Render (Frontend + Backend)

---

# 🚀 Quick Start (Instant Vibe Check)

Wanna see it live? This is the fastest way to get the whole stack—frontend and backend—running on your machine with **Docker**.

1.  Clone the repo:
    
    git clone [YOUR_REPO_URL]
    cd Ahmedabad_House_Price_Predictor/docker
    
2.  Build & Run the Containers:
    
    docker compose up --build
  
3.  Check the App:
    -   Frontend UI: `http://localhost:8501`
    -   Backend API: `http://localhost:8000/predict` (for health check: `http://localhost:8000/health`)



## What it does

- Predict house price using sqft, BHK, location, floor, price per sqft
- Backend API using FastAPI
- Frontend UI with Streamlit
- Auto preprocessing + ML pipeline with Prefect
- Shows EDA plot of top 10 expensive locations
- Logs RMSE & feature importance with MLflow

---

## Repo Structure

````

Ahmedabad\_House\_Price\_Predictor/
- │
- ├─ backend/
- │   ├─ main.py                   \# FastAPI app
- │   ├─ train\_pipeline\_prefect.py \# ML pipeline
- │   ├─ models/
- │   │    └─ house\_price\_pipeline\_prefect.pkl
- │   └─ requirements.txt
- │
- ├─ frontend/
- │   ├─ streamlit\_app.py          \# Streamlit UI
- │   └─ requirements.txt
- │
- ├─ data/
- │   └─ ahmedabad\_cleaned.csv     \# Cleaned dataset
- │
- ├─ docker/
- │   ├─ Dockerfile\_backend
- │   ├─ Dockerfile\_frontend
- │   └─ docker-compose.yml
- │
- └─ README.md

````

---

## **Backend (FastAPI)**

Go to backend folder:

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
````

  - API URL: `http://localhost:8000/predict`
  - Health check: `http://localhost:8000/health`

Sample JSON to test the prediction endpoint:

```json
{
  "total_sqft": 1200,
  "bhk": 3,
  "location": "Navrangpura",
  "floor_num": 1,
  "price_sqft": 3765
}
```

-----

## **Frontend (Streamlit)**

Go to frontend folder:

```bash
cd frontend
pip install -r requirements.txt
streamlit run streamlit_app.py
```

  - Open `http://localhost:8501`
  - **Heads up:** Make sure `API_URL` in `streamlit_app.py` points to your backend (e.g., `http://localhost:8000/predict`).

-----

## **ML Pipeline**

  - **Dataset:** `data/ahmedabad_cleaned.csv`
  - **EDA:** Top 10 locations by average price $\rightarrow$ `eda_top10_locations.png`
  - **Preprocessing:** `StandardScaler` + `OneHotEncoder`
  - **Model:** `XGBoost Regressor`
  - **Pipeline saved as:** `backend/models/house_price_pipeline_prefect.pkl`
  - Logs RMSE & feature importance (`feature_importance.csv`)

Run the Prefect flow to train and save the model:

```bash
python backend/train_pipeline_prefect.py
```

-----

## **Docker (The Easy Way)**

### Backend Dockerfile (`docker/Dockerfile_backend`)

```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Frontend Dockerfile (`docker/Dockerfile_frontend`)

```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### Docker Compose (`docker/docker-compose.yml`)

```yaml
services:
  backend:
    build:
      context: ../backend
      dockerfile: Dockerfile_backend
    ports:
      - "8000:8000"

  frontend:
    build:
      context: ../frontend
      dockerfile: Dockerfile_frontend
    ports:
      - "8501:8501"
```

To build and run:

```bash
cd docker
docker compose up --build
```

-----

## **Deployment** (Optional)

  - Backend $\rightarrow$ **Render**
  - Frontend $\rightarrow$ **Render**
  - Make sure the frontend Streamlit app points to your deployed backend API URL (not `localhost`).

-----

## **Extras**

  - Feature importance $\rightarrow$ `feature_importance.csv`
  - EDA plot $\rightarrow$ `eda_top10_locations.png`
  - Predict house prices in seconds 💸

-----


```
```
