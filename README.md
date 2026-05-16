# Telco Customer Churn MLOps Pipeline 🚀

An end-to-end, production-grade MLOps pipeline for predicting customer churn using the Telco Churn dataset. This project demonstrates best practices in data validation, feature engineering, model versioning, monitoring, and cloud-native deployment.

## 🏗️ Architecture & Stack

- **Pipeline Orchestration**: [DVC](https://dvc.org/) (Data Version Control)
- **Data Validation**: [Great Expectations](https://greatexpectations.io/)
- **Feature Store**: [Feast](https://feast.dev/) with [ClickHouse](https://clickhouse.com/) offline store
- **Experiment Tracking**: [MLflow](https://mlflow.org/)
- **Data Lineage**: [OpenLineage](https://openlineage.io/) & [Marquez](https://marquezproject.io/)
- **Drift Monitoring**: [Evidently AI](https://www.evidentlyai.com/)
- **Serving Layer**: [FastAPI](https://fastapi.tiangolo.com/) (Prometheus instrumented)
- **Frontend**: [React](https://reactjs.org/) Dashboard
- **Infrastructure**: Docker Compose, Kubernetes, ArgoCD
- **Testing**: Pytest & Promptfoo

---

## 📁 Project Structure

```text
.
├── api/                # FastAPI serving application
├── argocd/             # ArgoCD application manifests
├── artifacts/          # Model binaries, metrics, and reports
├── data/               # Local data storage (DVC tracked)
├── docker/             # Dockerfiles and infrastructure configs
├── frontend/           # React dashboard application
├── k8s/                # Kubernetes manifests (deployments, services)
├── monitoring/         # Prometheus/Grafana configurations
├── src/                # Core pipeline source code
│   ├── data/           # Ingestion, validation, preprocessing
│   ├── features/       # Feature store definitions
│   ├── models/         # Training and evaluation scripts
│   └── monitoring/     # Drift detection logic
├── tests/              # Unit and integration tests
├── dvc.yaml            # DVC pipeline definition
├── params.yaml         # Project parameters
└── requirements.txt    # Python dependencies
```

---

## 🚀 Getting Started

### 1. Environment Setup
```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
.\venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Start Infrastructure
Launch the core services (Database, MLflow, Monitoring, Lineage) using Docker Compose:
```bash
docker-compose -f docker/docker-compose.yml up -d
```

### 3. Run the MLOps Pipeline
Reproduce the entire lifecycle (ingestion -> evaluation) with a single command:
```bash
dvc repro
```

### 4. Start the API & Frontend
```bash
# Terminal 1: API
python -m uvicorn api.main:app --reload --port 8000

# Terminal 2: Frontend
cd frontend && npm start
```

---

## 🛠️ Tool Access Guide

| Tool | URL | Credentials | Description |
| :--- | :--- | :--- | :--- |
| **Prediction API** | [http://localhost:8000/docs](http://localhost:8000/docs) | - | Interactive Swagger UI for model serving. |
| **Frontend UI** | [http://localhost:3001](http://localhost:3001) | - | Customer Churn prediction dashboard. |
| **ArgoCD (GitOps)** | [https://localhost:8080](https://localhost:8080) | `admin` / `oKh8UsfXFab-SUkV` | GitOps deployment controller. |
| **MLflow** | [http://localhost:5001](http://localhost:5001) | - | Experiment tracking & model registry. |
| **Grafana** | [http://localhost:3002](http://localhost:3002) | `admin` / `admin` | System & Model monitoring dashboards. |
| **Prometheus** | [http://localhost:9090](http://localhost:9090) | - | Real-time metrics collection. |
| **Marquez (UI)** | [http://localhost:3000](http://localhost:3000) | - | Pipeline lineage & metadata visualization. |
| **ClickHouse** | [http://localhost:8123](http://localhost:8123) | `default` / (none) | High-performance OLAP database. |
| **Promptfoo UI** | [http://localhost:4000](http://localhost:4000) | - | Model behavior & segment evaluation UI. |
| **CI/CD** | [GitHub Actions](https://github.com/Spurthipendyala/Customerchurn_prediction/actions) | - | Automated build, test, and deploy pipeline. |

---

## 📊 Monitoring & Observability

- **API Metrics**: Exposed at `http://localhost:8000/metrics` for Prometheus.
- **Data Drift**: Generate latest reports via `python -m src.monitoring.drift_detection`.
- **Model Registry**: View experiments at `http://localhost:5001`.
- **Lineage**: Pipeline runs emit OpenLineage events to Marquez.

---

## ☁️ Deployment

### Kubernetes (Production)
```bash
kubectl apply -f k8s/
```
The project is configured for **GitOps** via ArgoCD. Apply the application manifest to enable automated synchronization:
```bash
kubectl apply -f argocd/application.yaml
```

---

## 🧪 Testing
```bash
# Run unit tests
pytest tests/

# Run API behavior evaluation
promptfoo eval
```

---
*Built with ❤️ for High-Performance MLOps.*
