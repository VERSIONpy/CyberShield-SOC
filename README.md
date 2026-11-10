# CyberShield SOC

CyberShield is a Streamlit dashboard for exploring network-flow threat detection with a Random Forest model trained on UNSW-NB15. It includes a FastAPI service and a bundled browser interface for local use as well.

**Live app:** [cybershield-ai-daqo.onrender.com](https://cybershield-ai-daqo.onrender.com/)

## Run the dashboard locally

Python 3.11 or 3.12 is recommended for the scikit-learn model artifacts.

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run streamlit_app.py
```

Open the URL printed by Streamlit (usually http://localhost:8501).

## Run the API and bundled web interface

```bash
python -m pip install -r requirements.txt
python server.py
```

Open http://localhost:8000. The interactive API documentation is at http://localhost:8000/docs.

## Deploy the dashboard

The FastAPI service and bundled web interface are hosted on Render using `render.yaml`. The Streamlit dashboard can also be hosted on [Streamlit Community Cloud](https://share.streamlit.io/): select this GitHub repository, set the app file to `streamlit_app.py`, and deploy.

The FastAPI service can be deployed separately on a Python web service using `uvicorn server:app --host 0.0.0.0 --port $PORT`. A Render configuration is included in `render.yaml`.

## What is included

- Random Forest binary classifier and preprocessing artifact.
- The UNSW-NB15 training and testing partitions used by the app.
- Streamlit SOC dashboard and FastAPI backend.
- Windows launcher (`start.bat`) for the API and bundled frontend.

The overview page contains fixed benchmark figures. Treat dashboard risk labels and feature-attribution display as a research/demo aid, not as operational security advice or a substitute for validating a model on your own traffic.

## Data and model notes

The bundled testing data is from UNSW-NB15. See the [UNSW Canberra Cyber dataset page](https://research.unsw.edu.au/projects/unsw-nb15-dataset) for dataset context and terms. The included model files are serialized scikit-learn artifacts; loading serialized models from an untrusted source can execute code. Use the files in this repository only if you trust its source.
