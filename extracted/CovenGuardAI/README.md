# COVENGUARD AI
**Byzantine-robust federated learning security for a simulated hospital network.**

A 3-hour hackathon MVP with a Halo-inspired sci-fi command interface (original visual assets; no game logos/assets).

## What works
- 10 simulated hospital clients with separate local partitions.
- Optional upload of the official MedMNIST `pneumoniamnist.npz`.
- A lightweight NumPy logistic-regression training loop (actual local training on image features).
- FedAvg, coordinate-wise median and trimmed-mean aggregation.
- Random-noise and sign-flip model-update poisoning simulations.
- Multi-metric inspection: L2 norm, cosine similarity and robust deviation.
- Explainable flagging, quarantine state and per-client trust ledger.
- SQLite incident/audit persistence.
- Dashboard with live experiment results and network topology.

## Important scope / honest claims
This is a **research/demo prototype**, not a clinically validated triage system or production security product.
- The demo's image classifier uses downsampled grayscale image features and binary labels.
- It does not diagnose patients or make clinical decisions.
- Anomaly detection is heuristic and can produce false positives/negatives.
- Coordinate-wise median / trimmed mean reduce some outlier influence; they do not guarantee defence against every Byzantine strategy.
- The default built-in data is synthetic. Upload PneumoniaMNIST for a real public benchmark experiment.
- Raw images stay in the backend process; only model updates are aggregated. This local single-machine simulation is not a privacy proof.

## Requirements
- Python 3.10+
- Node.js 18+
- npm

## Run locally

### 1. Backend
```bash
cd backend
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```
API docs: http://localhost:8000/docs

### 2. Frontend (new terminal)
```bash
cd frontend
npm install
npm run dev
```
Open the Vite URL (usually http://localhost:5173).

## Dataset
Download the PneumoniaMNIST NPZ from the official MedMNIST/Zenodo distribution:
- https://medmnist.com/
- https://zenodo.org/records/10519652

In the UI, choose **Load PneumoniaMNIST NPZ** and upload `pneumoniamnist.npz`.
Expected arrays: `train_images`, `train_labels`, `val_images`, `val_labels`, `test_images`, `test_labels`.
The app validates the archive and uses training images for client partitions. Validation/test remain held out.
If your archive uses different keys, check the official version and adapt `backend/app/dataset.py`.

## API
- `GET /api/health`
- `GET /api/state`
- `POST /api/dataset/upload` (multipart file)
- `POST /api/experiment/run` JSON: `{ "attack": "none|sign_flip|noise", "aggregation": "fedavg|median|trimmed_mean", "malicious_client": 7, "rounds": 1 }`
- `POST /api/reset`

## GitHub
Create a repository, then:
```bash
git init
git add .
git commit -m "Build CovenGuard AI federated security MVP"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/CovenGuardAI.git
git push -u origin main
```
Never commit `.env`, API keys, private patient data, or large restricted datasets. This project includes `.gitignore`.

## 3-hour build order
1. 0–20 min: install/run backend and frontend.
2. 20–50 min: upload NPZ and verify dataset counts.
3. 50–90 min: run honest FedAvg baseline and save metrics.
4. 90–125 min: run sign-flip attack; compare FedAvg vs median/trimmed mean.
5. 125–155 min: polish demo flow, verify logs and screenshots.
6. 155–180 min: README, GitHub push, rehearse 90-second pitch.

## Demo script
1. Show 10 healthy hospitals.
2. Run honest baseline.
3. Inject sign-flip attack at Hospital 07.
4. Show L2/cosine evidence and quarantine decision.
5. Compare FedAvg with robust aggregation.
6. Show test-set metrics and audit trail.
