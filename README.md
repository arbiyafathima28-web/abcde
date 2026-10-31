# COVENGUARD AI
**Byzantine-robust federated learning security for a distributed healthcare network.**

An educational research prototype featuring a Halo-inspired sci-fi tactical command HUD (original visual styling, dark graphite/navy palette, holographic cyan/teal telemetry, zero copyright assets).

---

## Architecture Overview

```
                      +---------------------------------------+
                      |       COVENGUARD AI TACTICAL HUD      |
                      |   React 19 + TypeScript + Tailwind   |
                      |   Radar Telemetry + Recharts Data    |
                      +-------------------+-------------------+
                                          |
                                          | HTTP JSON / Multipart (/api/*)
                                          v
                      +---------------------------------------+
                      |       PYTHON FASTAPI ML BACKEND       |
                      |          (Google Cloud Run)           |
                      +-------------------+-------------------+
                                          |
                +-------------------------+-------------------------+
                |                         |                         |
                v                         v                         v
     +--------------------+     +--------------------+    +--------------------+
     | 10 HOSPITAL NODES  |     | ANOMALY DETECTOR   |    | PERSISTENCE ENGINE |
     | - Local Partitions |     | - Finite Check     |    | - Firebase Admin   |
     | - PneumoniaMNIST   |     | - L2 Norm Dev      |    | - Firestore Audit  |
     | - Sign-Flip Attack |     | - Pairwise Cosine  |    | - SQLite Fallback  |
     | - Noise Poisoning  |     | - Robust MAD z     |    +--------------------+
     +--------------------+     | - Quarantine Logic |              |
                |               +--------------------+              v
                v                         |               +--------------------+
     +--------------------+               v               | GEMINI 2.5 ANALYST |
     | ROBUST AGGREGATION |     +--------------------+    | - Incident Brief   |
     | - Coordinate Median|     | DYNAMIC TRUST      |    | - False Positive   |
     | - Trimmed Mean     |     | - Per-Node Ledger  |    | - Audit Caveats    |
     | - FedAvg (Baseline)|     | - Quarantine State |    +--------------------+
     +--------------------+     +--------------------+
```

---

## What Works in Current Stage (Stage 1 Stabilized)

1. **10 Simulated Independent Hospital Clients**:
   - Each client receives an isolated partition of training data (PneumoniaMNIST NPZ or synthetic fallback benchmark).
   - Validation and test splits remain completely held out on the aggregator.

2. **Poisoning Simulation**:
   - Controlled sign-flipping (amplified gradient inversion) or Gaussian noise on designated client (e.g. Hospital 07).
   - Honest baseline option for clean calibration rounds.

3. **Multi-Metric Byzantine Anomaly Detection**:
   - Payload finite-value validation (rejects NaN/Inf gradient payloads).
   - L2 update norm calculation.
   - Pairwise cosine similarity matrix & mean directional alignment.
   - Robust Median Absolute Deviation (MAD) z-scoring to resist masking by extreme outliers.

4. **Quarantine & Dynamic Trust Ledger**:
   - Suspicious updates are automatically quarantined from the aggregation round.
   - Trust scores degrade upon quarantine and are tracked in the node registry.

5. **Byzantine-Robust Aggregation**:
   - **Coordinate-wise Median**: Evaluates median on every parameter dimension.
   - **Trimmed Mean**: Eliminates upper and lower extreme percentiles.
   - **FedAvg**: Standard baseline vulnerable to poisoning.

6. **Gemini Security Analyst**:
   - `/api/ai/explain` integration receives detector evidence JSON and returns structured incident briefings with false-positive caveats.

7. **Dual-Mode Persistence**:
   - Firebase Firestore in production (collections: `audit_logs`, `security_incidents`, `experiments`, etc.).
   - Automatic local SQLite fallback when Firebase credentials are not set.

---

## Local Setup & Run Commands

### 1. Backend (Python 3.10+)
```bash
cd backend
python -m venv .venv

# On Linux/macOS:
source .venv/bin/activate

# On Windows:
.venv\Scripts\activate

# Install dependencies:
pip install -r requirements.txt

# Start FastAPI server on port 8000:
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```
Interactive API docs: http://localhost:8000/docs

### 2. Frontend (Node.js 18+)
```bash
# In project root:
npm install
npm run dev
```
Open the Vite URL: http://localhost:3000 (or http://localhost:5173).

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/health` | Service health, active persistence mode, and dataset status |
| `GET` | `/api/state` | Current network topology, trust scores, round, and recent events |
| `POST` | `/api/dataset/upload` | Upload official MedMNIST `pneumoniamnist.npz` |
| `POST` | `/api/experiment/run` | Execute 1 federated training round with attack & aggregation options |
| `POST` | `/api/reset` | Reset network topology and telemetry to clean baseline |
| `POST` | `/api/ai/explain` | Request Gemini Security Analyst structured briefing |

---

## Verification & Automated Tests

To run the backend test suite:
```bash
cd backend
python -m unittest discover -s tests
```

---

## Important Scope & Research Caveats

- **Educational Prototype**: This is a research simulation demonstrating Byzantine fault tolerance and anomaly detection. It is **not** a clinical diagnostic tool or medical device.
- **Heuristic Detection**: Statistical and geometric metrics (MAD z-score, cosine similarity) are heuristics. In clinical networks with non-IID patient demographics, legitimate localized disease patterns can produce model updates that deviate from the median.
- **Privacy Notice**: In this single-process simulation, only model gradient updates are shared with the aggregator. Raw training images remain strictly within local partitions.
