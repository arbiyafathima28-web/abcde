# Firebase + Google Cloud Run Deployment Guide

## Active Firebase Provisioning Details

- **Project ID**: `gen-lang-client-0062959175`
- **Firestore Database ID**: `ai-studio-covenguardai-c8930497-30fb-44f9-87f8-5ca2011f6ac8`
- **Region**: `asia-southeast1`
- **Authorized Administrator Email**: `arbiyafathima975@gmail.com`

---

## Firestore Collections Schema

The database uses 8 structured collections:

| Collection | Path Pattern | Purpose & Access Model |
|---|---|---|
| `users` | `/users/{userId}` | Authenticated operator profiles and roles (`admin` or `viewer`). |
| `hospitals` | `/hospitals/{hospitalId}` | Real-time state of the 10 hospital nodes (trust scores, status). |
| `experiments` | `/experiments/{experimentId}` | Recorded federated rounds, evaluated test metrics, consensus method. |
| `federated_rounds` | `/federated_rounds/{roundId}` | Telemetry on participating vs quarantined client nodes. |
| `security_incidents` | `/security_incidents/{incidentId}` | Immutable Byzantine anomaly records (L2 norm, cosine similarity). |
| `audit_logs` | `/audit_logs/{logId}` | Append-only security journal entries for operator accountability. |
| `trust_history` | `/trust_history/{historyId}` | Historical trust score transitions with anomaly penalty reasons. |
| `model_metadata` | `/model_metadata/{metaId}` | Model dimensions, version numbers, and dataset benchmarks. |

---

## FastAPI Backend Integration & Token Validation

The FastAPI service in `backend/app/main.py` validates Firebase ID tokens via Bearer headers using the Firebase Admin SDK (`backend/app/auth.py`).

### Environment Variables on Google Cloud Run:
```bash
FIREBASE_PROJECT_ID="gen-lang-client-0062959175"
FIRESTORE_DATABASE_ID="ai-studio-covenguardai-c8930497-30fb-44f9-87f8-5ca2011f6ac8"
GEMINI_API_KEY="<YOUR_GEMINI_API_KEY>"
CORS_ORIGINS="*"
AUTH_REQUIRED="true"
```

### Local Development Run:
```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```
*Note: If Firebase credentials are not provided locally, the service automatically falls back to the embedded SQLite audit engine.*

---

## Cloud Run Backend Deployment

```bash
# Submit build to Google Container Registry / Artifact Registry:
gcloud builds submit --tag gcr.io/gen-lang-client-0062959175/covenguard-backend backend

# Deploy to Cloud Run:
gcloud run deploy covenguard-backend \
  --image gcr.io/gen-lang-client-0062959175/covenguard-backend \
  --region asia-southeast1 \
  --allow-unauthenticated \
  --set-env-vars FIREBASE_PROJECT_ID=gen-lang-client-0062959175,FIRESTORE_DATABASE_ID=ai-studio-covenguardai-c8930497-30fb-44f9-87f8-5ca2011f6ac8 \
  --port 8080
```

Once deployed, copy the Cloud Run service URL and paste it into the **Backend Config** panel in the CovenGuard AI HUD.
