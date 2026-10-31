# Firebase + Google Cloud Run deployment

## Firebase setup
1. Create a Firebase project at https://console.firebase.google.com/.
2. Add a Firestore database in production mode.
3. Keep Firestore client rules locked (the supplied `firebase/firestore.rules` denies direct client access).
4. In Google Cloud Console, enable Firestore API, Cloud Run API, Cloud Build API and Secret Manager API.
5. Create a service account for the backend with only the required Firestore permissions (e.g. Cloud Datastore User).
6. Store the service-account JSON in Secret Manager. Do not commit it or paste it into GitHub.
7. Set Cloud Run environment variables:
   - `FIREBASE_PROJECT_ID`
   - `FIREBASE_SERVICE_ACCOUNT_JSON` (prefer mount/injected secret; if using env var, supply the full JSON string)
   - `GEMINI_API_KEY` (optional; also use Secret Manager)
8. Backend writes audit records to `audit_events`; SQLite remains a local fallback.

## Local backend
```bash
cd backend
python -m venv .venv
# activate venv, then:
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```
For local Firebase, set the environment variables in your shell. If not set, SQLite is used.

## Cloud Run backend
From repository root:
```bash
gcloud builds submit --tag gcr.io/PROJECT_ID/covenguard-backend backend
gcloud run deploy covenguard-backend \
  --image gcr.io/PROJECT_ID/covenguard-backend \
  --region asia-south1 \
  --allow-unauthenticated \
  --port 8080
```
For a public hackathon demo, protect admin/attack endpoints before production. The starter currently has no authentication; do not expose it as a production service.

## Frontend
Build and deploy frontend to Vercel or Firebase Hosting. Set `VITE_API_BASE_URL` to the deployed backend URL (e.g. `https://YOUR-CLOUD-RUN-URL`).
If hosting frontend and backend behind one domain using the supplied Nginx proxy, leave `VITE_API_BASE_URL` empty so the app calls `/api/...` on the same origin.

## Gemini
The optional `/api/ai/explain` endpoint uses Gemini 2.5 Flash. Set `GEMINI_API_KEY` as a server-side secret. It receives structured anomaly evidence only, not medical images or raw hospital data.

## Firestore data model
Collection: `audit_events`
- `event`: string
- `details`: map
- `timestamp`: string

Do not store NPZ archives, images, model checkpoints or patient data in Firestore.
