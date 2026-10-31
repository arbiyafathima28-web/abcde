# Next prompts for your coding assistant

Use these one at a time. Ask it to inspect the existing repo before editing.

## Prompt 1 — Review and stabilize current MVP
"Inspect the complete CovenGuard AI repository. Run through frontend and backend integration. Fix import, type, CORS, API and build errors. Preserve current features. Add a README section for exact local run commands. Do not claim tests passed unless you run them. Report every changed file."

## Prompt 2 — Firebase integration
"Integrate Firebase Firestore into the existing Python FastAPI backend using Firebase Admin SDK. Use environment variables and Secret Manager-compatible configuration. Keep SQLite as local fallback. Persist audit events and experiment summaries only. Never store patient images or NPZ dataset in Firestore. Add startup health status showing active persistence backend. Do not commit credentials. Verify with a local fallback test and explain how to configure Firebase."

## Prompt 3 — Actual PyTorch + Flower federated training
"Add a separate Python ML module using PyTorch and Flower for 10 simulated clients. Load PneumoniaMNIST NPZ from a configured local path. Partition only training samples across clients; keep validation and test sets held out. Use a small CNN suitable for 28x28 grayscale images. Implement local training, FedAvg baseline and robust coordinate-wise median/trimmed mean aggregation as custom strategies or server-side aggregation hooks. Add controlled sign-flip and scaled-noise attacks to client model updates. Log per-round train/validation metrics and client update metadata. Add tests for tensor shapes, aggregation math, client partition isolation, and a deterministic attack seed. Keep the existing lightweight NumPy demo as fallback. Do not claim differential privacy or guaranteed Byzantine tolerance."

## Prompt 4 — Multi-metric detector and quarantine
"Implement a server-side update inspection module that calculates finite-value validation, L2 norm, pairwise cosine similarity, distance from coordinate median, and robust deviation score. Calibrate thresholds on clean rounds where possible. Add a trust ledger with bounded score changes, quarantine for current round, and operator-approved recovery after multiple clean rounds. Report precision, recall, false-positive rate and detection rate against known simulated attack labels. Do not automatically treat every outlier as malicious, especially under non-IID client distributions."

## Prompt 5 — Experiment comparison
"Add a reproducible benchmark runner comparing FedAvg, coordinate-wise median and trimmed mean under no attack, sign-flip and noise attacks. Use fixed seeds and multiple runs. Use the same held-out test set for all methods. Report mean and standard deviation for accuracy, precision, recall, F1, attack detection rate and false-positive rate. Export results to CSV and JSON. Never fabricate results."

## Prompt 6 — Gemini analyst
"Connect Gemini through the backend only. Add an incident explanation endpoint that receives detector-generated evidence JSON. Ask Gemini to explain evidence, uncertainty, false-positive possibility and suggested investigation. Gemini must not make the detection/quarantine decision. Never send raw images, patient records, API keys or full datasets to Gemini."

## Prompt 7 — Halo-inspired UI polish
"Polish the existing interface with an original sci-fi military command HUD aesthetic: dark graphite/navy, cyan/teal holographic accents, green healthy states, red threat states, angular panel geometry, tactical grid, restrained radar animation and condensed typography. Do not use copyrighted Halo logos, characters, screenshots or copied assets. Keep accessibility, responsive layout and all existing functionality."

## Prompt 8 — Deployment readiness
"Prepare this repo for Cloud Run backend and Vercel frontend deployment. Add Dockerfiles, health checks, environment variable validation, CORS allowlist, request size limits and production build verification. Ensure secrets are never committed. Document Firebase service-account setup via Secret Manager. Flag any endpoints that need authentication before public deployment."
