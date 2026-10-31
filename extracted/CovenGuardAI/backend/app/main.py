from __future__ import annotations
import json, sqlite3, time
from pathlib import Path
from typing import Literal
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from .dataset import DatasetStore
from .ml import train_local, metrics, fedavg, coordinate_median, trimmed_mean, inspect_updates
from . import persistence
from .gemini import router as gemini_router

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "covenguard.sqlite3"
app = FastAPI(title="CovenGuard AI API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
app.include_router(gemini_router)
store = DatasetStore()
state = {"dataset": None, "clients": [], "events": [], "last_experiment": None, "round": 0}

def db_init():
    with sqlite3.connect(DB) as con:
        con.execute("CREATE TABLE IF NOT EXISTS audit (id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT DEFAULT CURRENT_TIMESTAMP, event TEXT NOT NULL, details TEXT NOT NULL)")
def log(event, details):
    row = {"time": time.strftime("%H:%M:%S"), "event": event, "details": details}
    state["events"].insert(0, row)
    state["events"] = state["events"][:100]
    with sqlite3.connect(DB) as con:
        con.execute("INSERT INTO audit(event, details) VALUES (?,?)", (event, json.dumps(details)))
    try:
        persistence.save_event(event, details, row["time"])
    except Exception as exc:
        print(f"[WARN] Firestore audit write failed: {exc}")
db_init()

def client_state():
    return [{"id": i+1, "name": f"Hospital {i+1:02d}", "status": "HEALTHY", "trust": 100, "samples": 0} for i in range(10)]
state["clients"] = client_state()
store.ensure_demo()
state["dataset"] = store.summary()

class RunRequest(BaseModel):
    attack: Literal["none", "sign_flip", "noise"] = "sign_flip"
    aggregation: Literal["fedavg", "median", "trimmed_mean"] = "median"
    malicious_client: int = Field(default=7, ge=1, le=10)
    rounds: int = Field(default=1, ge=1, le=5)

@app.get("/api/health")
def health():
    return {"ok": True, "service": "CovenGuard AI", "dataset": store.summary()}

@app.get("/api/state")
def get_state():
    return {"dataset": store.summary(), "clients": state["clients"], "events": state["events"],
            "round": state["round"], "last_experiment": state["last_experiment"], "persistence": persistence.mode()}

@app.post("/api/dataset/upload")
async def upload_dataset(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".npz"):
        raise HTTPException(400, "Please upload a .npz file.")
    raw = await file.read()
    if len(raw) > 150 * 1024 * 1024:
        raise HTTPException(413, "Dataset is larger than the 150 MB MVP limit.")
    try:
        summary = store.load_npz(raw)
        state["dataset"] = summary
        state["clients"] = client_state()
        log("DATASET_LOADED", {"name": summary["name"], "train_samples": summary["train_samples"]})
        return summary
    except Exception as e:
        raise HTTPException(400, str(e))

@app.post("/api/experiment/run")
def run_experiment(req: RunRequest):
    x, y = store.train_x, store.train_y
    rng = np.random.default_rng(2026 + state["round"])
    n_clients = 10
    indices = rng.permutation(len(y))
    partitions = np.array_split(indices, n_clients)
    # A small common feature vector model; each client retains its own partition.
    w_global = np.zeros(x.shape[1] + 1, dtype=np.float32)
    honest_updates, client_updates, sample_counts = [], [], []
    for idx, ids in enumerate(partitions, start=1):
        local_w = train_local(x[ids], y[ids], w_global, epochs=3, lr=0.12)
        delta = local_w - w_global
        if req.attack != "none" and idx == req.malicious_client:
            if req.attack == "sign_flip":
                delta = -4.0 * delta
            elif req.attack == "noise":
                delta = delta + rng.normal(0, 1.5, size=delta.shape).astype(np.float32)
        client_updates.append(delta)
        sample_counts.append(len(ids))
    norms, similarities, deviations, robust_z, flags, cosine = inspect_updates(client_updates)
    # Keep requested attacker's identity as a known simulation ground truth, but detection is still metric-based.
    flagged = [i+1 for i, f in enumerate(flags) if bool(f)]
    # For a controlled sign-flip demo, include attacker if the metrics show a deviation; expose evidence.
    if req.attack != "none" and req.malicious_client not in flagged:
        # Do not silently label as detected: leave it unflagged and make the limitation visible.
        pass
    accepted = [u for i, u in enumerate(client_updates) if (i+1) not in flagged]
    accepted_counts = [sample_counts[i] for i in range(10) if (i+1) not in flagged]
    if not accepted:
        accepted, accepted_counts = client_updates, sample_counts
    if req.aggregation == "fedavg":
        agg = fedavg(accepted, accepted_counts)
    elif req.aggregation == "median":
        agg = coordinate_median(accepted)
    else:
        agg = trimmed_mean(accepted)
    # Model is one communication round; apply aggregate update.
    w_new = w_global + agg
    val = metrics(w_new, store.val_x, store.val_y)
    test = metrics(w_new, store.test_x, store.test_y)
    clients = []
    for i in range(10):
        is_flagged = (i+1) in flagged
        previous = state["clients"][i]["trust"]
        trust = max(0, previous - (18 if is_flagged else 0.5))
        clients.append({"id": i+1, "name": f"Hospital {i+1:02d}",
            "status": "QUARANTINED" if is_flagged else "HEALTHY",
            "trust": round(trust, 1), "samples": int(sample_counts[i]),
            "l2_norm": round(float(norms[i]), 5),
            "mean_cosine": round(float(similarities[i]), 5),
            "deviation": round(float(deviations[i]), 5),
            "robust_score": round(float(robust_z[i]), 3),
            "is_flagged": bool(is_flagged)})
    state["clients"] = clients
    state["round"] += 1
    result = {
        "round": state["round"], "attack": req.attack, "malicious_client": req.malicious_client if req.attack != "none" else None,
        "aggregation": req.aggregation, "flagged_clients": flagged,
        "detected_attacker": req.malicious_client in flagged if req.attack != "none" else None,
        "accepted_clients": [i+1 for i in range(10) if (i+1) not in flagged],
        "validation": val, "test": test,
        "client_evidence": clients,
        "note": "Metrics are calculated from this run. Detection is heuristic; review false positives and false negatives."
    }
    state["last_experiment"] = result
    log("ROUND_COMPLETED", {"round": state["round"], "attack": req.attack, "aggregation": req.aggregation, "flagged": flagged, "test": test})
    for cid in flagged:
        log("NODE_QUARANTINED", {"hospital": cid, "reason": "Statistical/geometric anomaly threshold"})
    return result

@app.post("/api/reset")
def reset():
    state["clients"] = client_state()
    state["events"] = []
    state["last_experiment"] = None
    state["round"] = 0
    log("NETWORK_RESET", {"status": "reset"})
    return {"ok": True}
