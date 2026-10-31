from __future__ import annotations
import json, os, time
from pathlib import Path
from typing import Literal, Optional
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from .dataset import DatasetStore
from .ml import (
    PneumoniaCNN, get_params_vector, set_params_vector,
    partition_dataset, train_local, metrics,
    fedavg, coordinate_median, trimmed_mean, inspect_updates
)
from . import persistence
from .auth import get_current_user, require_admin
from .gemini import router as gemini_router
import copy

app = FastAPI(title="CovenGuard AI API", version="2.0.0")

# CORS setup: allow all origins in development or accept comma-separated CORS_ORIGINS
cors_origins_env = os.getenv("CORS_ORIGINS", "")
if cors_origins_env.strip():
    allow_origins = [o.strip() for o in cors_origins_env.split(",") if o.strip()]
    allow_creds = True
else:
    allow_origins = ["*"]
    allow_creds = False

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=allow_creds,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(gemini_router)
store = DatasetStore()
global_model = PneumoniaCNN()
state = {
    "dataset": None,
    "clients": [],
    "events": [],
    "last_experiment": None,
    "round": 0,
    "loss_history": []
}

def client_state():
    return [{"id": i+1, "name": f"Hospital {i+1:02d}", "status": "HEALTHY", "trust": 100, "samples": 0} for i in range(10)]

def log(event: str, details: dict, operator_email: str = "system"):
    ts = time.strftime("%H:%M:%S")
    row = {"time": ts, "event": event, "details": details, "operator": operator_email}
    state["events"].insert(0, row)
    state["events"] = state["events"][:100]
    persistence.save_audit_log(event, details, ts, operator_email)

# Initialize on startup
persistence.initialize()
state["clients"] = client_state()
store.ensure_demo()
state["dataset"] = store.summary()
log("SYSTEM_BOOT", {"persistence": persistence.mode(), "dataset": state["dataset"]["name"], "model": "PyTorch PneumoniaCNN"})

class RunRequest(BaseModel):
    attack: Literal["none", "sign_flip", "noise"] = "sign_flip"
    aggregation: Literal["fedavg", "median", "trimmed_mean"] = "median"
    malicious_client: int = Field(default=7, ge=1, le=10)
    rounds: int = Field(default=1, ge=1, le=5)
    data_partition: Literal["iid", "non_iid"] = "iid"
    local_epochs: int = Field(default=2, ge=1, le=5)

@app.get("/health")
@app.get("/api/health")
def health():
    return {
        "ok": True,
        "service": "CovenGuard AI",
        "database": persistence.get_database_info(),
        "dataset": store.summary(),
        "round": state["round"],
        "clients_count": len(state["clients"])
    }

@app.get("/state")
@app.get("/api/state")
def get_state(user: dict = Depends(get_current_user)):
    return {
        "dataset": store.summary(),
        "clients": state["clients"],
        "events": state["events"],
        "round": state["round"],
        "last_experiment": state["last_experiment"],
        "persistence": persistence.mode(),
        "database": persistence.get_database_info(),
        "operator": {
            "email": user.get("email"),
            "role": user.get("role"),
            "isAuthenticated": user.get("is_authenticated", False)
        }
    }

@app.post("/api/dataset/upload")
async def upload_dataset(file: UploadFile = File(...), user: dict = Depends(get_current_user)):
    if not file.filename.lower().endswith(".npz"):
        raise HTTPException(400, "Please upload a valid MedMNIST .npz archive.")
    raw = await file.read()
    if len(raw) > 150 * 1024 * 1024:
        raise HTTPException(413, "Dataset exceeds the 150 MB MVP file size threshold.")
    try:
        summary = store.load_npz(raw)
        state["dataset"] = summary
        state["clients"] = client_state()
        log("DATASET_LOADED", {
            "name": summary["name"],
            "train_samples": summary["train_samples"],
            "filename": file.filename
        }, operator_email=user.get("email", "operator"))
        return summary
    except Exception as e:
        raise HTTPException(400, str(e))

@app.post("/api/experiment/run")
def run_experiment(req: RunRequest, user: dict = Depends(get_current_user)):
    global global_model
    # If injecting malicious attack vectors, require administrator role
    if req.attack != "none" and user.get("role") != "admin":
        raise HTTPException(
            status_code=403,
            detail="Permission Denied: Byzantine attack simulations require Administrator authorization."
        )

    # 1. Evaluate baseline metrics before round
    baseline_test = metrics(global_model, store.test_x, store.test_y)
    baseline_val = metrics(global_model, store.val_x, store.val_y)

    # 2. Partition dataset for 10 simulated hospitals (IID or Non-IID Dirichlet)
    rng = np.random.default_rng(2026 + state["round"])
    partitions = partition_dataset(store.train_y, n_clients=10, partition_type=req.data_partition, seed=2026 + state["round"])

    # 3. Local training for each hospital client
    client_updates, sample_counts, client_losses = [], [], []
    for idx, ids in enumerate(partitions, start=1):
        delta, losses = train_local(
            global_model,
            store.train_x[ids],
            store.train_y[ids],
            epochs=req.local_epochs,
            lr=0.002
        )
        # Apply simulated Byzantine attacks
        if req.attack != "none" and idx == req.malicious_client:
            if req.attack == "sign_flip":
                delta = -4.0 * delta
            elif req.attack == "noise":
                noise_scale = float(2.0 * (np.std(delta) + 1e-4))
                delta = delta + rng.normal(0, noise_scale, size=delta.shape).astype(np.float32)

        client_updates.append(delta)
        sample_counts.append(len(ids))
        client_losses.append(losses)

    # 4. Multi-metric Byzantine anomaly detection
    norms, similarities, deviations, robust_z, flags, cosine = inspect_updates(client_updates)
    flagged = [i+1 for i, f in enumerate(flags) if bool(f)]

    # 5. Measure Naive Poisoned FedAvg (unfiltered baseline comparison)
    delta_poisoned = fedavg(client_updates, sample_counts)
    temp_poisoned = copy.deepcopy(global_model)
    set_params_vector(temp_poisoned, get_params_vector(global_model) + delta_poisoned)
    poisoned_metrics = metrics(temp_poisoned, store.test_x, store.test_y)

    # 6. Robust Aggregation (quarantine flagged Byzantine nodes)
    accepted_indices = [i for i in range(10) if (i+1) not in flagged]
    if not accepted_indices:
        accepted_updates, accepted_counts = client_updates, sample_counts
    else:
        accepted_updates = [client_updates[i] for i in accepted_indices]
        accepted_counts = [sample_counts[i] for i in accepted_indices]

    if req.aggregation == "fedavg":
        agg = fedavg(accepted_updates, accepted_counts)
    elif req.aggregation == "median":
        agg = coordinate_median(accepted_updates if accepted_updates else client_updates)
    else:
        agg = trimmed_mean(accepted_updates if accepted_updates else client_updates)

    # 7. Apply robust update to global CNN
    w_new = get_params_vector(global_model) + agg
    set_params_vector(global_model, w_new)

    # 8. Post-round evaluations on independent splits
    val = metrics(global_model, store.val_x, store.val_y)
    test = metrics(global_model, store.test_x, store.test_y)

    # 9. Update hospital states and trust scores
    clients = []
    current_round = state["round"] + 1
    for i in range(10):
        cid = i + 1
        is_flagged = cid in flagged
        previous_trust = state["clients"][i]["trust"]
        new_trust = max(0.0, previous_trust - (20.0 if is_flagged else 0.5))

        if is_flagged:
            persistence.save_trust_history(
                cid, previous_trust, new_trust, current_round,
                "Quarantine penalty: Anomaly detection trigger"
            )

        clients.append({
            "id": cid,
            "name": f"Hospital {cid:02d}",
            "status": "QUARANTINED" if is_flagged else "HEALTHY",
            "trust": round(new_trust, 1),
            "samples": int(sample_counts[i]),
            "l2_norm": round(float(norms[i]), 5),
            "mean_cosine": round(float(similarities[i]), 5),
            "deviation": round(float(deviations[i]), 5),
            "robust_score": round(float(robust_z[i]), 3),
            "is_flagged": bool(is_flagged)
        })

    state["clients"] = clients
    state["round"] = current_round

    avg_train_loss = round(float(np.mean([l[-1] for l in client_losses if l])), 4)
    loss_point = {
        "round": current_round,
        "train_loss": avg_train_loss,
        "val_loss": val["loss"],
        "test_loss": test["loss"]
    }
    state["loss_history"].append(loss_point)

    result = {
        "round": current_round,
        "partition": req.data_partition,
        "attack": req.attack,
        "malicious_client": req.malicious_client if req.attack != "none" else None,
        "aggregation": req.aggregation,
        "flagged_clients": flagged,
        "detected_attacker": req.malicious_client in flagged if req.attack != "none" else None,
        "accepted_clients": [i+1 for i in range(10) if (i+1) not in flagged],
        "baseline": baseline_test,
        "poisoned_fedavg": poisoned_metrics,
        "robust_aggregation": test,
        "validation": val,
        "test": test,
        "loss_history": state["loss_history"],
        "client_evidence": clients,
        "note": "Evaluated on independent held-out split using PyTorch CNN with Byzantine robust aggregation."
    }
    state["last_experiment"] = result

    # Persist experiment and updated hospital states to Firestore
    persistence.save_experiment_record(result)
    persistence.save_hospital_states(clients, current_round)

    log("ROUND_COMPLETED", {
        "round": current_round,
        "partition": req.data_partition,
        "attack": req.attack,
        "aggregation": req.aggregation,
        "quarantined": flagged,
        "accuracy": test["accuracy"],
        "f1": test["f1"],
        "loss": test["loss"]
    }, operator_email=user.get("email", "operator"))

    for cid in flagged:
        incident_data = {
            "hospital": cid,
            "reason": "Byzantine update anomaly: L2 norm / directional deviation threshold exceeded",
            "l2_norm": next((c["l2_norm"] for c in clients if c["id"] == cid), None),
            "mean_cosine": next((c["mean_cosine"] for c in clients if c["id"] == cid), None),
            "robust_score": next((c["robust_score"] for c in clients if c["id"] == cid), None),
            "round": current_round
        }
        persistence.save_security_incident(incident_data)
        log("NODE_QUARANTINED", incident_data, operator_email=user.get("email", "operator"))

    return result

@app.post("/api/reset")
def reset(user: dict = Depends(require_admin)):
    global global_model
    global_model = PneumoniaCNN()
    state["clients"] = client_state()
    state["events"] = []
    state["last_experiment"] = None
    state["round"] = 0
    state["loss_history"] = []
    persistence.save_hospital_states(state["clients"], 0)
    log("NETWORK_RESET", {"status": "Full reset to initial baseline state (PyTorch CNN reinitialized)"}, operator_email=user.get("email", "admin"))
    return {"ok": True, "message": "Hospital network reset successfully."}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port)

