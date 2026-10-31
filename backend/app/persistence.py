from __future__ import annotations
import os, json, time, sqlite3
from pathlib import Path
from typing import Any, Optional

_db = None
_mode = "sqlite"
_project_id: Optional[str] = None
_database_id: Optional[str] = None

ROOT = Path(__file__).resolve().parents[2]
SQLITE_DB = Path(__file__).resolve().parents[1] / "covenguard.sqlite3"

def _find_firebase_config() -> dict:
    """Locate firebase-applet-config.json if available."""
    candidates = [
        ROOT / "firebase-applet-config.json",
        Path(__file__).resolve().parents[1] / "firebase-applet-config.json",
        Path.cwd() / "firebase-applet-config.json",
    ]
    for p in candidates:
        if p.exists():
            try:
                with open(p, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
    return {}

def initialize() -> str:
    """Initialize Firebase Admin with Firestore and Auth; fallback to SQLite."""
    global _db, _mode, _project_id, _database_id

    cfg = _find_firebase_config()
    _project_id = os.getenv("FIREBASE_PROJECT_ID") or cfg.get("projectId")
    _database_id = os.getenv("FIRESTORE_DATABASE_ID") or cfg.get("firestoreDatabaseId")

    cred_json = os.getenv("FIREBASE_SERVICE_ACCOUNT_JSON")
    cred_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

    # If no credentials and no project ID, use SQLite
    if not cred_json and not cred_path and not _project_id:
        _mode = "sqlite"
        return _mode

    try:
        import firebase_admin
        from firebase_admin import credentials, firestore

        if not firebase_admin._apps:
            if cred_json:
                cred = credentials.Certificate(json.loads(cred_json))
                firebase_admin.initialize_app(cred, {"projectId": _project_id} if _project_id else None)
            elif cred_path and os.path.exists(cred_path):
                cred = credentials.Certificate(cred_path)
                firebase_admin.initialize_app(cred, {"projectId": _project_id} if _project_id else None)
            else:
                # Application Default Credentials on GCP / Cloud Run
                firebase_admin.initialize_app(options={"projectId": _project_id} if _project_id else None)

        if _database_id:
            try:
                _db = firestore.client(database_id=_database_id)
            except Exception:
                _db = firestore.client()
        else:
            _db = firestore.client()

        # Verify Firestore read/write permissions before declaring mode
        try:
            # Probe collections
            _ = list(_db.collection("hospitals").limit(1).stream())
            _mode = "firestore"
            print(f"[INFO] Firebase Firestore connected and verified. Mode: {_mode}, DB: {_database_id or '(default)'}")
        except Exception as probe_err:
            print(f"[INFO] Firestore probe not authorized ({probe_err}). Using SQLite local development fallback.")
            _db = None
            _mode = "sqlite"
    except Exception as exc:
        print(f"[WARN] Firebase Admin initialization failed: {exc}. Retaining SQLite fallback.")
        _db = None
        _mode = "sqlite"

    return _mode

def mode() -> str:
    return _mode

def is_firestore() -> bool:
    return _mode == "firestore" and _db is not None

def get_database_info() -> dict:
    return {
        "mode": _mode,
        "project_id": _project_id or "local",
        "database_id": _database_id or "default",
        "is_firestore": is_firestore()
    }

# --- Structured Collection Persistence ---

def save_audit_log(event: str, details: dict[str, Any], timestamp: Optional[str] = None, operator_email: str = "system") -> str:
    ts = timestamp or time.strftime("%H:%M:%S")
    payload = {
        "event": event,
        "details": details,
        "timestamp": ts,
        "operator": operator_email
    }

    if is_firestore():
        try:
            _db.collection("audit_logs").add(payload)
            _db.collection("audit_events").add(payload)
            return "firestore"
        except Exception as e:
            print(f"[WARN] Firestore audit write error: {e}")

    # SQLite fallback
    try:
        with sqlite3.connect(SQLITE_DB) as con:
            con.execute("CREATE TABLE IF NOT EXISTS audit (id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT DEFAULT CURRENT_TIMESTAMP, event TEXT NOT NULL, details TEXT NOT NULL)")
            con.execute("INSERT INTO audit(event, details) VALUES (?,?)", (event, json.dumps(details)))
    except Exception as e:
        print(f"[WARN] SQLite insert failed: {e}")

    return "sqlite"

def save_hospital_states(clients: list[dict], round_num: int) -> None:
    """Sync hospital node trust scores and quarantine status."""
    if is_firestore():
        try:
            batch = _db.batch()
            for c in clients:
                doc_ref = _db.collection("hospitals").document(f"hospital_{c['id']:02d}")
                batch.set(doc_ref, {
                    "id": c["id"],
                    "name": c["name"],
                    "status": c["status"],
                    "trust": c["trust"],
                    "samples": c["samples"],
                    "lastUpdated": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "round": round_num
                }, merge=True)
            batch.commit()
        except Exception as e:
            print(f"[WARN] Firestore hospitals batch update error: {e}")

def save_experiment_record(exp: dict) -> None:
    """Save full experiment metrics and parameters to Firestore."""
    if is_firestore():
        try:
            doc_data = {
                "round": exp["round"],
                "attack": exp["attack"],
                "aggregation": exp["aggregation"],
                "maliciousClient": exp.get("malicious_client"),
                "flaggedClients": exp.get("flagged_clients", []),
                "detectedAttacker": exp.get("detected_attacker"),
                "acceptedClients": exp.get("accepted_clients", []),
                "test": exp.get("test", {}),
                "validation": exp.get("validation", {}),
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "note": exp.get("note", "")
            }
            _db.collection("experiments").document(f"round_{exp['round']:03d}").set(doc_data)
            _db.collection("federated_rounds").document(f"round_{exp['round']:03d}").set({
                "roundNumber": exp["round"],
                "participatingNodes": 10,
                "acceptedNodes": len(exp.get("accepted_clients", [])),
                "quarantinedNodes": exp.get("flagged_clients", []),
                "attackVector": exp["attack"],
                "aggregation": exp["aggregation"],
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ")
            })
        except Exception as e:
            print(f"[WARN] Firestore experiment write error: {e}")

def save_security_incident(incident: dict) -> None:
    """Save threat alerts and Byzantine anomaly quarantine records."""
    if is_firestore():
        try:
            _db.collection("security_incidents").add({
                "hospitalId": incident.get("hospital"),
                "reason": incident.get("reason", "Anomalous parameter update detected"),
                "l2Norm": incident.get("l2_norm"),
                "meanCosine": incident.get("mean_cosine"),
                "robustScore": incident.get("robust_score"),
                "round": incident.get("round"),
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ")
            })
        except Exception as e:
            print(f"[WARN] Firestore incident write error: {e}")

def save_trust_history(client_id: int, old_trust: float, new_trust: float, round_num: int, reason: str) -> None:
    """Save dynamic trust ledger adjustments."""
    if is_firestore():
        try:
            _db.collection("trust_history").add({
                "hospitalId": client_id,
                "oldTrust": old_trust,
                "newTrust": new_trust,
                "round": round_num,
                "reason": reason,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ")
            })
        except Exception as e:
            print(f"[WARN] Firestore trust history write error: {e}")

def save_event(event: str, details: dict[str, Any], timestamp: str) -> str:
    """Compatibility wrapper for main.py."""
    return save_audit_log(event, details, timestamp)
