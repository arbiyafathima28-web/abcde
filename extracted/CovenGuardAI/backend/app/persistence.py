import os, json
from typing import Any

_db = None
_mode = "sqlite"

def initialize():
    """Initialize Firebase Admin when configured; otherwise retain SQLite fallback."""
    global _db, _mode
    cred_json = os.getenv("FIREBASE_SERVICE_ACCOUNT_JSON")
    project_id = os.getenv("FIREBASE_PROJECT_ID")
    if not cred_json or not project_id:
        _mode = "sqlite"
        return _mode
    try:
        import firebase_admin
        from firebase_admin import credentials, firestore
        if not firebase_admin._apps:
            cred = credentials.Certificate(json.loads(cred_json))
            firebase_admin.initialize_app(cred, {"projectId": project_id})
        _db = firestore.client()
        _mode = "firestore"
    except Exception as exc:
        print(f"[WARN] Firebase unavailable; using SQLite audit fallback: {exc}")
        _mode = "sqlite"
    return _mode

def mode():
    return _mode

def save_event(event: str, details: dict[str, Any], timestamp: str):
    if _mode == "firestore" and _db is not None:
        _db.collection("audit_events").add({
            "event": event, "details": details, "timestamp": timestamp
        })
        return "firestore"
    return "sqlite"
