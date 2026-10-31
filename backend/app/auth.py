from __future__ import annotations
import os
from typing import Optional
from fastapi import Header, HTTPException, Depends

ADMIN_EMAIL = "arbiyafathima975@gmail.com"

def get_current_user(authorization: Optional[str] = Header(None)) -> dict:
    """Validate incoming Firebase ID token or handle local/dev mode gracefully."""
    auth_required = os.getenv("AUTH_REQUIRED", "false").lower() in ("true", "1", "yes")

    if not authorization or not authorization.startswith("Bearer "):
        if auth_required:
            raise HTTPException(
                status_code=401,
                detail="Authentication required. Please provide a valid Firebase Bearer token."
            )
        # Development fallback operator
        return {
            "uid": "local-dev-operator",
            "email": "operator@covenguard.local",
            "role": "admin",
            "is_authenticated": False
        }

    id_token = authorization.split("Bearer ")[1].strip()

    try:
        import firebase_admin
        from firebase_admin import auth as fb_auth

        if not firebase_admin._apps:
            # If Firebase is not initialized, allow fallback if auth not strictly enforced
            if auth_required:
                raise HTTPException(500, "Firebase Authentication service is not initialized on server.")
            return {
                "uid": "anonymous-operator",
                "email": "operator@covenguard.local",
                "role": "admin",
                "is_authenticated": False
            }

        decoded_token = fb_auth.verify_id_token(id_token)
        email = decoded_token.get("email", "")
        uid = decoded_token.get("uid", "")

        is_admin = (email.lower() == ADMIN_EMAIL.lower()) or bool(decoded_token.get("admin", False))
        role = "admin" if is_admin else "viewer"

        return {
            "uid": uid,
            "email": email,
            "name": decoded_token.get("name", ""),
            "role": role,
            "is_authenticated": True
        }
    except Exception as exc:
        print(f"[AUTH ERROR] Failed to verify Firebase token: {exc}")
        if auth_required:
            raise HTTPException(status_code=401, detail=f"Invalid or expired Firebase ID token: {exc}")
        return {
            "uid": "unverified-operator",
            "email": "unknown@covenguard.local",
            "role": "viewer",
            "is_authenticated": False
        }

def require_admin(user: dict = Depends(get_current_user)) -> dict:
    """Enforce administrator permissions for sensitive attack or network mutation commands."""
    if user.get("role") != "admin":
        raise HTTPException(
            status_code=403,
            detail="Forbidden: Operator does not possess Administrator privileges for Byzantine attack injection."
        )
    return user
