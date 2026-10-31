import os
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()

class ExplainRequest(BaseModel):
    evidence: dict

@router.post("/api/ai/explain")
def explain(req: ExplainRequest):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(503, "GEMINI_API_KEY is not configured in backend environment. Anomaly detection remains fully functional.")
    
    try:
        from google import genai
        client = genai.Client(api_key=api_key)
        prompt = (
            "You are a cybersecurity incident analyst for CovenGuard AI, a Byzantine-robust federated learning research prototype. "
            "Analyze the following detector-produced anomaly evidence JSON. Do NOT make medical or clinical claims. "
            "Do NOT make absolute claims of malicious intent; frame your analysis around statistical and geometric update deviations, "
            "explicitly addressing potential false positives (e.g. non-IID patient data skew) versus adversarial poisoning (sign-flipping or noise). "
            "Format your response with the following sections:\n"
            "1. INCIDENT OVERVIEW (Node ID, quarantine status, overall assessment)\n"
            "2. STATISTICAL & GEOMETRIC EVIDENCE (L2 update norm deviation, mean cosine alignment, robust z-score)\n"
            "3. FALSE POSITIVE & NON-IID CAVEATS (Why this might or might not be a legitimate clinical outlier)\n"
            "4. OPERATOR RECOMMENDATIONS (Quarantine review, weight inspection, client revalidation steps)\n\n"
            f"Evidence JSON: {req.evidence}"
        )
        # Attempt gemini-2.5-flash, fallback to gemini-2.0-flash
        model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        try:
            response = client.models.generate_content(model=model_name, contents=prompt)
            return {"explanation": response.text, "model": model_name}
        except Exception as inner_err:
            fallback = "gemini-2.0-flash"
            response = client.models.generate_content(model=fallback, contents=prompt)
            return {"explanation": response.text, "model": fallback, "note": f"Fallback used from {model_name}"}
    except Exception as exc:
        raise HTTPException(502, f"Gemini security explanation request failed: {exc}")
