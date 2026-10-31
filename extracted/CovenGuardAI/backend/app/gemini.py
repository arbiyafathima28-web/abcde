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
        raise HTTPException(503, "GEMINI_API_KEY is not configured. Detection remains fully functional without Gemini.")
    try:
        from google import genai
        client = genai.Client(api_key=api_key)
        prompt = (
            "You are a cautious cybersecurity incident analyst for a federated-learning research demo. "
            "Explain only the evidence supplied. Do not claim certainty from a heuristic score. "
            "Return: summary, observed evidence, uncertainty/false-positive caveat, recommended next action. "
            "This is not a clinical system. Evidence JSON: " + str(req.evidence)
        )
        response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
        return {"explanation": response.text, "model": "gemini-2.5-flash"}
    except Exception as exc:
        raise HTTPException(502, f"Gemini request failed: {exc}")
