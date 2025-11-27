import os
from dotenv import load_dotenv
import requests
import json
from typing import Dict, Any, Optional

BASE_DIR = os.path.dirname(__file__)
load_dotenv(os.path.join(BASE_DIR, ".env"))

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
MODEL_NAME = os.getenv("DEEPSEEK_MODEL", "DeepSeek-R1-0528")  # default to a robust R1 release

HEADERS = {
    "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
    "Content-Type": "application/json"
}

SYSTEM_PROMPT = """
You are a clinical reasoning assistant. Input: free-text medical record (admission notes, discharge summary, labs, vitals).
Output: JSON ONLY (strict), with the following keys:

{
  "icd10_code": "<preferred ICD-10 code (e.g. I50.9) or null>",
  "diagnosis_name": "<short diagnosis name>",
  "confidence": "<0-1 float>",
  "inpatient": true|false,
  "estimated_stay_days": <integer or null>,
  "ward_type": "<ICU|Cardiac|Neurological|General|Isolation|Outpatient>",
  "recommended_medicines": ["list of medicines (dose if obvious)"],
  "extracted": {
     "age": <int|null>,
     "sex": "M|F|Other|null",
     "blood_pressure_sys": <int|null>,
     "blood_pressure_dia": <int|null>,
     "heart_rate": <int|null>,
     "temperature_c": <float|null>,
     "wbc": <float|null>,
     "crp": <float|null>,
     "symptoms": ["list of extracted symptoms"],
     "comorbidities": ["list of comorbidities"]
  },
  "rationale": "<short plain-text explanation (max 120 words)>"
}

- If a field is unknown, set it to null (not empty string).
- Try to return the most precise ICD-10 code available.
- Keep 'recommended_medicines' conservative and list common medicine names; do not invent novel drugs.
- Output valid JSON only; do NOT prepend commentary.
"""

def call_deepseek_chat(prompt_user: str, system_prompt: str = SYSTEM_PROMPT, model: str = MODEL_NAME, max_tokens: int = 1000, temperature: float = 0.0) -> Optional[Dict[str, Any]]:
    """
    Calls DeepSeek chat completions endpoint (OpenAI-compatible interface).
    Returns parsed JSON dict or None on failure.
    """
    if not DEEPSEEK_API_KEY:
        raise RuntimeError("DEEPSEEK_API_KEY not set in environment (.env)")

    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt_user}
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": 1.0
    }

    url = f"{DEEPSEEK_BASE_URL.rstrip('/')}/chat/completions"
    resp = requests.post(url, headers=HEADERS, json=body, timeout=120)
    if resp.status_code != 200:
        raise RuntimeError(f"DeepSeek API error {resp.status_code}: {resp.text}")

    data = resp.json()
    # The OpenAI-style response usually includes choices[0].message.content
    content = None
    try:
        content = data["choices"][0]["message"]["content"]
    except Exception:
        # some DeepSeek wrappers might put text in 'choices'[0]['text']
        content = data["choices"][0].get("text") if "choices" in data and len(data["choices"])>0 else None

    if not content:
        raise RuntimeError("No content in DeepSeek response")

    # Ensure content is JSON
    try:
        parsed = json.loads(content)
        return parsed
    except Exception:
        # Attempt to extract JSON substring
        import re
        m = re.search(r'(\{.*\})', content, re.S)
        if m:
            try:
                parsed = json.loads(m.group(1))
                return parsed
            except Exception as e:
                raise RuntimeError(f"Failed to parse JSON from model output: {e}\nRaw:\n{content}")
        else:
            raise RuntimeError(f"Model output is not valid JSON:\n{content}")

def analyze_text_with_deepseek(text: str) -> Dict[str, Any]:
    """
    High-level wrapper that calls the model and ensures all expected keys exist.
    """
    res = call_deepseek_chat(text)
    if res is None:
        # Be explicit so static analyzers know `res` is not None beyond this point
        raise RuntimeError("DeepSeek returned no data")

    if not isinstance(res, dict):
        # defensive: if parser returned unexpected type, normalize to empty dict
        res = {}

    # normalize output - ensure keys exist with safe defaults
    keys = ["icd10_code","diagnosis_name","confidence","inpatient","estimated_stay_days","ward_type","recommended_medicines","extracted","rationale"]
    out: Dict[str, Any] = {}
    for k in keys:
        out[k] = res.get(k, None)

    # extracted subkeys - ensure extracted is a dict before populating
    extracted_defaults = {
        "age": None,"sex": None,"blood_pressure_sys": None,"blood_pressure_dia": None,
        "heart_rate": None,"temperature_c": None,"wbc": None,"crp": None,
        "symptoms": [], "comorbidities": []
    }
    extracted = res.get("extracted") or {}
    if not isinstance(extracted, dict):
        extracted = {}

    # ensure out["extracted"] is a dict we can safely populate
    out["extracted"] = {}
    for k, v in extracted_defaults.items():
        out["extracted"][k] = extracted.get(k, v)

    return out
