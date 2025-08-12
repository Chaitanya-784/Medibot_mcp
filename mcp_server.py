# mcp_server.py (Refactored for faster startup)
import os, io, json, base64, typing, random, pickle
from typing import Optional, Dict, Any

from starlette.applications import Starlette
from starlette.routing import Mount
from mcp.server.fastmcp import FastMCP

import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from keras.models import load_model
from PIL import Image
import fitz  # PyMuPDF

# Optional mime detection
try:
    import magic
    HAVE_MAGIC = True
except Exception:
    HAVE_MAGIC = False

# ---------- Env / Gemini ----------
import google.generativeai as genai
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "").strip()
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)
    GEMINI = genai.GenerativeModel("models/gemini-1.5-flash")
else:
    GEMINI = None

AUTH_TOKEN = os.getenv("AUTH_TOKEN", "").strip()
MY_NUMBER = os.getenv("MY_NUMBER", "").strip()
TOKENS_RAW = os.getenv("MCP_TOKENS", "").strip()
try:
    TOKEN_MAP: Dict[str, str] = json.loads(TOKENS_RAW) if TOKENS_RAW else {}
except Exception:
    TOKEN_MAP = {}

# ---------- NLTK & Keras Globals ----------
# REMOVE: The _ensure_nltk() function is no longer needed.
# The Dockerfile now handles this.

LEMMA = WordNetLemmatizer()
# We will load these at startup, not lazily.
_NLP = None
_WORDS = None
_CLASSES = None
_INTENTS = None

# ---------- FastMCP Instance ----------
mcp = FastMCP("MediBot MCP")

# ---------- ADD: Startup Event to Load Models ----------
@mcp.on_event("startup")
def load_ml_models():
    """Load all ML resources into memory when the app starts."""
    global _NLP, _WORDS, _CLASSES, _INTENTS
    print("[INFO] Loading Keras model and data files...")
    _NLP = load_model("model.h5")
    _WORDS = pickle.load(open("word.pkl", "rb"))
    _CLASSES = pickle.load(open("class.pkl", "rb"))
    with open("intents.json", "r", encoding="utf-8") as f:
        _INTENTS = json.load(f)
    print("[INFO] Model and data files loaded successfully.")

# ---------- Intent Prediction (Simplified) ----------
def _predict_intent(text: str, thresh: float = 0.9) -> Optional[str]:
    # REMOVE: The _lazy_load() call is no longer needed.
    # Models are guaranteed to be loaded at this point.
    if not text.strip(): return None
    
    tokens = [LEMMA.lemmatize(t.lower()) for t in nltk.word_tokenize(text)]
    bag = np.zeros(len(_WORDS), dtype=np.float32)
    for t in tokens:
        for i, w in enumerate(_WORDS):
            if w == t: bag[i] = 1.0
    probs = _NLP.predict(np.array([bag]), verbose=0)[0]
    idx = [(i, p) for i, p in enumerate(probs) if p > thresh]
    idx.sort(key=lambda x: x[1], reverse=True)
    if idx:
        tag = _CLASSES[idx[0][0]]
        for intent in _INTENTS["intents"]:
            if intent["tag"] == tag:
                return random.choice(intent["responses"])
    return None

#
# ... (the rest of your helper functions and tools like _mime_of, _gemini, get_health_advice, etc. remain unchanged) ...
#

# ---------- Helpers (No changes needed below this line) ----------
def _mime_of(data: bytes) -> str:
    if HAVE_MAGIC:
        try: return magic.from_buffer(data, mime=True)
        except Exception: pass
    if data[:5] == b"%PDF-": return "application/pdf"
    try:
        Image.open(io.BytesIO(data)); return "image/unknown"
    except Exception:
        return "application/octet-stream"

def _pdf_text(b: bytes) -> str:
    try:
        doc = fitz.open(stream=b, filetype="pdf")
        text = "".join(page.get_text() for page in doc)
        doc.close()
        return text or "No text could be extracted from the PDF."
    except Exception as e:
        return f"Error reading PDF: {e}"

def _gemini(parts: typing.List[typing.Any]) -> str:
    if GEMINI is None:
        return "Gemini key missing (set GOOGLE_API_KEY)."
    try:
        r = GEMINI.generate_content(parts)
        return (r.text or "").strip() or "No response."
    except Exception as e:
        return f"Gemini error: {e}"

def _coerce_bytes(file: Optional[Dict[str, Any]], file_b64: Optional[str]) -> Optional[bytes]:
    if file and isinstance(file, dict) and "data" in file:
        d = file["data"]
        if isinstance(d, (bytes, bytearray)): return bytes(d)
        if isinstance(d, str):
            try: return base64.b64decode(d, validate=True)
            except Exception: return d.encode()
    if file_b64 and isinstance(file_b64, str):
        try: return base64.b64decode(file_b64, validate=True)
        except Exception: pass
    return None

def _extract_questions(response: str) -> typing.List[str]:
    questions = []
    for line in response.split('\n'):
        line = line.strip(' -*\t')
        if not line: continue
        if (line.endswith('?') or line[:2].isdigit()) and len(line) < 200:
            questions.append(line)
        elif any(line.lower().startswith(prefix) for prefix in ('when', 'how', 'is', 'does', 'do', 'have', 'are', 'was', 'what', 'can', 'where', 'please', 'will')) and line.endswith('?'):
            questions.append(line)
    return questions

# ---------- Session Management ----------
_sessions: Dict[str, Dict[str, Any]] = {}

# ---------- Tools ----------
@mcp.tool()
def validate(bearer_token: str) -> dict:
    phone = TOKEN_MAP.get(bearer_token)
    if not phone and AUTH_TOKEN and bearer_token == AUTH_TOKEN:
        phone = MY_NUMBER
    if not phone or not phone.isdigit():
        raise ValueError("Authentication failed")
    return {"phone_number": phone}

@mcp.tool()
def get_health_advice(user_id: str,
                      message: str = "",
                      file: Optional[Dict[str, Any]] = None,
                      file_b64: Optional[str] = None) -> dict:
    message = (message or "").strip()
    session = _sessions.get(user_id)
    if session and session.get('stage') == 'questioning':
        session['answers'].append(message)
        session['current_q'] += 1
        if session['current_q'] < len(session['questions']):
            next_q = session['questions'][session['current_q']]
            _sessions[user_id] = session
            return {"advice": next_q}
        else:
            context = f"User initial message: {session['init_msg']}\n"
            for q, a in zip(session['questions'], session['answers']):
                context += f"Q: {q}\nA: {a}\n"
            context += "\nGiven all this, summarize: possible cause, remedies, and what to do next. Always include a disclaimer to consult a doctor."
            response = _gemini([context])
            del _sessions[user_id]
            return {"advice": response}

    canned = _predict_intent(message)
    if canned:
        return {"advice": canned}

    parts: typing.List[typing.Any] = []
    raw = _coerce_bytes(file, file_b64)

    if raw:
        mime = _mime_of(raw)
        if "image" in mime:
            try:
                img = Image.open(io.BytesIO(raw))
                prompt = ("You are a careful medical assistant. Analyze the user's query and this image. "
                          "Describe findings, likely causes (with uncertainty), next steps, and ALWAYS add a disclaimer.\n\n"
                          f"User: {message or '(no text)'}")
                parts = [prompt, img]
            except Exception as e:
                return {"advice": f"Could not read image: {e}"}
        elif "pdf" in mime:
            txt = _pdf_text(raw)
            prompt = ("You are a careful medical assistant. Use the user's question plus this PDF text. "
                      "Synthesize a concise, safe answer with actionable steps and a clear disclaimer.\n\n"
                      f"User: {message or '(no text)'}\n\n--- PDF (truncated) ---\n{txt[:2000]}...")
            parts = [prompt]
        else:
            parts = [f"Unknown file type; answer based on text only.\n\nUser: {message or '(no text)'}"]
        return {"advice": _gemini(parts)}

    if not message:
        return {"advice": "Please type a message or upload a file."}
    
    prompt = (
        f"User message: \"{message}\"\n"
        "Instruction: If this is a factual or informational question, answer directly. "
        "If this is a health complaint needing context, list 2-5 precise clarification questions as a bulleted list. "
        "Do not provide advice until you have all clarifications. Only list the questions if you are asking for clarification.\n"
        "Your response:"
    )
    gemini_response = _gemini([prompt])
    questions = _extract_questions(gemini_response)

    if questions:
        _sessions[user_id] = {
            'stage': 'questioning',
            'questions': questions,
            'answers': [],
            'current_q': 0,
            'init_msg': message
        }
        return {"advice": questions[0]}
    else:
        if user_id in _sessions: del _sessions[user_id]
        return {"advice": gemini_response}

# ---------- ASGI app at /mcp ----------
# The `on_event` decorator is tied to the FastMCP/Starlette app instance `mcp`.
# Now we create the final app using this configured instance.
app = Starlette(routes=[Mount("/mcp", app=mcp.sse_app())])

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    import uvicorn
    # This now runs the app which will trigger the 'startup' event.
    uvicorn.run(app, host="0.0.0.0", port=port)