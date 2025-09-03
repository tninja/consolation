import os, json
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
from tempfile import NamedTemporaryFile
from io import BytesIO
from typing import Tuple

def _init_openai_client() -> Optional[OpenAI]:
    """Initialize OpenAI client if API key is available; otherwise return None.
    Avoids raising during module import so unit tests can run without network/keys.
    """
    try:
        return OpenAI()
    except Exception:
        return None

# Lazily-initialized client; may be None in test environments without API key
client: Optional[OpenAI] = _init_openai_client()

app = FastAPI(title="Philosophy Comfort API (OpenAI SDK)")

# Change to your GitHub Pages domain (user page and/or project page)
ALLOWED_ORIGINS = [
    "https://tninja.github.io",
    "https://tninja.github.io/fastapi-helloworld"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ComfortQuery(BaseModel):
    language: str = "zh"
    situation: str
    philosophy_background: Optional[str] = "philosophy"
    max_passages: int = 3
    guidance: Optional[str] = ""

class Passage(BaseModel):
    ref: str
    short_quote: str
    reason: str
    full_passage_text: str

class ComfortResponse(BaseModel):
    passages: List[Passage]
    reflection: str
    exercise: str
    disclaimer: str

class TTSRequest(BaseModel):
    text: str
    language: Optional[str] = "zh"
    voice: Optional[str] = None  # if None, pick by language
    format: Optional[str] = "mp3"  # mp3 or wav

SYSTEM_PROMPT = """You are a calm, pluralistic philosophical counselor who draws from a wide range of philosophers (e.g., Aristotle, Epicurus, Stoics like Marcus Aurelius/Epictetus/Seneca, Confucius, Montaigne, Descartes, Spinoza, Hume, Kant, Schopenhauer, Nietzsche, Kierkegaard, Camus, Sartre) and from 'The Consolations of Philosophy' by Alain de Botton.
You MUST respond STRICTLY in the user's requested language (zh for Chinese, en for English) and DO NOT mix languages.
Select the most relevant, high-leverage ideas to comfort and guide the user; combine multiple perspectives when helpful.
You are encouraged to incorporate insights from 'The Consolations of Philosophy'—summarize its ideas clearly and practically.
Explicitly draw on Philosophy of Well-Being and practical wisdom aimed at living a happier life; name relevant concepts (eudaimonia, ataraxia, flourishing, virtue ethics, meaning, etc.).
When appropriate, cite or summarize ideas from Philosophy of Well-Being and "wisdom to live happier" traditions to enhance clarity and usefulness.
For copyrighted works (including modern books): prefer concise paraphrases rather than long verbatim quotes. For public-domain works, you may include short snippets but keep them brief (<= 20 words/chars).
Write a practical, compassionate philosophical reflection (300-500 zh characters / 300-400 English words) and provide a short step-by-step philosophical exercise (4-8 sentences), such as reframing, dichotomy of control, view-from-above, journaling prompts, or virtue rehearsal.
Avoid sectarian or religious framing; focus on agency, clarity, and emotional steadiness.
Return STRICT JSON only, matching exactly the schema the user supplies.
If unsure about exact sections, choose ones you are confident in and clearly name the work and section (e.g., "Meditations 2.1", "Nicomachean Ethics II").
"""

USER_PROMPT_TMPL = """User language: {language}
Philosophical background: {background}
Situation detail: {situation}
Additional guidance: {guidance}

Return JSON with fields:
- passages: array of at most {max_passages} objects with fields:
  - ref (string, e.g., "Meditations 2.1", "Enchiridion 1", "Aristotle, Nicomachean Ethics I–II (eudaimonia/virtue)", "Epicurus, Letter to Menoeceus (ataraxia)", "Montaigne, Essays I.20", "de Botton, The Consolations of Philosophy — Chapter X (summary)")
  - short_quote (string, <= 20 words/chars; a paraphrase or a short snippet; MAY be empty)
  - reason (string, 1-2 sentences why this fits)
  - full_passage_text (string, if public-domain in the requested language you may include brief original text; otherwise provide a faithful paraphrase and clearly name the source work)
- reflection: a 300-500 {lang_unit} philosophical reflection applying these ideas to the user's situation, emphasizing practical wisdom and well-being (living happier with clarity and agency).
- exercise: 4-8 sentences describing a concrete philosophical practice or protocol (e.g., reframing steps, dichotomy of control, journaling prompts, or view-from-above) that supports well-being and is suitable for immediate use.
- disclaimer: one sentence inviting the user to verify the passage in their preferred edition/translation and noting that copyrighted texts are summarized.

Use the requested language for everything.
"""

def build_messages(q: ComfortQuery) -> List[Dict[str, str]]:
    lang_unit = "characters" if q.language.startswith("zh") else "words"
    uprompt = USER_PROMPT_TMPL.format(
        language=q.language,
        background=(getattr(q, "philosophy_background", None) or "philosophy"),
        situation=q.situation,
        guidance=q.guidance or "None",
        max_passages=max(1, min(q.max_passages, 10)),
        lang_unit=lang_unit,
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": uprompt},
    ]

def get_comfort_from_openai(q: ComfortQuery, *, openai_client: Optional[OpenAI] = None) -> Dict[str, Any]:
    """
    Builds the prompt, calls the OpenAI API, and processes the response.
    This function is designed to be testable independently of the FastAPI framework.
    """
    messages = build_messages(q)

    try:
        oc = openai_client or client
        if oc is None:
            raise RuntimeError("OpenAI client not configured")

        resp = oc.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            response_format={"type": "json_object"},
        )
        content = resp.choices[0].message.content
        if not content:
            raise ValueError("LLM returned empty content")
        
        data = json.loads(content)

        # Constraint: Trim passages count and short quote length to avoid copyright/length issues
        max_passages = max(1, min(q.max_passages, 10))
        passages = (data.get("passages") or [])[:max_passages]
        for p in passages:
            sq = (p.get("short_quote") or "").strip()
            if q.language.startswith("zh"):
                if len(sq) > 40:
                    p["short_quote"] = ""
            else:
                if len(sq.split()) > 20:
                    p["short_quote"] = ""
        data["passages"] = passages

        # Ensure other required fields have default values if missing from LLM response
        data.setdefault("reflection", "")
        data.setdefault("exercise", "")

        if not data.get("disclaimer"):
            data["disclaimer"] = (
                "Please verify passages in your preferred edition/translation; non-public-domain texts are summarized, and this is supportive guidance only."
            )
        
        return data

    except json.JSONDecodeError as e:
        raise ValueError(f"LLM returned invalid JSON: {e}") from e
    except Exception as e:
        raise RuntimeError(f"LLM API call failed: {e}") from e

@app.post("/comfort", response_model=ComfortResponse)
def comfort(q: ComfortQuery):
    # Basic validation
    if not os.environ.get("OPENAI_API_KEY"):
        raise HTTPException(status_code=500, detail="Server missing OPENAI_API_KEY")

    try:
        result = get_comfort_from_openai(q)
        return result
    except (ValueError, RuntimeError) as e:
        raise HTTPException(status_code=502, detail=str(e))
    except Exception as e:
        # Catch any other unexpected errors
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")


def select_voice(language: str, override: Optional[str] = None) -> str:
    """Pick a reasonable default voice by language, unless override provided."""
    if override:
        return override
    # OpenAI voices like 'alloy', 'verse', 'onyx', 'nova' are multi-lingual.
    # Use 'alloy' as a safe default for both zh/en.
    return "alloy"


def generate_tts_audio(
    text: str,
    language: str = "zh",
    voice: Optional[str] = None,
    fmt: str = "mp3",
    *,
    openai_client: Optional[OpenAI] = None,
) -> Tuple[str, str]:
    """
    Generate TTS audio to a temporary file using OpenAI TTS.

    Returns a tuple (temp_file_path, media_type).
    The caller is responsible for deleting the temp file after streaming/usage.
    """
    if not text or not text.strip():
        raise ValueError("Missing text for TTS")

    # Basic caps to avoid extremely long synthesis
    max_chars = 6000
    text = text.strip()
    if len(text) > max_chars:
        text = text[:max_chars]

    chosen_voice = select_voice(language or "zh", voice)
    fmt = (fmt or "mp3").lower()
    if fmt not in {"mp3", "wav"}:
        fmt = "mp3"

    media_type = "audio/mpeg" if fmt == "mp3" else "audio/wav"

    oc = openai_client or client

    with NamedTemporaryFile(delete=False, suffix=f".{fmt}") as tmp:
        temp_path = tmp.name

    # Stream audio to file via SDK
    with oc.audio.speech.with_streaming_response.create(
        model="gpt-4o-mini-tts",
        voice=chosen_voice,
        input=text,
    ) as response:
        response.stream_to_file(temp_path)

    return temp_path, media_type


@app.post("/tts")
def tts(req: TTSRequest):
    if not os.environ.get("OPENAI_API_KEY"):
        raise HTTPException(status_code=500, detail="Server missing OPENAI_API_KEY")

    try:
        temp_path, media_type = generate_tts_audio(
            text=req.text,
            language=req.language or "zh",
            voice=req.voice,
            fmt=req.format or "mp3",
            openai_client=client,
        )

        def file_iterator(path: str, chunk_size: int = 8192):
            with open(path, "rb") as f:
                while True:
                    chunk = f.read(chunk_size)
                    if not chunk:
                        break
                    yield chunk
            try:
                os.remove(path)
            except Exception:
                pass
        return StreamingResponse(file_iterator(temp_path), media_type=media_type)

    except Exception as e:
        # Clean up temp file if allocation happened but streaming failed
        try:
            if 'temp_path' in locals() and os.path.exists(temp_path):
                os.remove(temp_path)
        except Exception:
            pass
        raise HTTPException(status_code=502, detail=f"TTS failed: {e}")
