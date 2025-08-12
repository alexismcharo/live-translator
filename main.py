import tempfile, subprocess, uvicorn, openai, whisper, os, json, uuid, re
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from dotenv import load_dotenv

# --------------------- Setup ---------------------
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = openai.AsyncOpenAI(api_key=api_key)

# Pre-warm ffmpeg so the first request is not slow.
try:
    subprocess.run(
        ["ffmpeg", "-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono", "-t", "0.5", "-ar", "16000", "-ac", "1", "-y", "/tmp/warm.wav"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
except:
    pass

# Whisper model
model = whisper.load_model("large-v3")
try:
    model.transcribe("/tmp/warm.wav", language="en", fp16=True)
except:
    pass

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

# --------------------- Filters (exact-match thank-you; original CTA) ---------------------

# Exact short interjections only (and standalone "you")
THANKS_RE = re.compile(r'^\s*(?:thank\s*you|thanks|thx|you)\s*[!.…]*\s*$', re.IGNORECASE)

def is_interjection_thanks(text: str) -> bool:
    """
    True only for pure short 'thank you' style interjections or a standalone 'you'.
    Does NOT match longer sentences like 'thank you for coming' or 'you should...'.
    """
    if not text:
        return False
    return bool(THANKS_RE.match(text.strip()))

# CTA patterns (same idea as your original examples)
_CTA_PATTERNS = [
    r'(?i)\blike (?:and )?subscribe\b',
    r'(?i)\bshare (?:this|the) (?:video|stream)\b',
    r'(?i)\bhit (?:the )?bell\b',
    r'(?i)\bturn on notifications?\b',
    r'(?i)\blink in (?:the )?(?:bio|description)\b',
    r'(?i)\bsee you (?:next time|in the next|tomorrow)\b',
    r"(?i)\bthanks for watching\b",
    r"(?i)\bthat's (?:it|all) for (?:today|now)\b",
    r'(?i)\bsmash (?:that )?like\b',
]

def is_cta_like(text: str) -> bool:
    """Returns True if the line contains a CTA phrase from the list above."""
    if not text or len(text.strip()) < 2:
        return False
    for pat in _CTA_PATTERNS:
        if re.search(pat, text):
            return True
    return False

# --------------------- Translation (default-only) ---------------------

async def translate_text(text: str, source_lang: str, target_lang: str) -> str:
    """
    Translate a single ASR segment into natural, live-caption style.
    No context merging or dedupe logic here.
    """
    system = "Translate live ASR segments into natural, idiomatic target-language captions. Return ONLY the translation text."
    user = f"""
<goal>
Produce fluent, idiomatic {target_lang} captions for this single ASR segment.
</goal>

<priorities>
1) Preserve meaning faithfully; do not invent content.
2) Prefer natural phrasing over literal word order when safe.
3) Mirror completeness: if input is a fragment, output a natural fragment.
4) Keep numbers as digits; preserve names and units verbatim.
5) Remove pure fillers (uh/um/えっと) unless they convey hesitation/tone.
6) If a phrase repeats with no new info (including ASR restarts), keep it only once.
7) If the input is already {target_lang}, return it unchanged.
8) If the input is a label/title/heading/meta comment, translate it as such without turning it into a full sentence.
9) Preserve mood/person; do not convert first-person statements into imperatives.
</priorities>

<style_targets>
- Tone: clear, concise, speech-like.
- Punctuation: minimal but natural for captions.
</style_targets>

<examples_positive>
<input>I want to … check whether it actually improves the translation quality.</input>
<output>I want to check whether it actually improves the translation quality.</output>

<input>Meeting Agenda — Thursday</input>
<output>Meeting Agenda — Thursday</output>

<input>They arrived late because because the train was delayed.</input>
<output>They arrived late because the train was delayed.</output>
</examples_positive>

<input>
{text}
</input>
""".strip()

    try:
        response = await client.chat.completions.create(
            model="gpt-5",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            # Reasoning-style GPT-5: keep it minimal; sampling/penalties not supported.
            reasoning_effort="minimal",
            max_completion_tokens=140
        )
        return (response.choices[0].message.content or "").strip()
    except Exception as e:
        print("Translation error:", e)
        return text

# --------------------- HTTP ---------------------

@app.get("/")
async def serve_index():
    return FileResponse(os.path.join("frontend", "index.html"))

# --------------------- WebSocket ---------------------

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("WebSocket connected")

    try:
        settings = await websocket.receive_text()
        config = json.loads(settings)
        direction = config.get("direction")
        if direction == "en-ja":
            source_lang, target_lang = "English", "Japanese"
        elif direction == "ja-en":
            source_lang, target_lang = "Japanese", "English"
        else:
            await websocket.close()
            return

        while True:
            msg = await websocket.receive()
            if "bytes" not in msg:
                continue

            audio = msg["bytes"]
            if not audio:
                continue

            # Transcode to WAV
            with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as raw:
                raw.write(audio)
                raw.flush()
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as wav:
                    try:
                        subprocess.run(
                            ["ffmpeg", "-y", "-i", raw.name, "-af", "silenceremove=1:0:-40dB", "-ar", "16000", "-ac", "1", wav.name],
                            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True
                        )
                    except:
                        continue

                    result = model.transcribe(
                        wav.name,
                        fp16=True,
                        temperature=0.0,
                        condition_on_previous_text=True,
                        hallucination_silence_threshold=0.3,
                        no_speech_threshold=0.3,
                        language="en" if source_lang == "English" else "ja",
                        compression_ratio_threshold=2.4,
                        logprob_threshold=-1.0
                    )

                    src_text = (result.get("text") or "").strip()
                    if not src_text:
                        continue
                    print("ASR:", src_text)

                    # Skip short thank-you interjections entirely
                    if is_interjection_thanks(src_text):
                        print("Skipped short thank-you interjection.")
                        continue

                    # Skip CTA/meta filler lines entirely
                    if is_cta_like(src_text):
                        print("Dropped CTA/meta filler (keyword).")
                        continue

                    # Translate this single chunk
                    segment_id = str(uuid.uuid4())
                    translated = await translate_text(src_text, source_lang, target_lang)

                    # Apply the same filters post-translation as a safety net
                    if is_interjection_thanks(translated):
                        print("Post-translation: skipped short thank-you interjection.")
                        continue
                    if is_cta_like(translated):
                        print("Post-translation: dropped CTA/meta filler.")
                        continue

                    await websocket.send_text(f"[DONE]{json.dumps({'id': segment_id, 'text': translated})}")

    except Exception as e:
        print("WebSocket error:", e)
        await websocket.close()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
