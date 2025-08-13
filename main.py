import tempfile, subprocess, uvicorn, openai, whisper, os, json, uuid, re
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from dotenv import load_dotenv

# basic setup and client
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = openai.AsyncOpenAI(api_key=api_key)

# pre-warm ffmpeg so the first request isn't slow
try:
    subprocess.run(
        ["ffmpeg", "-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono", "-t", "0.5", "-ar", "16000", "-ac", "1", "-y", "/tmp/warm.wav"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
except:
    pass

# load whisper once
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

# tiny context buffers for current-line translation only
recent_src_segments: list[str] = []
recent_targets: list[str] = []
MAX_SRC_CTX = 3
MAX_RECENT = 10

# exact short interjections only (and standalone "you")
THANKS_RE = re.compile(r'^\s*(?:thank\s*you|thanks|thx|you)\s*[!.…]*\s*$', re.IGNORECASE)

def is_interjection_thanks(text: str) -> bool:
    # only treat pure short interjections or a lone "you" as thank-you lines
    if not text:
        return False
    return bool(THANKS_RE.match(text.strip()))

# CTA patterns (keep these as hard filters)
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
    if not text or len(text.strip()) < 2:
        return False
    for pat in _CTA_PATTERNS:
        if re.search(pat, text):
            return True
    return False

async def translate_text(text: str, source_lang: str, target_lang: str) -> str:
    # use a small source + target tail only to help this line (no previous-line updates)
    source_context = " ".join(recent_src_segments[-MAX_SRC_CTX:])
    recent_target_str = "\n".join(recent_targets[-MAX_RECENT:])

    system = "Translate live ASR segments into natural, idiomatic target-language captions. Return ONLY the translation text."
    user = f"""
<goal>
Produce fluent, idiomatic {target_lang} for THIS single ASR segment, using context only to disambiguate and avoid repetition.
</goal>

<context_use>
- Use <source_context> to resolve continuations or ambiguous references.
- Do not re-translate content already fully covered in <recent_target> unless the new input adds substantive information.
- If the current input repeats a clause from <source_context>, keep it once in the cleanest form.
</context_use>

<priorities>
1) Preserve meaning faithfully; do not invent content.
2) Prefer natural phrasing over literal word order when safe.
3) Mirror completeness: if input is a fragment, output a natural fragment (do not guess endings).
4) Keep numbers as digits; preserve names and units verbatim.
5) Remove pure fillers (uh/um/えっと) unless they convey hesitation/tone.
6) If a phrase repeats with no new info (including overlap/restarts), keep it only once.
7) If input is already {target_lang}, return it unchanged.
8) If input is a label/title/heading/meta comment, translate it as such without turning it into a full sentence.
9) Preserve mood/person; do not convert first-person statements into imperatives.
10) Preserve names as heard; do not invent or ‘correct’ them. If unsure, leave as-is without adding titles.
</priorities>

<style_targets>
- Tone: clear, concise, speech-like.
- Punctuation: minimal but natural for captions.
</style_targets>

<source_context>
{source_context}
</source_context>

<recent_target>
{recent_target_str}
</recent_target>

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
        resp = await client.chat.completions.create(
            model="gpt-5",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            reasoning_effort="minimal",
            max_completion_tokens=140
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        print("Translation error:", e)
        return text

@app.get("/")
async def serve_index():
    return FileResponse(os.path.join("frontend", "index.html"))

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

            with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as raw:
                raw.write(audio)
                raw.flush()
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as wav:
                    try:
                        subprocess.run(
                            ["ffmpeg", "-y", "-i", raw.name, "-af", "silenceremove=1:0:-35dB", "-ar", "16000", "-ac", "1", wav.name],
                            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True
                        )
                    except:
                        continue

                    result = model.transcribe(
                        wav.name,
                        fp16=True,
                        temperature=0.0,
                        condition_on_previous_text=False,
                        hallucination_silence_threshold=0.50,
                        no_speech_threshold=0.4,
                        language="en" if source_lang == "English" else "ja",
                        compression_ratio_threshold=2.2,
                        logprob_threshold=-0.8
                    )

                    src_text = (result.get("text") or "").strip()
                    if not src_text:
                        continue
                    print("ASR:", src_text)

                    if is_interjection_thanks(src_text):
                        print("Skipped short thank-you interjection (source).")
                        continue
                    if is_cta_like(src_text):
                        print("Dropped CTA/meta filler (source).")
                        continue

                    segment_id = str(uuid.uuid4())
                    translated = await translate_text(src_text, source_lang, target_lang)

                    if is_interjection_thanks(translated):
                        print("Skipped short thank-you interjection (target).")
                        continue
                    if is_cta_like(translated):
                        print("Dropped CTA/meta filler (target).")
                        continue

                    # update tiny context buffers only when emitting
                    recent_src_segments.append(src_text)
                    if len(recent_src_segments) > MAX_SRC_CTX * 3:
                        recent_src_segments.pop(0)

                    recent_targets.append(translated)
                    if len(recent_targets) > MAX_RECENT:
                        recent_targets.pop(0)

                    await websocket.send_text(f"[DONE]{json.dumps({'id': segment_id, 'text': translated})}")

    except Exception as e:
        print("WebSocket error:", e)
        await websocket.close()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
