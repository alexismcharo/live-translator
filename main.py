import tempfile, subprocess, uvicorn, openai, os, json, uuid, re
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from dotenv import load_dotenv

# basic setup and client
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = openai.AsyncOpenAI(api_key=api_key)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

# tiny context buffers for current-line translation only (used for EN->JA)
recent_src_segments: list[str] = []
recent_targets: list[str] = []
MAX_SRC_CTX = 2
MAX_RECENT = 10

# exact short interjections only (and standalone "you") — English-only hallucination filter (pre-translation)
THANKS_RE = re.compile(r'^\s*(?:thank\s*you|thanks|thx|you)\s*[!.…]*\s*$', re.IGNORECASE)

def is_interjection_thanks(text: str) -> bool:
    if not text:
        return False
    return bool(THANKS_RE.match(text.strip()))

# CTA patterns (English-only hallucination filter; pre-translation)
_CTA_PATTERNS = [
    r'(?i)\blike\s*(?:and\s*)?subscribe\b',
    r'(?i)\bsubscribe\s*(?:and\s*)?like\b',
    r'(?i)\bshare\s+(?:this|the)\s+(?:video|stream|clip|content)\b',
    r'(?i)\bplease\s+share\b',
    r'(?i)\bhit\s+(?:the\s+)?bell\b',
    r'(?i)\bturn\s+on\s+notifications?\b',
    r"(?i)\bdon'?t\s+forget\s+to\s+turn\s+on\s+notifications?\b",
    r'(?i)\blink\s+in\s+(?:the\s+)?(?:bio|description)\b',
    r'(?i)\bcheck\s+(?:the\s+)?link\s+(?:below|above)\b',
    r'(?i)\bsee\s+you\s+(?:next\s*time|tomorrow|soon|in\s+the\s+next)\b',
    r'(?i)\bthanks?\s+for\s+watching\b',
    r'(?i)\bthank\s+you\s+for\s+watching\b',
    r'(?i)\bthank\s+you\s+so\s+much\s+for\s+watching\b',
    r'(?i)\bthat\'?s\s+(?:it|all)\s+(?:for\s+(?:today|now))\b',
    r'(?i)\bthat\'?s\s+(?:it|all)\b',
    r'(?i)\bsmash\s+(?:that\s+)?like\b',
]

def is_cta_like(text: str) -> bool:
    if not text or len(text.strip()) < 2:
        return False
    for pat in _CTA_PATTERNS:
        if re.search(pat, text):
            return True
    return False

async def translate_text(text: str, source_lang: str, target_lang: str) -> str:
    # small source + target tails only to help this line (no previous-line updates)
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
1) Preserve meaning; do not add or remove information.
2) Prefer idiomatic phrasing over literal order when safe.
3) Mirror completeness with context: if the input alone is a fragment, but combining it with <source_context> (the immediately preceding ASR text) yields a clear, unambiguous complete sentence, output the completed sentence with natural punctuation. Otherwise keep a natural fragment and omit the final full stop; internal commas/dashes may still be used.
4) Keep numbers as digits; preserve units and proper names exactly as heard.
5) Remove pure fillers (uh/um/えっと) unless they convey hesitation/tone.
6) Collapse overlap/restarts: keep a repeated phrase only once if no new info.
7) If input is already {target_lang}, return it unchanged.
8) Translate labels/titles/meta as such; do not expand into full sentences.
9) Preserve grammatical person/mood; do not convert first-person into imperatives.
10) If target is Japanese: output katakana + Latin in parentheses on first mention, katakana thereafter. If unsure of a name, do not “correct” it.
11) Drop standalone low-content interjections (e.g., “newspaper.” / “article.” / “video.”) unless they add source/brand/modifier info.
12) Maintain one consistent spelling per proper noun within the session.
13) If the line restates content already fully covered in <recent_target> with no new detail, output nothing.
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
        return ""

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

            # transcode to wav
            with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as raw:
                raw.write(audio)
                raw.flush()
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as wav:
                    try:
                        subprocess.run(
                            ["ffmpeg", "-y", "-i", raw.name,
                             "-af", "silenceremove=1:0:-30dB",
                             "-ar", "16000", "-ac", "1", wav.name],
                            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True
                        )
                    except:
                        continue

                    # 1) EN->JA: gpt-4o-transcribe for ASR, then gpt-5 for translation
                    if source_lang == "English" and target_lang == "Japanese":
                        try:
                            with open(wav.name, "rb") as f:
                                en_text = await client.audio.transcriptions.create(
                                    model="gpt-4o-transcribe",
                                    file=f,
                                    response_format="text"
                                )
                            src_text = (en_text or "").strip()
                        except Exception as e:
                            print("Transcriptions API error (EN):", e)
                            continue

                        if not src_text:
                            continue
                        print("ASR (EN):", src_text)

                        # 3) English-only hallucination filters BEFORE translation
                        if is_interjection_thanks(src_text):
                            print("Skipped short thank-you interjection (EN pre-translate).")
                            continue
                        if is_cta_like(src_text):
                            print("Dropped CTA/meta filler (EN pre-translate).")
                            continue

                        segment_id = str(uuid.uuid4())
                        translated = await translate_text(src_text, source_lang, target_lang)
                        translated = translated.strip()
                        if not translated or not re.search(r'[A-Za-z0-9ぁ-んァ-ン一-龯々ー]', translated):
                            print("Suppressed empty/no-op output.")
                            continue

                        recent_src_segments.append(src_text)
                        if len(recent_src_segments) > MAX_SRC_CTX * 3:
                            recent_src_segments.pop(0)

                        recent_targets.append(translated)
                        if len(recent_targets) > MAX_RECENT:
                            recent_targets.pop(0)

                        await websocket.send_text(f"[DONE]{json.dumps({'id': segment_id, 'text': translated})}")

                    # 2) JA->EN: gpt-4o-transcribe for both transcription+translation to EN
                    else:
                        try:
                            with open(wav.name, "rb") as f:
                                ja_en_text = await client.audio.translations.create(
                                    model="gpt-4o-transcribe",
                                    file=f,
                                    response_format="text"
                                )
                            translated = (ja_en_text or "").strip()
                        except Exception as e:
                            print("Translations API error (JA->EN):", e)
                            continue

                        if not translated:
                            continue

                        if is_interjection_thanks(translated):
                            print("Skipped short thank-you interjection (EN pre-emit).")
                            continue
                        if is_cta_like(translated):
                            print("Dropped CTA/meta filler (EN pre-emit).")
                            continue
                        
                        segment_id = str(uuid.uuid4())
                        if not re.search(r'[A-Za-z0-9ぁ-んァ-ン一-龯々ー]', translated):
                            print("Suppressed empty/no-op output.")
                            continue

                        recent_targets.append(translated)
                        if len(recent_targets) > MAX_RECENT:
                            recent_targets.pop(0)

                        await websocket.send_text(f"[DONE]{json.dumps({'id': segment_id, 'text': translated})}")

    except Exception as e:
        print("WebSocket error:", e)
        await websocket.close()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
