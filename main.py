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
MAX_SRC_CTX = 2
MAX_RECENT = 10

# exact short interjections only (and standalone "you")
THANKS_RE = re.compile(r'^\s*(?:thank\s*you|thanks|thx|you)\s*[!.…]*\s*$', re.IGNORECASE)

def is_interjection_thanks(text: str) -> bool:
    # only treat pure short interjections or a lone "you" as thank-you lines
    if not text:
        return False
    return bool(THANKS_RE.match(text.strip()))

_CTA_PATTERNS = [
    # Like / subscribe combos
    r'(?i)\blike\s*(?:and\s*)?subscribe\b',
    r'(?i)\bsubscribe\s*(?:and\s*)?like\b',

    # Share requests
    r'(?i)\bshare\s+(?:this|the)\s+(?:video|stream|clip|content)\b',
    r'(?i)\bplease\s+share\b',

    # Notifications / bell
    r'(?i)\bhit\s+(?:the\s+)?bell\b',
    r'(?i)\bturn\s+on\s+notifications?\b',
    r'(?i)\bdon\'?t\s+forget\s+to\s+turn\s+on\s+notifications?\b',

    # Link references
    r'(?i)\blink\s+in\s+(?:the\s+)?(?:bio|description)\b',
    r'(?i)\bcheck\s+(?:the\s+)?link\s+(?:below|above)\b',

    # Farewell / see you next time
    r'(?i)\bsee\s+you\s+(?:next\s*time|tomorrow|soon|in\s+the\s+next)\b',

    # Thanks for watching (all variants)
    r'(?i)\bthanks?\s+for\s+watching\b',
    r'(?i)\bthank\s+you\s+for\s+watching\b',
    r'(?i)\bthank\s+you\s+so\s+much\s+for\s+watching\b',
    r"(?i)\bthank\s+you\s+y['’]?all\b",
    r"(?i)\bthanks?\s+y['’]?all\b",

    # That's it / that's all
    r'(?i)\bthat\'?s\s+(?:it|all)\s+for\s+(?:today|now)\b',
    r'(?i)\bthat\'?s\s+(?:it|all)\b',

    # Smash that like
    r'(?i)\bsmash\s+(?:that\s+)?like\b',

    # Subtitle / caption credits
    r'(?i)\bsubtitles?\s+by\b',
    r'(?i)\bcaptions?\s+by\b',
    r'(?i)\bsubtitled\s+by\b',
    r'(?i)\bclosed\s+captions?\s+by\b',
    r'(?i)\bamara\.org\b',
    r'(?i)\btranscription\s+by\b'
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
Produce fluent, idiomatic {target_lang} for THIS single ASR segment exactly as spoken. Preserve original word strength, tone, and register without upgrading or downgrading. Use context only to resolve ambiguous pronouns or incomplete words.
</goal>

<context_use>
- Refer to <source_context> only to resolve ambiguity in pronouns, ellipses, or cut-off words. Do not merge, rewrite, or restate the current input beyond what is necessary to produce a faithful translation of *this segment alone*.
- Do not re-translate content already fully covered in <recent_target> unless the new input adds substantive information.
- If the current input repeats a clause from <source_context>, keep it once in the cleanest form.
</context_use>

<priorities>
1) Fidelity above all — Translate exactly what is said, preserving both meaning and *emotional strength* of each word or phrase. 
2) No upgrades or downgrades in intensity — keep subjective strength identical. ("weak" ≠ "terrible", "minimal effort" ≠ "for free", "quite strong" ≠ "dominant").
3) No invented filler phrases, rhetorical asides, or emphasis markers unless explicitly present in the source audio or text.
4) When choosing natural {target_lang} phrasing, do not change register or tone from the original.
5) If the current input overlaps with <source_context> or <recent_target>, remove duplicated words but do not rephrase earlier material unless it is literally repeated here.
6) Keep numbers as digits; preserve units, symbols, and proper names exactly as heard.
7) Only describe vocal events if explicitly audible in the source. Do not invent them.
8) If input is already {target_lang}, return it unchanged.
9) Translate labels, titles, and meta comments as such; do not expand into full sentences.
10) Preserve grammatical person and mood; do not change first-person statements into imperatives or alter register.
11) Drop standalone low-content interjections unless they provide brand/source/modifier info.
12) Maintain consistent spelling and form for each proper noun in the session.
13) Redundancy filter: if the current line restates content already fully covered in <recent_target> with no new detail, output nothing.
</priorities>



<style_targets>
- Clear and concise.
- Natural phrasing only if it preserves the original meaning, tone, and intensity exactly.
- No added flair, emphasis, or commentary.
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

            with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as raw:
                raw.write(audio)
                raw.flush()
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as wav:
                    try:
                        subprocess.run(
                            ["ffmpeg", "-y", "-i", raw.name, "-ar", "16000", "-ac", "1", wav.name],
                            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True
                        )
                    except:
                        continue

                    result = model.transcribe(
                        wav.name,
                        fp16=True,
                        temperature=0.0,
                        beam_size=6, 
                        condition_on_previous_text=False,
                        hallucination_silence_threshold=0.30,
                        no_speech_threshold=0.6,
                        language="en" if source_lang == "English" else "ja",
                        compression_ratio_threshold=2.0,
                        logprob_threshold=-1.2,  
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

                    translated = translated.strip()
                    if not translated or not re.search(r'[A-Za-z0-9ぁ-んァ-ン一-龯々ー]', translated):
                        print("Suppressed empty/no-op output.")
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
