import tempfile, subprocess, uvicorn, openai, whisper, os, json, uuid, re
from difflib import SequenceMatcher
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from dotenv import load_dotenv

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

# Histories for repetition control
transcript_history = []   # [(segment_id, original_text)]
recent_targets = []       # last translated outputs shown to the user
MAX_RECENT = 15
MAX_TRANSCRIPTS = 200

# --------------------- Normalization helpers ---------------------

def _normalize_for_compare(s: str) -> str:
    if not s:
        return ""
    s = s.lower()
    s = s.replace("kg.", "kg").replace(" kg", "kg")
    s = re.sub(r"[^a-z0-9£$€¥%\-\. ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _simple_words(s: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+|£|\$|€|¥|kg|lb|lbs|%|\.|,", s.lower()) if t.strip()]

# --------------------- Source-side overlap stripper ---------------------

def strip_overlap(prev_src_tail: str, curr_src: str, window_words: int = 30, min_match: int = 4) -> str:
    """
    Remove duplicated prefix in the current ASR text that already appeared
    as the suffix of the previous ASR text.
    """
    if not prev_src_tail:
        return curr_src

    pw = _simple_words(prev_src_tail)
    cw = _simple_words(curr_src)
    if not pw or not cw:
        return curr_src

    max_k = min(window_words, len(pw), len(cw))
    best_k = 0
    for k in range(max_k, min_match - 1, -1):
        if pw[-k:] == cw[:k]:
            best_k = k
            break

    if best_k == 0:
        return curr_src

    count = 0
    cut_index = 0
    for m in re.finditer(r"[a-z0-9]+|£|\$|€|¥|kg|lb|lbs|%|\.|,|\s+|.", curr_src, flags=re.IGNORECASE):
        tok = m.group(0)
        if re.fullmatch(r"[a-z0-9]+|£|\$|€|¥|kg|lb|lbs|%|\.|,", tok, flags=re.IGNORECASE):
            count += 1
            if count == best_k:
                cut_index = m.end()
                break
    while cut_index < len(curr_src) and curr_src[cut_index].isspace():
        cut_index += 1
    return curr_src[cut_index:] or ""

# --------------------- In-line de-dupe (single translation) ---------------------

def dedupe_repeated_ngrams(text: str, n: int = 3, min_run_chars: int = 6) -> str:
    """
    Remove adjacent duplicate n-gram runs such as "included ... included".
    """
    if not text:
        return text
    tokens = re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)
    lower = [t.lower() for t in tokens]

    i = 0
    out = []
    while i < len(tokens):
        out.append(tokens[i])
        if i + n < len(tokens):
            prev_chunk = lower[max(0, len(out)-n):len(out)]
            next_chunk = lower[i+1:i+1+n]
            if len(prev_chunk) == n and prev_chunk == next_chunk:
                if sum(len(t) for t in next_chunk) >= min_run_chars:
                    i += n  # skip duplicate run
        i += 1
    s = "".join(out)
    s = re.sub(r"\s+\.", ".", s)
    s = re.sub(r"\s+,", ",", s)
    return s.strip()

# --------------------- Cross-line near-duplicate guard ---------------------

def looks_like_recent_duplicate(new_text: str, history: list[str],
                                ratio_threshold: float = 0.9,
                                contain_threshold: float = 0.9) -> bool:
    norm_new = _normalize_for_compare(new_text)
    if not norm_new:
        return False
    for old in reversed(history):
        norm_old = _normalize_for_compare(old)
        if not norm_old:
            continue
        short, long_ = (norm_new, norm_old) if len(norm_new) <= len(norm_old) else (norm_old, norm_new)
        if len(short) >= 6 and short in long_ and len(short)/len(long_) >= contain_threshold:
            return True
        if SequenceMatcher(None, norm_new, norm_old).ratio() >= ratio_threshold:
            return True
    return False

# --------------------- CTA / thank-you filtering ---------------------

THANKS_RE = re.compile(r'(?i)^\s*(?:thank\s*you|thanks|thx)\s*[!.…]*\s*$')

def is_interjection_thanks(text: str) -> bool:
    if not text:
        return False
    return bool(THANKS_RE.match(text.strip()))


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

_SOFT_VIEWER_ADDRESS = [
    r"(?i)\bhey (?:guys|everyone|folks)\b",
    r"(?i)\bwhat's up\b"
]

def is_cta_like(text: str) -> bool:
    if not text or len(text.strip()) < 2:
        return False
    for pat in _CTA_PATTERNS:
        if re.search(pat, text):
            return True
    return False

def is_soft_address(text: str) -> bool:
    if not text:
        return False
    return any(re.search(pat, text) for pat in _SOFT_VIEWER_ADDRESS)

async def hallucination_check(text: str) -> bool:
    pass

# --------------------- Translation ---------------------

async def translate_text(text, source_lang, target_lang, mode="default"):
    """
    Produce natural, live-caption style translations with minimal flicker.
    Polishes for idiomatic phrasing while preserving meaning and completeness.
    """
    # Recent target memory (assumes globals recent_targets, MAX_RECENT exist)
    recent_target_str = "\n".join(recent_targets[-MAX_RECENT:])

    if mode == "context":
        # text = (previous, current)
        previous, current = text
        system = (
            "You are polishing live captions to sound idiomatic and natural in the target language. "
            "Your job: improve the PREVIOUS line using CURRENT as context, then output ONLY the improved previous."
        )
        user = f"""
<goal>
Return a single, natural, idiomatic {target_lang} line that preserves the meaning of <previous>,
optionally using <current> for disambiguation. Improve fluency, word choice, and rhythm.
</goal>

<priorities>
1) Faithful meaning > natural flow > brevity.
2) If <previous> is a fragment, keep it a natural fragment (do not complete it).
3) Prefer common collocations and paraphrases over literal word order.
4) Remove filler/disfluencies (uh/um/えっと) unless semantically meaningful.
5) Avoid repeating lines already present in <recent_target>.
</priorities>

<language_tips>
- Japanese: prefer everyday phrasing; smooth particles; avoid stiff calques; naturalize set phrases.
- English: prefer contractions and common phrasing; avoid source-language word order when awkward.
</language_tips>

<forbidden>
- Adding information not implied by <previous> or <current>.
- Glosses, brackets, notes, quotes, or explanations. Output text only.
</forbidden>

<recent_target>
{recent_target_str}
</recent_target>

<previous>
{previous}
</previous>

<current>
{current}
</current>
""".strip()
    else:
        system = (
            "Translate live ASR segments into natural, idiomatic target-language captions. "
            "Return ONLY the translation text."
        )
        user = f"""
<goal>
Produce fluent, idiomatic {target_lang} that reads like a native speaker wrote it.
</goal>

<priorities>
1) Preserve meaning faithfully; do not invent content.
2) Prefer natural phrasing over literal word order when safe.
3) Mirror completeness: if input is a fragment, output a natural fragment.
4) Remove pure fillers (uh/um/えっと) unless they convey hesitation or tone that matters.
5) Keep numbers as digits; preserve names and units verbatim.
6) If a phrase repeats with no new info, keep it once.
7) If input is already in {target_lang}, return it unchanged.
</priorities>

<language_tips>
- Japanese: prefer everyday collocations; avoid stiff 「〜することができる」 when 「〜できる」 is natural;
  drop unnecessary subjects; choose particles for smoothness; avoid calques.
- English: use contractions, natural verbs, and common collocations; avoid source-language punctuation/order when awkward.
</language_tips>

<style_targets>
- Tone: clear, concise, speech-like.
- Punctuation: minimal but natural for captions.
</style_targets>

<recent_target>
{recent_target_str}
</recent_target>

<input>
{text}
</input>
""".strip()

    try:
        response = await client.chat.completions.create(
            model="gpt-5",
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            reasoning_effort="minimal",
            verbosity="low"
        )
        raw = (response.choices[0].message.content or "").strip()
        out = raw if mode == "context" else dedupe_repeated_ngrams(raw, n=3)
        return out
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

    prev_src_tail = ""

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
                            ["ffmpeg", "-y", "-i", raw.name, "-af", "silenceremove=1:0:-40dB", "-ar", "16000", "-ac", "1", wav.name],
                            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True
                        )
                    except:
                        continue

                    # Stable settings for live use
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

                    # Short backchannels pass (they are part of natural speech)
                    if src_text.strip().lower() in {"yeah", "yep", "okay", "ok", "right", "sure", "actually", "fair enough"}:
                        pass

                    # Narrow thank-you filter: only drop pure interjections.
                    if is_interjection_thanks(src_text):
                        print("Skipped short thank-you interjection.")
                        continue

                    # Fast CTA keyword gate; if no hit, avoid LLM call unless needed.
                    if is_cta_like(src_text):
                        print("Dropped CTA/meta filler (keyword).")
                        continue

                    needs_model_check = len(src_text.split()) >= 4 and not is_soft_address(src_text)

                    # Remove overlap against previous ASR tail.
                    delta_src = strip_overlap(prev_src_tail, src_text)
                    if not delta_src:
                        print("Skipped chunk (entirely overlap).")
                        continue

                    # Update previous tail (keep last ~30 words).
                    joined = (prev_src_tail + " " + src_text).strip()
                    tail_words = _simple_words(joined)[-30:]
                    prev_src_tail = " ".join(tail_words)

                    # Translate only the new portion.
                    segment_id = str(uuid.uuid4())
                    transcript_history.append((segment_id, delta_src))
                    if len(transcript_history) > MAX_TRANSCRIPTS:
                        transcript_history.pop(0)

                    translated = await translate_text(delta_src, source_lang, target_lang)
                    translated = dedupe_repeated_ngrams(translated, n=3)

                    # Cross-line near-duplicate guard.
                    if looks_like_recent_duplicate(translated, recent_targets):
                        print("Dropped near-duplicate line.")
                    else:
                        await websocket.send_text(f"[DONE]{json.dumps({'id': segment_id, 'text': translated})}")
                        recent_targets.append(translated)
                        if len(recent_targets) > MAX_RECENT:
                            recent_targets.pop(0)

                    # Gentle refinement of the previous line with new context.
                    if len(transcript_history) >= 2:
                        prev, curr = transcript_history[-2][1], transcript_history[-1][1]
                        improved = await translate_text((prev, curr), source_lang, target_lang, mode="context")
                        await websocket.send_text(f"[UPDATE]{json.dumps({'id': transcript_history[-2][0], 'text': improved})}")

    except Exception as e:
        print("WebSocket error:", e)
        await websocket.close()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
