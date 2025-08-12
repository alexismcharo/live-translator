import tempfile, subprocess, uvicorn, openai, whisper, os, json, uuid, re
from difflib import SequenceMatcher
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from dotenv import load_dotenv

#  use torch (if available) to decide fp16 usage
try:
    import torch  # type: ignore
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False

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
    # NOTE: fp16 only if CUDA is available. Previously forced True.
    model.transcribe("/tmp/warm.wav", language="en", fp16=(_HAS_TORCH and torch.cuda.is_available()))
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
    s = re.sub(r"[^a-z0-9£$€¥%\-\.ぁ-んァ-ヶ一-龯 ]+", " ", s)  # allow CJK
    s = re.sub(r"\s+", " ", s).strip()
    return s

_CJK_RE = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff]")

def _simple_words(s: str) -> list[str]:
    # Keep previous behavior for Latin; treat continuous CJK as separate chars
    if not s:
        return []
    tokens = []
    for ch in s:
        if _CJK_RE.match(ch):
            tokens.append(ch)
        else:
            # collect latin/nums/symbols similarly to before
            pass
    # Fallback to previous tokenization and then merge
    base = [t for t in re.findall(r"[a-z0-9]+|£|\$|€|¥|kg|lb|lbs|%|\.|,", s.lower()) if t.strip()]
    return tokens + base

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

# NOTE: no longer calling
async def hallucination_check(text: str) -> bool:
    return False

# --------------------- Translation ---------------------

async def translate_text(text, source_lang, target_lang, mode="default"):
    """
    Produce natural, live-caption style translations.
    For updates, keep edits minimal to avoid flicker.
    """
    target_register = "polite" if target_lang == "Japanese" else "neutral"
    recent_target_str = "\n".join(recent_targets[-MAX_RECENT:])

    if mode == "context":
        previous, current = text
        system = "Refine the previous caption using the current one for context. Output only the improved previous."
        user = f"""
<rules>
- Output only the improved translation of <previous>.
- Be natural, do not add new information.
- If <previous> was a fragment, keep it a natural fragment.
- Avoid repeating lines already in <recent_target>.
- Register: {"です・ます" if target_register=="polite" else "casual"} for Japanese; {target_register} for English.
- Do not re-punctuate unless the source line was closed by punctuation.
</rules>
<recent_target>{recent_target_str}</recent_target>
<previous>{previous}</previous>
<current>{current}</current>
""".strip()
    else:
        system = "Translate live ASR segments. Return only the translation text."
        user = f"""
<rules>
- Be natural, not literal; keep meaning faithful.
- Mirror completeness (fragments stay fragments; do not guess endings).
- Remove pure fillers (uh/um/えっと) unless meaningful.
- Keep numbers as digits; preserve names and units.
- If a phrase appears twice with no new info, keep it once.
- Do not add greetings/sign-offs/CTAs.
- If input is already in {target_lang}, return it unchanged.
- Register: {"です・ます" if target_register=="polite" else "casual"} for Japanese; {target_register} for English.
- Do not re-punctuate unless the source line was closed by punctuation.
</rules>
<recent_target>{recent_target_str}</recent_target>
<input>{text}</input>
""".strip()

    try:
        response = await client.chat.completions.create(
            model="gpt-5",
            messages=[{"role":"system","content":system},{"role":"user","content":user}],
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

            # Write temp files and ensure cleanup
            raw_path = None
            wav_path = None
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as raw:
                    raw.write(audio)
                    raw.flush()
                    raw_path = raw.name

                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as wav:
                    wav_path = wav.name
                    try:
                        subprocess.run(
                            ["ffmpeg", "-y", "-i", raw_path, "-af", "silenceremove=1:0:-40dB", "-ar", "16000", "-ac", "1", wav_path],
                            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True
                        )
                    except:
                        continue

                    # Stable settings for live use
                    result = model.transcribe(
                        wav_path,
                        fp16=(_HAS_TORCH and torch.cuda.is_available()),
                        temperature=0.0,
                        condition_on_previous_text=False,
                        # allow biasing via initial prompt
                        initial_prompt=(
                            "Tech for Impact Summit, Socious Global, Beyond Boundaries: Building 2050 Together, "
                            "Toranomon Hills Forum, Tokyo, AI, quantum computing, biotechnology, clean energy, "
                            "impact investing, social entrepreneurship, sustainability, Audrey Tang, Charles Hoskinson, "
                            "Seira Yun, Hector Zenil, Ken Kodama, Alex Zapesochny, DEI, brain-computer interfaces"
                        ),
                        # CHANGE: tighter silence / hallucination gates
                        hallucination_silence_threshold=0.55,
                        no_speech_threshold=0.4,
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

                    # Fast CTA keyword gate only (drop extra LLM call to reduce latency).
                    if is_cta_like(src_text):
                        print("Dropped CTA/meta filler (keyword).")
                        continue

                    # Remove overlap against previous ASR tail.
                    delta_src = strip_overlap(prev_src_tail, src_text)
                    if not delta_src:
                        print("Skipped chunk (entirely overlap).")
                        continue

                    # Update previous tail (keep last ~30 tokens).
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
            finally:
                # cleanup temp files
                try:
                    if raw_path and os.path.exists(raw_path):
                        os.unlink(raw_path)
                except Exception:
                    pass
                try:
                    if wav_path and os.path.exists(wav_path):
                        os.unlink(wav_path)
                except Exception:
                    pass

    except Exception as e:
        print("WebSocket error:", e)
        await websocket.close()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
