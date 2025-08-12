import tempfile, subprocess, uvicorn, openai, whisper, os, json, uuid, re
from difflib import SequenceMatcher
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = openai.AsyncOpenAI(api_key=api_key)

# Warm up ffmpeg so the first request isn't sluggish.
try:
    subprocess.run(
        ["ffmpeg","-f","lavfi","-i","anullsrc=r=16000:cl=mono","-t","0.5","-ar","16000","-ac","1","-y","/tmp/warm.wav"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
except:
    pass

# Whisper: great quality, we’ll keep settings stable and predictable for live use.
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

# Light histories to help with repetition control.
transcript_history = []   # [(segment_id, original_text)]
recent_targets = []       # last few translations shown to the user
MAX_RECENT = 15
MAX_TRANSCRIPTS = 200

# --------------------- Normalization helpers (keep them fast/simple) ---------------------

def _normalize_for_compare(s: str) -> str:
    """Lowercase, trim spaces, keep common symbols/units. Good enough for fuzzy checks."""
    if not s:
        return ""
    s = s.lower()
    s = s.replace("kg.", "kg").replace(" kg", "kg")
    s = re.sub(r"[^a-z0-9£$€¥%\-\. ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _simple_words(s: str) -> list[str]:
    """
    Very plain word tokenizer for overlap checks.
    Works well for English source (your case). For JA, it still helps on numbers/units.
    """
    # keep words/numbers + currency/unit-ish tokens
    return [t for t in re.findall(r"[a-z0-9]+|£|\$|€|¥|kg|lb|lbs|%|\.|,", s.lower()) if t.strip()]

# --------------------- 1) Source-side overlap stripper (pre-translation) -----------------

def strip_overlap(prev_src_tail: str, curr_src: str, window_words: int = 30, min_match: int = 4) -> str:
    """
    Remove duplicated *prefix* in the current ASR text that already appeared as the *suffix*
    of the previous ASR text. This stops us from translating the same chunk twice.
    """
    if not prev_src_tail:
        return curr_src

    pw = _simple_words(prev_src_tail)
    cw = _simple_words(curr_src)
    if not pw or not cw:
        return curr_src

    # Compare only the last N words of prev to the first N words of curr.
    max_k = min(window_words, len(pw), len(cw))
    best_k = 0
    for k in range(max_k, min_match - 1, -1):
        if pw[-k:] == cw[:k]:
            best_k = k
            break

    if best_k == 0:
        return curr_src

    # We matched on words; now cut the corresponding prefix *in the original text*.
    # We do this by walking curr_src and counting word tokens until we've skipped best_k.
    count = 0
    cut_index = 0
    for m in re.finditer(r"[a-z0-9]+|£|\$|€|¥|kg|lb|lbs|%|\.|,|\s+|.", curr_src, flags=re.IGNORECASE):
        tok = m.group(0)
        if re.fullmatch(r"[a-z0-9]+|£|\$|€|¥|kg|lb|lbs|%|\.|,", tok, flags=re.IGNORECASE):
            count += 1
            if count == best_k:
                cut_index = m.end()
                break
    # Add any trailing spaces after the matched region
    while cut_index < len(curr_src) and curr_src[cut_index].isspace():
        cut_index += 1
    return curr_src[cut_index:] or ""

# --------------------- 2) In-line de-dupe (within a single translation) ------------------

def dedupe_repeated_ngrams(text: str, n: int = 3, min_run_chars: int = 6) -> str:
    """
    Kill obvious in-line repeats like "included ... included" or
    "22 kg per child ... 22 kg per child".
    We remove *adjacent* duplicate n-gram runs; keeps legitimate emphasis intact.
    """
    if not text:
        return text
    # Tokenize gently (don’t destroy punctuation for display)
    tokens = re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)
    lower = [t.lower() for t in tokens]

    i = 0
    out = []
    while i < len(tokens):
        out.append(tokens[i])
        # Try to see if the next n tokens are exactly the same as the previous n tokens.
        if i + n < len(tokens):
            prev_chunk = lower[max(0, len(out)-n):len(out)]
            next_chunk = lower[i+1:i+1+n]
            if len(prev_chunk) == n and prev_chunk == next_chunk:
                # Drop the duplicate run (skip ahead by n)
                # Only do this if it's not just tiny punctuation noise.
                if sum(len(t) for t in next_chunk) >= min_run_chars:
                    i += n  # skip the duplicate
        i += 1
    return re.sub(r"\s+\.", ".", re.sub(r"\s+,", ",", "".join(out))).strip()

# --------------------- 3) Cross-line near-duplicate guard (history) ---------------------

def looks_like_recent_duplicate(new_text: str, history: list[str],
                                ratio_threshold: float = 0.82,
                                contain_threshold: float = 0.80) -> bool:
    """
    Final sanity check: if this whole line is basically the same as something
    we just showed, don't send it again.
    """
    norm_new = _normalize_for_compare(new_text)
    if not norm_new:
        return False
    for old in reversed(history):
        norm_old = _normalize_for_compare(old)
        if not norm_old:
            continue
        # containment
        short, long_ = (norm_new, norm_old) if len(norm_new) <= len(norm_old) else (norm_old, norm_new)
        if len(short) >= 6 and short in long_ and len(short)/len(long_) >= contain_threshold:
            return True
        # char-level similarity
        if SequenceMatcher(None, norm_new, norm_old).ratio() >= ratio_threshold:
            return True
    return False

# --------------------- CTA / fluff filter (same behavior, friendlier comments) ----------

async def hallucination_check(text: str) -> bool:
    """
    Return True if the segment looks like channel fluff (subscribe/like/sign-off).
    Normal conversation should pass through.
    """
    try:
        seg = (text or "").strip()
        if not seg:
            return False

        system = "You are a binary classifier for a LIVE ASR → translation pipeline. Return exactly YES or NO."
        user = f"Is this broadcast-style filler we should drop?\n<segment>{seg}</segment>"

        result = await client.chat.completions.create(
            model="gpt-5",
            messages=[{"role":"system","content":system},{"role":"user","content":user}],
            reasoning_effort="minimal", verbosity="low"
        )
        out = (result.choices[0].message.content or "").strip().upper()
        return out == "YES"
    except Exception as e:
        print("❌ Hallucination check error:", e)
        return False

# --------------------- Translator (unchanged behavior, clearer rules) -------------------

async def translate_text(text, source_lang, target_lang, mode="default"):
    """
    Natural, live-caption style translation. We keep edits small on updates.
    """
    target_register = "polite" if target_lang == "Japanese" else "neutral"
    recent_target_str = "\n".join(recent_targets[-MAX_RECENT:])

    if mode == "context":
        previous, current = text
        system = "You refine the previous caption using the current one for context. Output only the improved previous."
        user = f"""
<rules>
- Output ONLY the improved translation of <previous>.
- Keep it natural in {target_lang} and don't add new info.
- If <previous> was a fragment, keep it a natural fragment.
- Avoid repeating lines seen in <recent_target>.
- Register: {"です・ます" if target_register=="polite" else "casual"} for Japanese; {target_register} for English.
</rules>
<recent_target>{recent_target_str}</recent_target>
<previous>{previous}</previous>
<current>{current}</current>
""".strip()
    else:
        system = "You are a live translator. Return only the translation text—no extra words."
        user = f"""
<rules>
- Be natural, not literal; keep meaning faithful.
- Mirror completeness (fragments stay fragments; no guessing endings).
- Remove pure fillers (uh/um/えっと).
- Keep numbers as digits; preserve names/units.
- If a phrase appears twice with no new info, keep it once.
- Don't add greetings/sign-offs/CTAs.
- If input is already in {target_lang}, return it unchanged.
- Register: {"です・ます" if target_register=="polite" else "casual"} for Japanese; {target_register} for English.
</rules>
<recent_target>{recent_target_str}</recent_target>
<input>{text}</input>
""".strip()

    try:
        response = await client.chat.completions.create(
            model="gpt-5",
            messages=[{"role":"system","content":system},{"role":"user","content":user}],
            reasoning_effort="minimal", verbosity="low"
        )
        raw = (response.choices[0].message.content or "").strip()
        # For [UPDATE] keep exact-only de-dupe (no flicker); for [DONE] do fuzzy de-dupe.
        out = raw if mode == "context" else dedupe_repeated_ngrams(raw, n=3)
        return out
    except Exception as e:
        print("❌ Translation error:", e)
        return text

# --------------------- Minimal UI server ---------------------

@app.get("/")
async def serve_index():
    return FileResponse(os.path.join("frontend", "index.html"))

# --------------------- WebSocket streaming ---------------------

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("🔌 WebSocket connected")

    # Keep a small tail of the previous raw ASR (source) to strip overlaps.
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
                raw.write(audio); raw.flush()

                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as wav:
                    try:
                        subprocess.run(
                            ["ffmpeg","-y","-i",raw.name,"-af","silenceremove=1:0:-40dB","-ar","16000","-ac","1",wav.name],
                            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True
                        )
                    except:
                        continue

                    # Back to condition_on_previous_text=True for better stability.
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
                    print("📝 ASR:", src_text)

                    # Quietly drop trivial thank-you spam (keeps captions clean).
                    if any(x in src_text.lower() for x in ["thank you","thanks"]) or "ありがとう" in src_text:
                        print("🚫 Skipped a thank-you phrase.")
                        continue

                    # Drop obvious CTA-style fluff.
                    try:
                        if await hallucination_check(src_text):
                            print("🧠 Dropped CTA/meta filler.")
                            continue
                    except Exception as e:
                        print("⚠️ CTA check failed, passing segment:", e)

                    # (1) Strip overlap against previous ASR tail.
                    delta_src = strip_overlap(prev_src_tail, src_text)
                    if not delta_src:
                        print("🔁 Entire chunk was overlap; skipped.")
                        continue

                    # Update the tail (keep last ~30 words worth of context).
                    joined = (prev_src_tail + " " + src_text).strip()
                    tail_words = _simple_words(joined)[-30:]
                    # Rebuild a tail string approximately from the last words (OK for overlap use)
                    prev_src_tail = " ".join(tail_words)

                    # Translate only the *new* bit.
                    segment_id = str(uuid.uuid4())
                    transcript_history.append((segment_id, delta_src))
                    if len(transcript_history) > MAX_TRANSCRIPTS:
                        transcript_history.pop(0)

                    translated = await translate_text(delta_src, source_lang, target_lang)

                    # (2) In-line de-dupe inside the translated string.
                    translated = dedupe_repeated_ngrams(translated, n=3)

                    # (3) Cross-line near-duplicate guard.
                    if looks_like_recent_duplicate(translated, recent_targets):
                        print("🔁 Dropped near-duplicate line.")
                    else:
                        await websocket.send_text(f"[DONE]{json.dumps({'id': segment_id, 'text': translated})}")
                        recent_targets.append(translated)
                        if len(recent_targets) > MAX_RECENT:
                            recent_targets.pop(0)

                    # Gentle refinement of the previous line with new context.
                    if len(transcript_history) >= 2:
                        prev, curr = transcript_history[-2][1], transcript_history[-1][1]
                        improved = await translate_text((prev, curr), source_lang, target_lang, mode="context")
                        # Keep updates minimal; no extra de-dupe to avoid flicker.
                        await websocket.send_text(f"[UPDATE]{json.dumps({'id': transcript_history[-2][0], 'text': improved})}")

    except Exception as e:
        print("❌ WebSocket error:", e)
        await websocket.close()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
