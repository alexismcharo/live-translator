import tempfile, subprocess, uvicorn, openai, whisper, os, json, uuid, re, difflib
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from dotenv import load_dotenv
from difflib import SequenceMatcher

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = openai.AsyncOpenAI(api_key=api_key)

# 🔧 Pre-warm ffmpeg & audio stack so the first request isn't sluggish.
try:
    subprocess.run(
        [
            "ffmpeg", "-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono",
            "-t", "0.5", "-ar", "16000", "-ac", "1",
            "-y", "/tmp/warm.wav"
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
except:
    pass

# 🎧 Whisper: large-v3 is great quality. We'll tweak runtime settings below for live use.
model = whisper.load_model("large-v3")

# Warm the model with a tiny file to avoid first-translation hiccups.
try:
    model.transcribe("/tmp/warm.wav", language="en", fp16=True)
except:
    pass

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

# 🧠 Light histories to help with repetition control.
transcript_history = []  # [(segment_id, original_text)]
recent_targets = []      # just the latest translated outputs we showed the user
MAX_RECENT = 15          # a bit longer memory cuts down on deja-vu lines
MAX_TRANSCRIPTS = 200

# --------------------- Fuzzy/partial repetition utilities ---------------------

def _normalize_text_for_compare(s: str) -> str:
    """
    Keep it simple: lowercase, collapse spaces, and keep common symbols/units.
    This helps us compare apples-to-apples across tiny formatting differences.
    """
    if not s:
        return ""
    s = s.lower()
    s = s.replace("kg.", "kg").replace(" kg", "kg")
    # Keep alnum + currency/percent/units-ish stuff; strip the rest.
    s = re.sub(r"[^a-z0-9£$€¥%\-\. ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _tokenize_simple(s: str) -> list[str]:
    # Hyphens/periods can split words in captions; treat them as spaces for overlap checks.
    s = re.sub(r"[\-\.]", " ", s)
    toks = [t for t in s.split() if t]
    return toks

def _ngrams(tokens: list[str], n: int = 6) -> set[tuple[str, ...]]:
    """
    Turn a sentence into rolling n-grams. Using 5–8 tends to work best for live ASR;
    6 is a nice middle ground.
    """
    if len(tokens) < n:
        return {tuple(tokens)} if tokens else set()
    return {tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)}

def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0

def _is_fuzzy_duplicate(a: str, b: str,
                        jaccard_threshold: float = 0.75,
                        ratio_threshold: float = 0.80,
                        contain_threshold: float = 0.80,
                        ngram_n: int = 6,
                        ngram_threshold: float = 0.70) -> bool:
    """
    Decide if two sentences/clauses are "basically the same".
    We check:
      - containment: short is mostly inside long
      - character-level similarity (SequenceMatcher)
      - n-gram overlap (catches mid-sentence repeats)
      - token-set Jaccard as a coarse backup
    """
    na, nb = _normalize_text_for_compare(a), _normalize_text_for_compare(b)
    if not na or not nb:
        return False

    # Containment (lets us catch "included baggage... included" style tails)
    short, long_ = (na, nb) if len(na) <= len(nb) else (nb, na)
    if len(short) >= 6 and short in long_:
        if len(short) / len(long_) >= contain_threshold:
            return True

    # Character-level similarity
    ratio = SequenceMatcher(None, na, nb).ratio()
    if ratio >= ratio_threshold:
        return True

    # N-gram overlap (robust to punctuation/small edits)
    ta, tb = _tokenize_simple(na), _tokenize_simple(nb)
    ga, gb = _ngrams(ta, n=ngram_n), _ngrams(tb, n=ngram_n)
    if ga and gb and _jaccard(ga, gb) >= ngram_threshold:
        return True

    # Token-set Jaccard as a gentle last check
    ja, jb = set(ta), set(tb)
    if _jaccard(ja, jb) >= jaccard_threshold:
        return True

    return False

def collapse_repetition(text: str, *, fuzzy: bool = True) -> str:
    """
    Squash obvious repeats from a single piece of text.
    - If fuzzy=True, we aggressively remove near-duplicates (better for [DONE]).
    - If fuzzy=False, we only remove exact duplicates (safer for [UPDATE]).
    This function also catches *mid-sentence* repetition via n-grams.
    """
    if not text:
        return text

    # Split on sentence-ish boundaries but also keep clauses; live ASR is messy.
    parts = re.split(
        r'(?<=[.!?！？。…])\s+|[\u2014\u2013\-]{1,2}\s+|…\s*|[、，,;；]\s*',
        text
    )
    parts = [p.strip() for p in parts if p and p.strip()]

    kept = []
    seen_exact = set()

    for p in parts:
        if not p:
            continue

        if not fuzzy:
            # Exact-only mode for minimal jumpiness in [UPDATE]
            key = p.strip().lower()
            if key in seen_exact:
                continue
            seen_exact.add(key)
            kept.append(p)
            continue

        # Fuzzy mode: drop if "basically the same" as anything we've kept so far
        is_dup = any(_is_fuzzy_duplicate(p, prev) for prev in kept)
        if not is_dup:
            kept.append(p)

    cleaned = " ".join(kept).strip()
    return cleaned

def is_partial_duplicate_against_history(new_text: str, history: list[str],
                                         ratio_threshold: float = 0.80,
                                         contain_threshold: float = 0.80,
                                         ngram_n: int = 6,
                                         ngram_threshold: float = 0.70) -> bool:
    """
    Final gate before we send a line to the user:
    if this new line is mostly the same as any of the recent lines we've already shown,
    we just don't send it.
    """
    norm_new = _normalize_text_for_compare(new_text)
    if not norm_new:
        return False

    for old in reversed(history):
        norm_old = _normalize_text_for_compare(old)
        if not norm_old:
            continue

        # Quick containment
        short, long_ = (norm_new, norm_old) if len(norm_new) <= len(norm_old) else (norm_old, norm_new)
        if len(short) >= 6 and short in long_:
            if len(short) / len(long_) >= contain_threshold:
                return True

        # Char-level similarity
        if SequenceMatcher(None, norm_new, norm_old).ratio() >= ratio_threshold:
            return True

        # N-gram overlap
        ta, tb = _tokenize_simple(norm_new), _tokenize_simple(norm_old)
        ga, gb = _ngrams(ta, n=ngram_n), _ngrams(tb, n=ngram_n)
        if ga and gb and _jaccard(ga, gb) >= ngram_threshold:
            return True

    return False

# ----------------------------------------------------------------------

@app.get("/")
async def serve_index():
    return FileResponse(os.path.join("frontend", "index.html"))

# 🛑 Lightweight hallucination/CTA filter.
async def hallucination_check(text: str) -> bool:
    """
    Returns True if the segment looks like channel fluff (subscribe/like/sign-off),
    which we don't want in live captions. Normal conversational stuff passes through.
    """
    try:
        seg = (text or "").strip()
        if not seg:
            return False

        system = (
            "You are a binary classifier for a LIVE ASR → translation pipeline. "
            "Return exactly one token: YES or NO."
        )
        user = f"""
We are processing short, possibly incomplete ASR segments in real time.
Decide if the segment is *broadcast-style filler* (audience address, subscribe/like calls, end-of-video sign-off) that should be dropped from live translation.

Guidelines:
- Return YES only if the segment is clearly a CTA/sign-off or meta-address to viewers.
- Return NO for normal conversational content, even if brief or disfluent.

Segment:
<segment>{seg}</segment>

Answer with exactly YES or NO.
""".strip()

        result = await client.chat.completions.create(
            model="gpt-5",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ],
            reasoning_effort="minimal",
            verbosity="low"
        )
        out = (result.choices[0].message.content or "").strip().upper()
        return out == "YES"
    except Exception as e:
        print("❌ Hallucination check error:", e)
        return False

# 🎙️ Main translation function: natural, live-caption style output.
async def translate_text(text, source_lang, target_lang, mode="default"):
    """
    Keep translations idiomatic and avoid guessing endings.
    We also pass in some recent outputs so the model doesn't rehash them.
    """
    target_register = "polite" if target_lang == "Japanese" else "neutral"
    recent_target_str = "\n".join(recent_targets[-MAX_RECENT:])

    if mode == "context":
        previous, current = text
        system = (
            "You are a live simultaneous interpreter refining captions. "
            "Revise ONLY the translation of <previous> using <current> for context."
        )
        user = f"""
<task>
Produce a natural, idiomatic {target_lang} caption for <previous>, updating it only if <current> clarifies meaning.
</task>

<recent_target>
{recent_target_str}
</recent_target>

<rules>
- Output: ONLY the improved translation of <previous>. No quotes, no commentary.
- Make phrasing natural in {target_lang}; add no new information.
- If <previous> was a fragment, keep it a natural fragment; don't invent endings.
- Resolve details only if <current> makes them clear.
- Remove pure filler (uh/um/えっと/あの).
- Keep numbers as digits; preserve names/units.
- Avoid repeating sentences or phrases already present in <recent_target> unless they add new factual content.
- Keep edits minimal to avoid visual jumpiness.
- Register: for Japanese use {"です・ます" if target_register=="polite" else "casual speech"}; for English use {target_register} spoken style.
</rules>

<previous>{previous}</previous>
<current>{current}</current>
""".strip()
    else:
        system = (
            "You are a live, natural translator for streaming ASR. "
            "Return ONLY the translation text—no quotes or extra words."
        )
        user = f"""
<task>
Translate a short, possibly incomplete ASR segment from {source_lang} to {target_lang} for live captions. Aim for natural, idiomatic speech.
</task>

<recent_target>
{recent_target_str}
</recent_target>

<rules>
- Natural over literal; keep meaning faithful.
- Mirror completeness: if the source is a fragment, keep a natural fragment.
- Remove pure filled pauses unless meaningful.
- Keep numbers as digits; preserve names, technical terms, and units.
- No extra greetings/sign-offs/CTAs.
- Avoid repeating phrases already in <recent_target>; merge repeats into one clean line.
- If input is already in {target_lang}, return it unchanged.
- Register: for Japanese use {"です・ます" if target_register=="polite" else "casual speech"}; for English use {target_register} spoken style.
</rules>

<input>{text}</input>
""".strip()

    try:
        response = await client.chat.completions.create(
            model="gpt-5",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ],
            reasoning_effort="minimal",
            verbosity="low"
        )
        raw = (response.choices[0].message.content or "").strip()
        # For final lines we de-dupe fuzzily; for updates we stick to exact-only to avoid flicker.
        if mode == "context":
            return collapse_repetition(raw, fuzzy=False)
        else:
            return collapse_repetition(raw, fuzzy=True)
    except Exception as e:
        print("❌ Translation error:", e)
        return text

# 🧵 WebSocket for streaming translation
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("🔌 WebSocket connected")

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

            # Save raw audio chunk and convert to 16kHz mono WAV with mild silence trimming.
            with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as raw:
                raw.write(audio)
                raw.flush()

                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as wav:
                    try:
                        subprocess.run(
                            [
                                "ffmpeg", "-y",
                                "-i", raw.name,
                                "-af", "silenceremove=1:0:-40dB",
                                "-ar", "16000",
                                "-ac", "1",
                                wav.name
                            ],
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.PIPE,
                            check=True
                        )
                    except:
                        continue

                    # 🎯 Whisper tuned for live streaming:
                    # - condition_on_previous_text=False reduces "echoed" repeats.
                    # - slightly higher hallucination_silence_threshold trims rambly tails.
                    result = model.transcribe(
                        wav.name,
                        fp16=True,
                        temperature=0.0,  # greedy is fine for live; keeps latency down
                        condition_on_previous_text=False,
                        hallucination_silence_threshold=0.3,
                        no_speech_threshold=0.3,
                        language="en" if source_lang == "English" else "ja",
                        compression_ratio_threshold=2.4,
                        logprob_threshold=-1.0
                    )

                    text = (result.get("text") or "").strip()
                    print("📝 Transcribed:", text)
                    if not text:
                        continue

                    # 🙏 We quietly ignore common "thanks" so they don't spam the captions.
                    text_lower = text.lower()
                    if (
                        "thank you" in text_lower
                        or "thanks" in text_lower
                        or "ありがとう" in text
                        or "ありがとうございます" in text
                        or "ありがと" in text
                    ):
                        print("🚫 Skipping thank-you/ありがとう phrase:", text)
                        continue

                    # 🧹 CTA/broadcast style filler filter (best-effort; fails open).
                    try:
                        if await hallucination_check(text):
                            print("🧠 GPT flagged as broadcast/CTA filler:", text)
                            continue
                    except Exception as e:
                        print("⚠️ Hallucination check failed open (passing segment):", e)

                    # Assign an ID and translate.
                    segment_id = str(uuid.uuid4())
                    transcript_history.append((segment_id, text))
                    if len(transcript_history) > MAX_TRANSCRIPTS:
                        transcript_history.pop(0)

                    translated = await translate_text(text, source_lang, target_lang)

                    # 🚧 Final gate: if this looks like something we *just* showed the user, drop it.
                    if is_partial_duplicate_against_history(translated, recent_targets):
                        print("🔁 Dropped near-duplicate translation:", translated)
                    else:
                        await websocket.send_text(f"[DONE]{json.dumps({'id': segment_id, 'text': translated})}")
                        recent_targets.append(translated)
                        if len(recent_targets) > MAX_RECENT:
                            recent_targets.pop(0)

                    # 🔄 Try a gentle refinement of the *previous* line with the new context.
                    if len(transcript_history) >= 2:
                        prev, curr = transcript_history[-2][1], transcript_history[-1][1]
                        improved = await translate_text((prev, curr), source_lang, target_lang, mode="context")
                        # In update mode we already do exact-only dedupe to keep the text stable.
                        await websocket.send_text(f"[UPDATE]{json.dumps({'id': transcript_history[-2][0], 'text': improved})}")

    except Exception as e:
        print("❌ WebSocket error:", e)
        await websocket.close()

if __name__ == "__main__":
    # Keep it simple: bind to all interfaces; tweak the port as you like.
    uvicorn.run(app, host="0.0.0.0", port=8000)
