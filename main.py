import tempfile, subprocess, uvicorn, openai, whisper, os, json, uuid, re, difflib
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = openai.AsyncOpenAI(api_key=api_key)

# Pre-warm ffmpeg & audio stack
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

# Optimized for T4 GPU (if available)
model = whisper.load_model("large-v3")

# Warm Whisper model once to reduce first-latency (fix path)
try:
    model.transcribe("/tmp/warm.wav", language="en", fp16=True)
except:
    pass

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

# Tunables
MAX_RECENT = 5
MAX_TRANSCRIPTS = 200

# --------------------- Fuzzy repetition utilities ---------------------

def _normalize_text_for_compare(s: str) -> str:
    """Lowercase, normalize spacing/punctuation; keep alnum + a few symbols."""
    if not s:
        return ""
    s = s.lower()
    s = s.replace("kg.", "kg").replace(" kg", "kg")
    s = re.sub(r"[^a-z0-9£$€¥%\-\. ]+", " ", s)  # keep common currency/units
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _token_set(s: str) -> set:
    s = re.sub(r"[\-\.]", " ", s)
    toks = [t for t in s.split() if t]
    return set(toks)

def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0

def _is_fuzzy_duplicate(a: str, b: str,
                        jaccard_threshold: float = 0.82,
                        ratio_threshold: float = 0.86,
                        contain_threshold: float = 0.92) -> bool:
    """
    Decide if two sentences are near-duplicates using:
      - Jaccard overlap on token sets
      - difflib.SequenceMatcher ratio
      - containment (short inside long)
    """
    na, nb = _normalize_text_for_compare(a), _normalize_text_for_compare(b)
    if not na or not nb:
        return False

    short, long_ = (na, nb) if len(na) <= len(nb) else (nb, na)
    if len(short) >= 6 and short in long_:
        if len(short) / len(long_) >= contain_threshold:
            return True

    ja = _token_set(na)
    jb = _token_set(nb)
    jacc = _jaccard(ja, jb)
    ratio = difflib.SequenceMatcher(None, na, nb).ratio()

    return (jacc >= jaccard_threshold) or (ratio >= ratio_threshold)

def collapse_repetition(text: str, *, fuzzy: bool = True) -> str:
    """
    Remove exact or near-exact repeated sentences/clauses.
    - fuzzy=True: use near-duplicate checks (for initial lines).
    - fuzzy=False: only drop exact duplicates (for minimal jumpiness).
    """
    if not text:
        return text

    # Split by sentence boundaries and strong separators
    parts = re.split(r'(?<=[.!?！？。…])\s+|[\u2014\u2013\-]{1,2}\s+|…\s*', text)
    parts = [p.strip() for p in parts if p and p.strip()]

    kept = []
    seen_exact = set()
    for p in parts:
        if not p:
            continue
        if not fuzzy:
            key = p.strip().lower()
            if key in seen_exact:
                continue
            seen_exact.add(key)
            kept.append(p)
        else:
            is_dup = any(_is_fuzzy_duplicate(p, prev) for prev in kept)
            if not is_dup:
                kept.append(p)

    cleaned = " ".join(kept)
    return cleaned.strip()

# --------------------- Source-side de-stutter ---------------------

def dedupe_source_fragments(s: str) -> str:
    """Collapse trivial repeats from ASR like '...含まれます 手荷物が 含まれています'."""
    if not s:
        return s
    # collapse immediate token repeats
    tokens = re.split(r'(\s+)', s.strip())
    out, prev = [], None
    for t in tokens:
        key = t.strip()
        if key and prev and key == prev:
            continue
        out.append(t)
        if key:
            prev = key
    s = "".join(out)
    # remove trivially duplicated bigrams: "... baggage included baggage included"
    s = re.sub(r'\b(\S+\s+\S+)\s+\1\b', r'\1', s, flags=re.IGNORECASE)
    return s

# --------------------- Emission controller ---------------------

def classify_relation(prev: str, curr: str):
    """Return 'drop' | 'update' | 'new' based on similarity/containment."""
    pa, ca = _normalize_text_for_compare(prev), _normalize_text_for_compare(curr)
    if not ca:
        return "drop"
    r = difflib.SequenceMatcher(None, pa, ca).ratio()
    if r >= 0.96:
        return "drop"  # near-identical to last
    # clear extension
    if pa and ca.startswith(pa) and len(ca) > len(pa) + 3:
        return "update"
    # substantial overlap → treat as update if largely a superset
    if r >= 0.85 and len(ca) >= len(pa) and pa in ca:
        return "update"
    return "new"

# ----------------------------------------------------------------------

@app.get("/")
async def serve_index():
    return FileResponse(os.path.join("frontend", "index.html"))

# --- Live-aware hallucination filter (keeps normal chatter; blocks broadcast CTAs) ---
async def hallucination_check(text: str) -> bool:
    """
    Returns True if the segment should be dropped as broadcast-style filler
    (subscribe/like/sign-off/viewer address), False otherwise.
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
- Return NO for normal conversational content, even if brief (e.g., 'thanks', 'sorry', 'okay') or disfluent (um/uh/えっと).
- If the segment looks incomplete or mid-utterance, default to NO unless it already contains a clear CTA cue.

Positive examples (YES):
- Thanks for watching!
- Don't forget to subscribe.
- Click the bell icon.
- See you in the next video.
- Link in the description.
- 皆さんこんにちは (as a YouTuber-style opener addressing viewers)
- チャンネル登録お願いします / 高評価お願いします / ベルマーク通知をオンに

Negative examples (NO):
- Thank you. (as a normal conversational turn)
- Sorry about that.
- Okay, next step.
- ありがとうございます。 / すみません。 / はい、次に進みます。
- (partial) "Click the..." — unless it’s clearly a CTA (insufficient on its own → NO)

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
            verbosity="low",
        )
        out = (result.choices[0].message.content or "").strip().upper()
        return out == "YES"
    except Exception as e:
        print("❌ Hallucination check error:", e)
        return False

# --- Main translation function (natural, live-caption style) ---
async def translate_text(text, source_lang, target_lang, *, recent_targets, mode="default"):
    """
    Natural, idiomatic live captions; mirrors fragment completeness; minimal edits on refinements.
    May return an empty string if there's no new information compared to recent targets.
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
If this largely repeats earlier content in <recent_target>, output only the new information; if there's nothing new, output nothing.
</task>

<recent_target>
{recent_target_str}
</recent_target>

<rules>
- Output: ONLY the improved translation of <previous>. No quotes, no commentary; empty output is allowed if nothing new.
- Make phrasing natural in {target_lang} (not literal), but add no new information.
- If <previous> was a fragment, keep it a natural fragment; don't invent endings.
- Resolve pronouns, names, tense, or ellipsis only if <current> makes them clear.
- Remove filler like uh/um/えっと/あの unless meaningful.
- Keep numbers as digits and preserve proper nouns/terminology.
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
Translate a short, possibly incomplete ASR segment from {source_lang} to {target_lang} for live captions.
Aim for natural, idiomatic speech. If this segment largely repeats earlier translated content in <recent_target>, output only the new information; if there's nothing new, output nothing.
</task>

<recent_target>
{recent_target_str}
</recent_target>

<rules>
- Natural over literal: use target-language word order and phrasing; keep meaning faithful.
- Mirror completeness: if the source is a fragment, keep a natural fragment; do not guess the rest.
- Remove pure filled pauses (uh/um/えっと/あの) unless they carry meaning.
- Keep numbers as digits; preserve names, technical terms, and units.
- Do not add greetings/sign-offs/explanations or CTAs unless in source.
- Avoid repeating sentences or phrases already present in <recent_target> unless they add new factual content.
- If repetition occurs, merge into one concise, natural sentence. It is okay to return an empty string if nothing new.
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
            verbosity="low",
        )
        raw = (response.choices[0].message.content or "").strip()
        if mode == "context":
            return collapse_repetition(raw, fuzzy=False)  # gentle on updates
        else:
            return collapse_repetition(raw, fuzzy=True)
    except Exception as e:
        print("❌ Translation error:", e)
        return ""

# --- WebSocket for streaming translation ---
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("🔌 WebSocket connected")

    # Per-connection state (prevents cross-talk)
    transcript_history = []   # [(segment_id, original_text)]
    recent_targets = []       # last few translated outputs only
    last_emitted_id = None
    last_emitted_text = ""
    processing = False

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
            if processing:
                # simple backpressure—drop this chunk if one is in flight
                continue
            processing = True

            raw_path = wav_path = None
            try:
                audio = msg["bytes"]
                if not audio:
                    continue

                with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as raw:
                    raw.write(audio)
                    raw.flush()
                    raw_path = raw.name

                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as wav:
                    wav_path = wav.name
                    try:
                        subprocess.run(
                            [
                                "ffmpeg", "-y",
                                "-i", raw_path,
                                "-af", "silenceremove=1:0:-35dB",
                                "-ar", "16000",
                                "-ac", "1",
                                wav_path
                            ],
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.PIPE,
                            check=True
                        )
                    except:
                        continue

                    result = model.transcribe(
                        wav_path,
                        fp16=True,
                        temperature=0.0,
                        # beam_size omitted (greedy implied with temperature=0.0)
                        condition_on_previous_text=True,
                        hallucination_silence_threshold=0.2,
                        no_speech_threshold=0.3,
                        language="en" if source_lang == "English" else "ja",
                        compression_ratio_threshold=2.4,
                        logprob_threshold=-1.0
                    )

                text = dedupe_source_fragments((result.get("text") or "").strip())
                print("📝 Transcribed:", text)
                if not text:
                    continue

                # ---------- Keep thank-you filter ----------
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
                # ------------------------------------------

                # GPT-based hallucination filter
                try:
                    if await hallucination_check(text):
                        print("🧠 GPT flagged as broadcast/CTA filler:", text)
                        continue
                except Exception as e:
                    print("⚠️ Hallucination check failed open (passing segment):", e)

                # Translate
                translated = await translate_text(
                    text, source_lang, target_lang, recent_targets=recent_targets, mode="default"
                )

                # Allow empty output when nothing new
                if not translated:
                    continue

                # Decide how to emit relative to last
                action = classify_relation(last_emitted_text, translated)

                if action == "drop":
                    continue

                elif action == "update" and last_emitted_id is not None:
                    # refine the existing caption instead of adding a new one
                    await websocket.send_text(f"[UPDATE]{json.dumps({'id': last_emitted_id, 'text': translated})}")
                    # update last state and replace last recent target
                    if recent_targets:
                        recent_targets[-1] = translated
                    else:
                        recent_targets.append(translated)
                    last_emitted_text = translated

                else:  # "new"
                    segment_id = str(uuid.uuid4())
                    transcript_history.append((segment_id, text))
                    if len(transcript_history) > MAX_TRANSCRIPTS:
                        transcript_history.pop(0)

                    await websocket.send_text(f"[DONE]{json.dumps({'id': segment_id, 'text': translated})}")
                    # record as last emitted
                    last_emitted_id = segment_id
                    last_emitted_text = translated
                    # push to recent_targets
                    recent_targets.append(translated)
                    if len(recent_targets) > MAX_RECENT:
                        recent_targets.pop(0)

            except Exception as e:
                print("❌ WebSocket loop error:", e)
            finally:
                # cleanup temps and release processing gate
                for p in (raw_path, wav_path):
                    if p and os.path.exists(p):
                        try:
                            os.unlink(p)
                        except:
                            pass
                processing = False

    except Exception as e:
        print("❌ WebSocket error:", e)
        await websocket.close()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
