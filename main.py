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

def dedupe_repeated_ngrams(text: str, n: int = 3, min_run_chars: int = 4) -> str:
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
                                ratio_threshold: float = 0.80,
                                contain_threshold: float = 0.80) -> bool:
    norm_new = _normalize_for_compare(new_text)
    if not norm_new:
        return False
    for old in reversed(history):
        norm_old = _normalize_for_compare(old)
        if not norm_old:
            continue
        short, long_ = (norm_new, norm_old) if len(norm_new) <= len(norm_old) else (norm_old, norm_new)
        if len(short) >= 4 and short in long_ and len(short)/len(long_) >= contain_threshold:
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
    # Recent target memory 
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
Return ONE natural {target_lang} line for <previous>. Use <current> only to clarify or complete it.
</goal>

<completion_policy>
- Complete ONLY if <current> clearly finishes <previous>. If not obvious, keep <previous> as a natural fragment.
- Preserve mood/person from source. Do NOT turn 1st-person statements into imperatives.
- Remove exact or near-duplicate wording that appears in both <previous> and <current> (ASR overlap). Keep the clearest version once in the best position.
- Compress clause restarts and self-corrections without changing meaning.
- Keep stance words (actually, maybe) but drop pure fillers (uh/um/えっと) with no meaning.
- If two adjacent clauses/sentences assert the same fact (e.g., “My name is Alexis.” repeated),
  keep it once.
</completion_policy>

<priorities>
1) Faithful meaning > natural flow > brevity.
2) Prefer common collocations over source word order.
3) Do NOT repeat the same phrase, clause, or object more than once unless the source clearly intends emphasis.
4) Avoid repeating lines already in <recent_target>.
5) If <previous> is a heading/label/title, preserve that non-sentence style (don’t narrativize).
6) Preserve fragment type: heading stays heading; spoken clause becomes smooth spoken language.
</priorities>

<language_tips>
- Japanese: everyday phrasing; smooth particles; avoid calques; naturalize set phrases.
- English: use contractions and common phrasing; avoid source-like punctuation/order when awkward.
</language_tips>

<forbidden>
- Adding information not implied by <previous> or <current>.
- Explanations, notes, quotes, brackets. Output text only.
</forbidden>

<positive_examples>
<previous>He said the problem was</previous>
<current>the problem was caused by a software bug.</current>
<output>He said the problem was caused by a software bug.</output>

<previous>My name is</previous>
<current>name is Alexis.</current>
<output>My name is Alexis.</output>

<previous>Uh, I think we should</previous>
<current>we should probably call them now.</current>
<output>I think we should probably call them now.</output>

<previous>They arrived late because</previous>
<current>because the train was delayed.</current>
<output>They arrived late because the train was delayed.</output>

<previous>Meeting Agenda</previous>
<current>for Thursday</current>
<output>Meeting Agenda for Thursday</output>

<previous>The minister said that</previous>
<current>the minister said that this policy would be reviewed.</current>
<output>The minister said that this policy would be reviewed.</output>

<previous>He plans to visit Paris</previous>
<current>visit Paris next summer.</current>
<output>He plans to visit Paris next summer.</output>

<previous>We need to buy more supplies</previous>
<current>more supplies for the trip.</current>
<output>We need to buy more supplies for the trip.</output>

<previous>She told me</previous>
<current>she told me to be careful.</current>
<output>She told me to be careful.</output>
</positive_examples>

<negative_examples>
<previous>He said the problem was</previous>
<current>the problem was caused by a software bug.</current>
<output>He said the problem was the problem was caused by a software bug.</output>  <!-- WRONG -->

<previous>My name is</previous>
<current>name is Alexis.</current>
<output>My name is Alexis. My name is Alexis.</output>  <!-- WRONG -->

<previous>Uh, I think we should</previous>
<current>we should probably call them now.</current>
<output>Uh, I think we should probably call them now.</output>  <!-- WRONG -->

<previous>They arrived late because</previous>
<current>because the train was delayed.</current>
<output>They arrived late because because the train was delayed.</output>  <!-- WRONG -->

<previous>Meeting Agenda</previous>
<current>for Thursday</current>
<output>This is the meeting agenda for Thursday.</output>  <!-- WRONG -->

<previous>The minister said that</previous>
<current>the minister said that this policy would be reviewed.</current>
<output>The minister said that. The minister said that this policy would be reviewed.</output>  <!-- WRONG -->

<previous>He plans to visit Paris</previous>
<current>visit Paris next summer.</current>
<output>He plans to visit Paris visit Paris next summer.</output>  <!-- WRONG -->

<previous>We need to buy more supplies</previous>
<current>more supplies for the trip.</current>
<output>We need to buy more supplies more supplies for the trip.</output>  <!-- WRONG -->

<previous>She told me</previous>
<current>she told me to be careful.</current>
<output>She told me she told me to be careful.</output>  <!-- WRONG -->
</negative_examples>


<merge_test>
Merge only if replacing <previous> with the result does not lose meaning and does not introduce repeated clauses.
If unsure, do not merge.
</merge_test>

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
Produce fluent, idiomatic {target_lang} captions for this single ASR segment.
</goal>

<priorities>
1) Preserve meaning faithfully; do not invent content.
2) Prefer natural phrasing over literal order when safe.
3) Mirror completeness: if input is a fragment, output a natural fragment.
4) Keep numbers as digits; preserve names and units verbatim.
5) Remove pure fillers (uh/um/えっと) unless they convey hesitation/tone.
6) If a phrase repeats with no new info — including exact or near-duplicate wording from ASR overlap/restarts — keep it only once in its clearest form.
7) If the input is already {target_lang}, return it unchanged.
8) If the input is a label/title/heading/meta comment, translate it as such without turning it into a full sentence.
9) Preserve mood/person; do NOT convert 1st-person statements into imperatives.
</priorities>

<language_tips>
- Japanese: prefer everyday collocations; use 「〜できる」 over stiff 「〜することができる」; drop unnecessary subjects; pick particles for smoothness; avoid calques.
- English: use contractions and common collocations; avoid source-language punctuation/order when awkward.
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
                        
                        # --- Context update duplication guard ---
                        original_translation = next((t for tid, t in zip(
                            [x[0] for x in transcript_history], recent_targets) 
                            if tid == transcript_history[-2][0]), None)
                        
                        if original_translation and looks_like_recent_duplicate(improved, [original_translation]):
                            print("Skipped context update: near-duplicate of existing output.")
                        else:
                            await websocket.send_text(f"[UPDATE]{json.dumps({'id': transcript_history[-2][0], 'text': improved})}")


    except Exception as e:
        print("WebSocket error:", e)
        await websocket.close()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
