import os
import json
import uuid
import time
import tempfile
import subprocess
from collections import deque

import uvicorn
import whisper
import openai
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from dotenv import load_dotenv

# ──────────────────────────────────────────────────────────────────────────────
# Setup
# ──────────────────────────────────────────────────────────────────────────────
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY not set")
client = openai.AsyncOpenAI(api_key=api_key)

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def safe_output_text(resp) -> str:
    """
    Robustly extract text from a Responses API object.
    Works with both final non-stream responses and stream finalization objects.
    """
    if not resp:
        return ""
    txt = getattr(resp, "output_text", None)
    if isinstance(txt, str) and txt.strip():
        return txt.strip()

    out = []
    output = getattr(resp, "output", []) or []
    for item in output:
        it_text = getattr(item, "text", None)
        if isinstance(it_text, str) and it_text:
            out.append(it_text)
        content = getattr(item, "content", []) or []
        for c in content:
            t = getattr(c, "text", None) or getattr(c, "input_text", None)
            if isinstance(t, str) and t:
                out.append(t)
    if out:
        return "".join(out).strip()

    for attr in ("message", "content", "text"):
        val = getattr(resp, attr, None)
        if isinstance(val, str) and val.strip():
            return val.strip()

    return ""

def looks_japanese(s: str) -> bool:
    return any(0x3040 <= ord(ch) <= 0x30FF or 0x4E00 <= ord(ch) <= 0x9FFF for ch in s)

def looks_english(s: str) -> bool:
    return any(("A" <= ch <= "Z") or ("a" <= ch <= "z") for ch in s)

def violates_target_lang(output: str, target_lang: str) -> bool:
    out = (output or "").strip()
    if not out:
        return True
    if target_lang == "Japanese":
        return not looks_japanese(out)
    if target_lang == "English":
        return not looks_english(out)
    return False

# Trivial fragment filter (to drop lone "you", "um", 「えっと」, etc.)
PRONOUNS_EN = {"you","i","me","we","they","he","she","it"}
FILLER_EN   = {"uh","um","er","ah","oh","hmm","huh","uh-huh","nah","yep","nope","like"}
PRONOUNS_JA = {"あなた","私","僕","俺","我々","彼","彼女"}
FILLER_JA   = {"えっと","あの","うーん","えーと","まぁ","その"}

def should_skip_fragment(text: str, source_lang: str) -> bool:
    t = (text or "").strip()
    if not t:
        return True
    if source_lang == "English":
        import re
        words = re.findall(r"[A-Za-z]+", t.lower())
        if len(words) == 1 and (words[0] in PRONOUNS_EN or words[0] in FILLER_EN):
            return True
        return False
    if source_lang == "Japanese":
        if t in FILLER_JA or t in PRONOUNS_JA:
            return True
        # very short kana-only interjections
        if len(t) <= 2 and any("ぁ" <= ch <= "ん" or "ァ" <= ch <= "ン" for ch in t):
            return True
        return False
    return False

# ──────────────────────────────────────────────────────────────────────────────
# Warm FFmpeg & Whisper
# ──────────────────────────────────────────────────────────────────────────────
WARMUP_WAV = "/tmp/warm.wav"
try:
    subprocess.run(
        ["ffmpeg", "-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono", "-t", "0.5",
         "-ar", "16000", "-ac", "1", "-y", WARMUP_WAV],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True
    )
except Exception:
    pass

model = whisper.load_model("large-v3")
try:
    model.transcribe(WARMUP_WAV, language="en", fp16=True)
except Exception:
    pass

# ──────────────────────────────────────────────────────────────────────────────
# FastAPI
# ──────────────────────────────────────────────────────────────────────────────
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# Store original (source) text for context refinement: (segment_id, original_text)
transcript_history = deque(maxlen=500)

# ──────────────────────────────────────────────────────────────────────────────
# Hallucination / Filler filter  (IMPROVED)
# ──────────────────────────────────────────────────────────────────────────────
_HALLU_PHRASES = [
    # english CTA / video tropes
    "welcome to my channel",
    "thanks for watching",
    "don't forget to subscribe",
    "like and subscribe",
    "click the bell",
    "smash that like button",
    "see you in the next video",
    "in this video",
    "today i will show you",
    "i'm going to show you",
    "hope you enjoy watching",
    # jp equivalents (rough/common)
    "チャンネル登録", "高評価よろしく", "この動画では", "本日はご紹介します", "最後までご覧ください",
]

def _excessive_repetition(text: str) -> bool:
    """Detect obvious loops like 'I hope you will enjoy it.' repeated many times."""
    import re
    s = re.sub(r"\s+", " ", (text or "")).strip().lower()
    if not s:
        return False
    # same sentence repeated ≥3 times
    parts = re.split(r"[。！？\.\!\?]+", s)
    parts = [p.strip() for p in parts if p.strip()]
    if not parts:
        return False
    from collections import Counter
    c = Counter(parts)
    if any(v >= 3 and len(k) >= 8 for k, v in c.items()):
        return True
    # repeated 3–5-gram loops
    words = re.findall(r"[a-zA-Z\u3040-\u30FF\u4E00-\u9FFF]+", s)
    for n in (3, 4, 5):
        grams = [" ".join(words[i:i+n]) for i in range(max(0, len(words)-n+1))]
        cc = Counter(grams)
        if any(v >= 3 and len(k) >= 8 for k, v in cc.items()):
            return True
    # very long + contains hallmark phrases
    if len(s) > 300 and any(p in s for p in _HALLU_PHRASES):
        return True
    return False

def _contains_hallu_phrase(text: str) -> bool:
    s = (text or "").lower()
    return any(p in s for p in _HALLU_PHRASES)

async def hallucination_check(text: str) -> bool:
    """
    Returns True if 'text' is generic filler/CTA or shows repetition/looping.
    Heuristic pass first; if inconclusive, ask a tiny model.
    """
    # Heuristics catch obvious junk quickly (no API call)
    if _excessive_repetition(text) or _contains_hallu_phrase(text):
        return True

    # Model pass (YES/NO only). Keep >=16 tokens per GPT-5 Responses rules.
    try:
        prompt = (
            "You are a strict hallucination/filler detector for live captions.\n"
            "Reply YES only if the sentence is a generic intro/outro/CTA (e.g., 'welcome to my channel', "
            "'don't forget to subscribe', 'thanks for watching', 'in this video'), OR if it contains obvious "
            "repetitive/looping filler. Otherwise reply NO.\n\n"
            "Reply ONLY YES or NO.\n\n"
            "Sentence:\n" + (text or "")
        )
        resp = await client.responses.create(
            model="gpt-5-nano",
            input=[
                {"role": "system", "content": "Return ONLY YES or NO."},
                {"role": "user",   "content": prompt},
            ],
            max_output_tokens=32
        )
        content = (safe_output_text(resp) or "").strip().upper()
        if content.startswith("YES"):
            return True
        if content.startswith("NO"):
            return False
        return False
    except Exception as e:
        print("⚠️ hallucination_check error:", e)
        return False

# ──────────────────────────────────────────────────────────────────────────────
# Translation (non-stream + enforced)
# ──────────────────────────────────────────────────────────────────────────────
def translator_system_prompt(source_lang: str, target_lang: str) -> str:
    return (
        f"You are a professional {source_lang}↔{target_lang} translator.\n"
        f"- Output ONLY in {target_lang}. No quotes, no commentary.\n"
        f"- Translate even if the source is a fragment or incomplete.\n"
        f"- Never return an empty string; if uncertain, translate literally."
    )

def strict_user_prompt(text: str, source_lang: str, target_lang: str) -> str:
    return (
        f"Translate from {source_lang} to {target_lang}.\n\n"
        f"Sentence:\n{text}\n\n"
        f"Return ONLY the {target_lang} translation (no extra words)."
    )

async def translate_text(text, source_lang: str, target_lang: str, mode: str = "default") -> str:
    """
    Non-streaming translation. For mode='context', `text` is (previous, current).
    NEVER returns tuples; only strings.
    """
    if mode == "context":
        previous, current = text
        user_prompt = (
            f"Refine the translation of the PREVIOUS sentence using the NEW sentence as context.\n\n"
            f"PREVIOUS ({source_lang}): {previous}\n"
            f"NEW ({source_lang}): {current}\n\n"
            f"Return ONLY the improved translation of PREVIOUS in {target_lang}."
        )
    else:
        user_prompt = strict_user_prompt(text, source_lang, target_lang)

    try:
        resp = await client.responses.create(
            model="gpt-5",
            input=[
                {"role": "system", "content": translator_system_prompt(source_lang, target_lang)},
                {"role": "user",   "content": user_prompt},
            ],
            max_output_tokens=200
        )
        out = safe_output_text(resp)
        if not out:
            print("⚠️ translate_text returned empty")
        if not out and mode == "context":
            previous, _ = text
            return str(previous)
        return out or (str(text) if isinstance(text, str) else "")
    except Exception as e:
        print("❌ translate_text error:", e)
        if mode == "context":
            previous, _ = text
            return str(previous)
        return text if isinstance(text, str) else ""

async def translate_text_enforced(text: str, source_lang: str, target_lang: str) -> str:
    """
    Enforced translation of a single sentence (no context).
    """
    try:
        resp = await client.responses.create(
            model="gpt-5",
            input=[
                {"role": "system", "content": translator_system_prompt(source_lang, target_lang) +
                 "\nCRITICAL: Do NOT echo the source. Output must be in the target language script."},
                {"role": "user",   "content": strict_user_prompt(text, source_lang, target_lang)},
            ],
            max_output_tokens=220
        )
        return safe_output_text(resp) or ""
    except Exception as e:
        print("❌ translate_text_enforced error:", e)
        return ""

async def refine_previous(prev_text: str, curr_text: str,
                          source_lang: str, target_lang: str):
    """
    Returns refined previous translation string, or None if not confident.
    """
    out = await translate_text((prev_text, curr_text), source_lang, target_lang, mode="context")
    if out and not violates_target_lang(out, target_lang):
        return out

    enforced = await translate_text_enforced(prev_text, source_lang, target_lang)
    if enforced and not violates_target_lang(enforced, target_lang):
        return enforced

    print("ℹ️ refinement skipped (empty/invalid)")
    return None

# ──────────────────────────────────────────────────────────────────────────────
# Streaming (GPT-5 new event types supported)  ← CHANGED
# ──────────────────────────────────────────────────────────────────────────────
async def stream_translate(websocket: WebSocket, text: str,
                           source_lang: str, target_lang: str) -> str:
    """
    Stream translation with GPT-5. Supports both legacy delta events and the
    newer response.output_item.* events. If stream yields nothing, use non-stream
    fallback, then enforced retry. Also rejects GPT-side hallucinations by retrying.
    """
    final_text = ""
    types_seen = set()

    try:
        buf = []
        last_partial_ts = time.monotonic()

        async with client.responses.stream(
            model="gpt-5",
            input=[
                {"role": "system", "content": translator_system_prompt(source_lang, target_lang)},
                {"role": "user",   "content": strict_user_prompt(text, source_lang, target_lang)},
            ],
            max_output_tokens=200
        ) as stream:

            async for event in stream:
                etype = getattr(event, "type", "") or ""
                types_seen.add(etype)

                # 1) Legacy: delta/text on the event itself
                delta = getattr(event, "delta", None)
                if isinstance(delta, str) and delta:
                    buf.append(delta)
                else:
                    t = getattr(event, "text", None)
                    if isinstance(t, str) and t:
                        buf.append(t)

                # 2) GPT-5: response.output_item.{added,done} carry the text
                if etype.startswith("response.output_item."):
                    item = getattr(event, "item", None)
                    content = getattr(item, "content", None) if item is not None else None
                    # content is typically a list of parts, each may have .text or dict-like
                    if content:
                        try:
                            for part in content:
                                part_text = getattr(part, "text", None)
                                if isinstance(part_text, str) and part_text:
                                    buf.append(part_text)
                                elif isinstance(part_text, dict):
                                    v = part_text.get("value") or part_text.get("text") or ""
                                    if v:
                                        buf.append(v)
                        except Exception:
                            pass

                # Partial send throttle
                s = "".join(buf)
                if s:
                    now = time.monotonic()
                    if s.endswith(("。","、",".","!","?","！","？")) or (now - last_partial_ts) > 0.18 or len(s.split()) >= 8:
                        await websocket.send_text(f"[PARTIAL]{json.dumps({'text': s}, ensure_ascii=False)}")
                        last_partial_ts = now

            # Finalization — GPT-5 may deliver only here
            final_resp = await stream.get_final_response()
            final_text = "".join(buf).strip()
            if not final_text:
                final_text = safe_output_text(final_resp) or ""

    except Exception as e:
        print("❌ stream_translate error:", e)

    if not final_text:
        # Non-streaming fallback
        print(f"⚠️ stream empty — types_seen={sorted(types_seen)}; trying non-stream")
        try:
            resp = await client.responses.create(
                model="gpt-5",
                input=[
                    {"role": "system", "content": translator_system_prompt(source_lang, target_lang)},
                    {"role": "user",   "content": strict_user_prompt(text, source_lang, target_lang)},
                ],
                max_output_tokens=200
            )
            final_text = safe_output_text(resp) or ""
        except Exception as e:
            print("❌ non-streaming fallback error:", e)

    # catch GPT-side hallucinations in the translation itself
    try:
        if final_text and await hallucination_check(final_text):
            print("🧠 GPT translation flagged as hallucination — retrying literal enforcement")
            retry = await translate_text_enforced(text, source_lang, target_lang)
            if retry:
                final_text = retry
    except Exception as e:
        print("⚠️ hallucination_check on translation failed:", e)

    # Enforce language if needed
    if not final_text or violates_target_lang(final_text, target_lang):
        print("⚠️ enforcing target language (current segment)…")
        enforced = await translate_text_enforced(text, source_lang, target_lang)
        if enforced and not violates_target_lang(enforced, target_lang):
            final_text = enforced

    if not final_text:
        print("‼️ FINAL EMPTY after stream+fallback; echoing source to avoid blank UI")
        final_text = text

    await websocket.send_text(f"[FINAL]{json.dumps({'text': final_text}, ensure_ascii=False)}")
    return final_text

# ──────────────────────────────────────────────────────────────────────────────
# WebSocket
# ──────────────────────────────────────────────────────────────────────────────
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("🔌 WebSocket connected")

    try:
        settings = await websocket.receive_text()
        cfg = json.loads(settings or "{}")
        direction = cfg.get("direction", "")
        if direction == "en-ja":
            source_lang, target_lang = "English", "Japanese"
        elif direction == "ja-en":
            source_lang, target_lang = "Japanese", "English"
        else:
            await websocket.close()
            return

        while True:
            # Receive a chunk and transcribe
            try:
                audio = await websocket.receive_bytes()
            except Exception:
                break
            if not audio:
                continue

            with tempfile.TemporaryDirectory() as td:
                raw_path = os.path.join(td, "in.webm")
                wav_path = os.path.join(td, "in.wav")
                with open(raw_path, "wb") as f:
                    f.write(audio)

                p = subprocess.run(
                    ["ffmpeg", "-hide_banner", "-loglevel", "error",
                     "-y", "-i", raw_path, "-ar", "16000", "-ac", "1", "-f", "wav", wav_path],
                    stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
                )
                if p.returncode != 0:
                    err = p.stderr.decode("utf-8", errors="ignore")
                    print("ffmpeg error:", err[:300])
                    continue

                try:
                    result = model.transcribe(
                        wav_path,
                        fp16=True,
                        temperature=0.0,
                        beam_size=1,
                        condition_on_previous_text=True,
                        no_speech_threshold=0.3,
                        compression_ratio_threshold=2.4,
                        logprob_threshold=-1.0,
                        language="en" if source_lang == "English" else "ja",
                    )
                except Exception as e:
                    print("❌ Whisper transcribe error:", e)
                    continue

            text = (result.get("text") or "").strip()
            print("📝 Transcribed:", text)
            if not text:
                continue

            # Thank-you filter
            tl = text.lower()
            if ("thank you" in tl or "thanks" in tl or
                "ありがとう" in text or "ありがとうございます" in text or "ありがと" in text):
                print("🚫 Skipping thank-you:", text)
                continue

            # Filler/CTA classifier (source-side)
            try:
                if await hallucination_check(text):
                    print("🧠 Skipping filler/hallucination:", text)
                    continue
            except Exception as e:
                print("⚠️ hallucination_check failed:", e)

            # Trivial fragment filter
            if should_skip_fragment(text, source_lang):
                print("🚫 Skipping trivial fragment:", text)
                continue

            # New segment (translate immediately)
            segment_id = str(uuid.uuid4())
            transcript_history.append((segment_id, text))

            final_translation = await stream_translate(websocket, text, source_lang, target_lang)
            print(f"✅ [SEG {segment_id[:8]}] Final translation:", final_translation[:120])
            await websocket.send_text(f"[DONE]{json.dumps({'id': segment_id, 'text': final_translation}, ensure_ascii=False)}")

            # Refinement: improve the previous segment with context from current
            if len(transcript_history) >= 2:
                prev_id, prev_text = transcript_history[-2]
                curr_id, curr_text = transcript_history[-1]
                try:
                    improved = await refine_previous(prev_text, curr_text, source_lang, target_lang)
                    if improved and improved.strip() and not violates_target_lang(improved, target_lang):
                        print(f"🔄 Refinement for prev {prev_id[:8]} using curr {curr_id[:8]}:", improved[:120])
                        await websocket.send_text(f"[UPDATE]{json.dumps({'id': prev_id, 'text': improved}, ensure_ascii=False)}")
                    else:
                        print(f"ℹ️ No valid refinement for prev {prev_id[:8]} (keeping original).")
                except Exception as e:
                    print("❌ refinement error:", e)

    except Exception as e:
        print("❌ WebSocket error (outer loop):", e)
    finally:
        try:
            await websocket.close()
        except Exception:
            pass
        print("🔌 WebSocket disconnected")

# ──────────────────────────────────────────────────────────────────────────────
# Static
# ──────────────────────────────────────────────────────────────────────────────
@app.get("/")
async def serve_index():
    return FileResponse(os.path.join("frontend", "index.html"))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
