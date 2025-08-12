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
# Hallucination / Filler filter
# ──────────────────────────────────────────────────────────────────────────────
async def hallucination_check(text: str) -> bool:
    """
    Returns True if 'text' is generic filler/CTA to skip.
    (min tokens >= 16; no temperature param)
    """
    try:
        prompt = (
            "You are a strict, conservative classifier for filler/CTA phrases.\n"
            "Say YES only for generic outros/engagement CTAs; otherwise NO.\n"
            "Output YES or NO only.\n\nSentence:\n" + text
        )
        resp = await client.responses.create(
            model="gpt-5-nano",
            input=[
                {"role": "system", "content": "Reply ONLY with YES or NO. Be conservative; unknowns default to NO."},
                {"role": "user", "content": prompt},
            ],
            max_output_tokens=16  # GPT-5 responses API requires >= 16
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
        # Do NOT return tuples
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
# Streaming (robust for GPT-5)
# ──────────────────────────────────────────────────────────────────────────────
async def stream_translate(websocket: WebSocket, text: str,
                           source_lang: str, target_lang: str) -> str:
    """
    Stream translation with GPT-5, compatible with models that may emit no deltas.
    If the stream yields nothing, use non-streaming fallback, then enforced retry.
    Never returns empty.
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

                # Capture any event that carries text
                delta = getattr(event, "delta", None)
                if isinstance(delta, str) and delta:
                    buf.append(delta)
                else:
                    t = getattr(event, "text", None)
                    if isinstance(t, str) and t:
                        buf.append(t)

                # Throttle partials by time / punctuation / token count
                s = "".join(buf)
                if s:
                    now = time.monotonic()
                    if s.endswith(("。", "、", ".", "!", "?", "！", "？")) or (now - last_partial_ts) > 0.18 or len(s.split()) >= 8:
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

            # Filler/CTA classifier
            try:
                if await hallucination_check(text):
                    print("🧠 Skipping filler/hallucination:", text)
                    continue
            except Exception as e:
                print("⚠️ hallucination_check failed:", e)

            # Trivial fragment filter (optional; comment out if you want every crumb)
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
