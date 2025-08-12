import os
import json
import uuid
import time
import asyncio
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

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = openai.AsyncOpenAI(api_key=api_key)

# ---------------- Utilities ----------------
def safe_output_text(resp) -> str:
    """
    Robustly extract text from a Responses API object.
    Falls back to walking resp.output[...] if output_text is missing.
    """
    txt = getattr(resp, "output_text", None)
    if isinstance(txt, str) and txt.strip():
        return txt.strip()

    out = []
    output = getattr(resp, "output", []) or []
    for item in output:
        content = getattr(item, "content", []) or []
        for c in content:
            t = getattr(c, "text", None) or getattr(c, "input_text", None)
            if isinstance(t, str):
                out.append(t)
    return "".join(out).strip()

# ------------- Warm FFMPEG & Whisper --------------
WARMUP_WAV = "/tmp/warm.wav"
try:
    subprocess.run(
        ["ffmpeg","-f","lavfi","-i","anullsrc=r=16000:cl=mono","-t","0.5","-ar","16000","-ac","1","-y",WARMUP_WAV],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True
    )
except Exception:
    pass

model = whisper.load_model("large-v3")
try:
    model.transcribe(WARMUP_WAV, language="en", fp16=True)
except Exception:
    pass

# ---------------- FastAPI ----------------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

transcript_history = deque(maxlen=500)  # [(segment_id, original_text)]

# ---------------- Debounce / Flush ----------------
LAST_AUDIO_TS = time.monotonic()   # updated when a blob arrives
DEBOUNCE_SEC = 0.30                # min trailing quiet before "ready"
INACTIVITY_FLUSH_SEC = 0.60        # finalize if quiet for this long

_pending_text = ""
_pending_lock = asyncio.Lock()
_pending_task = None

def segment_ready(text: str, now_ts: float, last_ts: float) -> bool:
    """True if we've had enough trailing quiet and content is long enough."""
    quiet = now_ts - last_ts
    short_and_quiet = (len(text) >= 3) and (quiet >= 0.50)
    long_enough = (
        len(text) >= 6
        or any(p in text for p in "。！？!?、")
        or sum(1 for c in text if ord(c) >= 0x4E00) >= 3
        or short_and_quiet
    )
    return (quiet >= DEBOUNCE_SEC) and long_enough

def set_pending_text(t: str):
    global _pending_text
    _pending_text = (t or "").strip()

async def _schedule_inactivity_flush(websocket: WebSocket, source_lang: str, target_lang: str):
    """Start/restart a timer that finalizes pending ASR if the mic is quiet."""
    global _pending_task
    if _pending_task and not _pending_task.done():
        _pending_task.cancel()

    async def _flush_when_quiet():
        try:
            await asyncio.sleep(INACTIVITY_FLUSH_SEC)
            quiet = time.monotonic() - LAST_AUDIO_TS
            if quiet < INACTIVITY_FLUSH_SEC:
                return
            async with _pending_lock:
                text = _pending_text.strip()
                if not text:
                    return
                set_pending_text("")
            # finalize → translate
            segment_id = str(uuid.uuid4())
            transcript_history.append((segment_id, text))
            final_translation = await stream_translate(websocket, text, source_lang, target_lang)
            print("✅ Final translation:", final_translation[:120])
            await websocket.send_text(f"[DONE]{json.dumps({'id': segment_id, 'text': final_translation}, ensure_ascii=False)}")
            # context refine previous
            if len(transcript_history) >= 2:
                prev, curr = transcript_history[-2][1], transcript_history[-1][1]
                improved = await translate_text((prev, curr), source_lang, target_lang, mode="context")
                await websocket.send_text(f"[UPDATE]{json.dumps({'id': transcript_history[-2][0], 'text': improved}, ensure_ascii=False)}")
        except asyncio.CancelledError:
            pass

    _pending_task = asyncio.create_task(_flush_when_quiet())

# ---------------- Hallucination filter ----------------
async def hallucination_check(text: str) -> bool:
    try:
        prompt = (
            "You are a strict, conservative classifier for filler/CTA phrases.\n"
            "Say YES only for generic outros/engagement CTAs; otherwise NO.\n"
            "Output YES or NO only.\n\nSentence:\n" + text
        )
        resp = await client.responses.create(
            model="gpt-5-nano",
            input=[
                {"role":"system","content":"Reply ONLY with YES or NO. Be conservative; unknowns default to NO."},
                {"role":"user","content":prompt}
            ],
            max_output_tokens=1
        )
        content = safe_output_text(resp)
        return content.upper() == "YES"
    except Exception:
        return False

# ---------------- Translation ----------------
async def translate_text(text, source_lang: str, target_lang: str, mode: str = "default"):
    system_prompt = (
        f"You are a professional {source_lang}↔{target_lang} translator.\n"
        f"- Output ONLY in {target_lang}. No quotes, no commentary.\n"
        f"- If translating INTO Japanese, write in kanji/kana (no romaji, no English)."
    )
    if mode == "context":
        previous, current = text
        user_prompt = (
            f"Refine the translation of the PREVIOUS sentence using the NEW sentence as context.\n\n"
            f"PREVIOUS ({source_lang}): {previous}\n"
            f"NEW ({source_lang}): {current}\n\n"
            f"Return ONLY the improved translation of PREVIOUS in {target_lang}."
        )
    else:
        user_prompt = (
            f"Translate from {source_lang} to {target_lang}.\n\n"
            f"Sentence:\n{text}\n\n"
            f"Return ONLY the {target_lang} translation."
        )

    try:
        resp = await client.responses.create(
            model="gpt-5",
            input=[
                {"role":"system","content":system_prompt},
                {"role":"user","content":user_prompt}
            ],
            max_output_tokens=200
        )
        out = safe_output_text(resp)
        return out
    except Exception as e:
        print("❌ translate_text error:", e)
        return ""

# ---------------- Streaming translate (Responses API) ----------------
async def stream_translate(websocket: WebSocket, text: str, source_lang: str, target_lang: str) -> str:
    system_prompt = (
        f"You are a professional {source_lang}↔{target_lang} translator.\n"
        f"- Output ONLY in {target_lang}. No quotes, no commentary.\n"
        f"- If translating INTO Japanese, write in kanji/kana (no romaji, no English)."
    )
    user_prompt = (
        f"Translate from {source_lang} to {target_lang}.\n\n"
        f"Sentence:\n{text}\n\n"
        f"Return ONLY the {target_lang} translation."
    )

    try:
        buf, last = [], time.monotonic()

        # Correct streaming pattern: async context manager + iterate events
        async with client.responses.stream(
            model="gpt-5",
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_output_tokens=200,
        ) as stream:

            async for event in stream:
                et = getattr(event, "type", "")
                if et == "response.output_text.delta":
                    delta = getattr(event, "delta", "") or ""
                    if not delta:
                        continue
                    buf.append(delta)
                    s = "".join(buf)
                    now = time.monotonic()
                    if s.endswith(("。","、",".","!","?","！","？")) or (now - last) > 0.18 or len(s.split()) >= 8:
                        await websocket.send_text(f"[PARTIAL]{json.dumps({'text': s}, ensure_ascii=False)}")
                        last = now
                # (optional) handle other event types as needed

            final_resp = await stream.get_final_response()
            final_text = "".join(buf).strip()
            if not final_text:
                final_text = safe_output_text(final_resp)

        await websocket.send_text(f"[FINAL]{json.dumps({'text': final_text}, ensure_ascii=False)}")
        return final_text

    except Exception as e:
        print("❌ stream_translate error:", e)
        strict = await translate_text(text, source_lang, target_lang)
        await websocket.send_text(f"[FINAL]{json.dumps({'text': strict}, ensure_ascii=False)}")
        return strict

# ---------------- WebSocket ----------------
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("🔌 WebSocket connected")

    global LAST_AUDIO_TS

    try:
        # config
        settings = await websocket.receive_text()
        cfg = json.loads(settings)
        direction = cfg.get("direction")
        if direction == "en-ja":
            source_lang, target_lang = "English", "Japanese"
        elif direction == "ja-en":
            source_lang, target_lang = "Japanese", "English"
        else:
            await websocket.close()
            return

        # audio loop
        while True:
            try:
                audio = await websocket.receive_bytes()
                LAST_AUDIO_TS = time.monotonic()
            except Exception:
                continue
            if not audio:
                continue

            with tempfile.TemporaryDirectory() as td:
                raw_path = os.path.join(td, "in.webm")
                wav_path = os.path.join(td, "in.wav")
                with open(raw_path, "wb") as f:
                    f.write(audio)

                # Transcode WITHOUT silenceremove; surface errors
                p = subprocess.run(
                    ["ffmpeg","-y","-i",raw_path,"-ar","16000","-ac","1",wav_path],
                    stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
                )
                if p.returncode != 0:
                    err = p.stderr.decode("utf-8", errors="ignore")
                    print("ffmpeg error:", err[:500])
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

            # quick thank-you filter
            tl = text.lower()
            if ("thank you" in tl or "thanks" in tl or "ありがとう" in text or "ありがとうございます" in text or "ありがと" in text):
                print("🚫 Skipping thank-you/ありがとう phrase:", text)
                continue

            # hallucination filter
            if await hallucination_check(text):
                print("🧠 GPT flagged as hallucination:", text)
                continue

            # debounce + min-length gate
            now_ts = time.monotonic()
            if not segment_ready(text, now_ts, LAST_AUDIO_TS):
                await websocket.send_text(f"[PARTIAL_ASR]{json.dumps({'text': text}, ensure_ascii=False)}")
                async with _pending_lock:
                    set_pending_text(text)
                await _schedule_inactivity_flush(websocket, source_lang, target_lang)
                continue
            else:
                async with _pending_lock:
                    set_pending_text("")

            # finalize immediately
            segment_id = str(uuid.uuid4())
            transcript_history.append((segment_id, text))
            final_translation = await stream_translate(websocket, text, source_lang, target_lang)
            print("✅ Final translation:", final_translation[:120])
            await websocket.send_text(f"[DONE]{json.dumps({'id': segment_id, 'text': final_translation}, ensure_ascii=False)}")

            if len(transcript_history) >= 2:
                prev, curr = transcript_history[-2][1], transcript_history[-1][1]
                improved = await translate_text((prev, curr), source_lang, target_lang, mode="context")
                await websocket.send_text(f"[UPDATE]{json.dumps({'id': transcript_history[-2][0], 'text': improved}, ensure_ascii=False)}")

    except Exception as e:
        print("❌ WebSocket error:", e)
        await websocket.close()

@app.get("/")
async def serve_index():
    return FileResponse(os.path.join("frontend", "index.html"))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
