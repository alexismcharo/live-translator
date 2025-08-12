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

# Keep original (source) text history for contextual refinement: [(segment_id, original_text)]
transcript_history = deque(maxlen=500)

# ---------------- Hallucination / Filler Filter ----------------
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
        return (content or "").strip().upper() == "YES"
    except Exception as e:
        print("⚠️ hallucination_check error:", e)
        return False

# ---------------- Translation ----------------
def translator_system_prompt(source_lang: str, target_lang: str) -> str:
    return (
        f"You are a professional {source_lang}↔{target_lang} translator.\n"
        f"- Output ONLY in {target_lang}. No quotes, no commentary.\n"
        f"- Translate even if the source is a fragment or incomplete.\n"
        f"- Never return an empty string; if uncertain, translate literally."
    )

async def translate_text(text, source_lang: str, target_lang: str, mode: str = "default") -> str:
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
            f"Return ONLY the {target_lang} translation (no extra words)."
        )

    try:
        resp = await client.responses.create(
            model="gpt-5",
            input=[
                {"role":"system","content":translator_system_prompt(source_lang, target_lang)},
                {"role":"user","content":user_prompt}
            ],
            max_output_tokens=200
        )
        out = safe_output_text(resp)
        if not out:
            print("⚠️ translate_text returned empty; falling back to literal source.")
        return out or (text if isinstance(text, str) else str(text))
    except Exception as e:
        print("❌ translate_text error:", e)
        return text if isinstance(text, str) else str(text)

async def stream_translate(websocket: WebSocket, text: str, source_lang: str, target_lang: str) -> str:
    """
    Stream translation to the UI with [PARTIAL] updates and a [FINAL] at the end.
    Returns final translated text. On failure, falls back to non-streaming call.
    """
    try:
        buf, last = [], time.monotonic()
        async with client.responses.stream(
            model="gpt-5",
            input=[
                {"role": "system", "content": translator_system_prompt(source_lang, target_lang)},
                {"role": "user", "content":
                    f"Translate from {source_lang} to {target_lang}.\n\n"
                    f"Sentence:\n{text}\n\n"
                    f"Return ONLY the {target_lang} translation (no extra words)."
                },
            ],
            max_output_tokens=200,
        ) as stream:

            async for event in stream:
                et = getattr(event, "type", "")
                if et in ("response.output_text.delta", "response.delta", "message.delta"):
                    delta = (
                        getattr(event, "delta", None)
                        or getattr(event, "text", None)
                        or ""
                    )
                    if not isinstance(delta, str) or not delta:
                        continue

                    buf.append(delta)
                    s = "".join(buf)
                    now = time.monotonic()
                    # Send partials on punctuation, time, or token count thresholds
                    if s.endswith(("。","、",".","!","?","！","？")) or (now - last) > 0.18 or len(s.split()) >= 8:
                        await websocket.send_text(f"[PARTIAL]{json.dumps({'text': s}, ensure_ascii=False)}")
                        last = now

            final_resp = await stream.get_final_response()
            final_text = "".join(buf).strip()
            if not final_text:
                final_text = safe_output_text(final_resp)

        if not final_text:
            # Fallback non-streaming
            resp = await client.responses.create(
                model="gpt-5",
                input=[
                    {"role":"system","content":translator_system_prompt(source_lang, target_lang)},
                    {"role":"user","content":
                        f"Translate from {source_lang} to {target_lang}.\n\n"
                        f"Sentence:\n{text}\n\n"
                        f"Return ONLY the {target_lang} translation (no extra words)."
                    },
                ],
                max_output_tokens=200
            )
            final_text = safe_output_text(resp)

        if not final_text:
            print("⚠️ stream_translate produced no text; returning source.")
            final_text = text

        await websocket.send_text(f"[FINAL]{json.dumps({'text': final_text}, ensure_ascii=False)}")
        return final_text

    except Exception as e:
        print("❌ stream_translate error:", e)
        fallback = await translate_text(text, source_lang, target_lang)
        await websocket.send_text(f"[FINAL]{json.dumps({'text': fallback or text}, ensure_ascii=False)}")
        return fallback or text

# ---------------- WebSocket ----------------
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

        # ALWAYS translate every chunk immediately; then refine the previous one
        while True:
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

            # Thank-you filter
            tl = text.lower()
            if ("thank you" in tl or "thanks" in tl or
                "ありがとう" in text or "ありがとうございます" in text or "ありがと" in text):
                print("🚫 Skipping thank-you:", text)
                continue

            # Filler/hallucination filter
            try:
                if await hallucination_check(text):
                    print("🧠 Skipping filler/hallucination:", text)
                    continue
            except Exception as e:
                print("⚠️ hallucination_check failed:", e)

            # New segment for THIS chunk (translate immediately)
            segment_id = str(uuid.uuid4())
            transcript_history.append((segment_id, text))

            # Stream translation now (immediate display)
            final_translation = await stream_translate(websocket, text, source_lang, target_lang)
            print(f"✅ [SEG {segment_id[:8]}] Final translation:", final_translation[:120])
            await websocket.send_text(f"[DONE]{json.dumps({'id': segment_id, 'text': final_translation}, ensure_ascii=False)}")

            # Refine PREVIOUS segment (if any) using CURRENT as context
            if len(transcript_history) >= 2:
                prev_id, prev_text = transcript_history[-2]
                curr_id, curr_text = transcript_history[-1]
                try:
                    improved = await translate_text((prev_text, curr_text), source_lang, target_lang, mode="context")
                    print(f"🔄 Refinement for prev {prev_id[:8]} using curr {curr_id[:8]}:", improved[:120])
                    await websocket.send_text(f"[UPDATE]{json.dumps({'id': prev_id, 'text': improved}, ensure_ascii=False)}")
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

@app.get("/")
async def serve_index():
    return FileResponse(os.path.join("frontend", "index.html"))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
