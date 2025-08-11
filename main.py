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

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = openai.AsyncOpenAI(api_key=api_key)

# --- Warm FFMPEG and Whisper -------------------------------------------------
WARMUP_WAV = "/tmp/warm.wav"
try:
    subprocess.run(
        [
            "ffmpeg", "-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono",
            "-t", "0.5", "-ar", "16000", "-ac", "1",
            "-y", WARMUP_WAV
        ],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True
    )
except Exception:
    pass

# Optimized for T4 GPU
model = whisper.load_model("large-v3")

try:
    model.transcribe(WARMUP_WAV, language="en", fp16=True)
except Exception:
    pass

# --- FastAPI -----------------------------------------------------------------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# bounded to avoid unbounded growth
transcript_history = deque(maxlen=500)  # [(segment_id, original_text)]

# --- Debounce helpers ---------------------------------------------------------
LAST_AUDIO_TS = time.monotonic()  # updated on each received audio frame
DEBOUNCE_SEC = 0.30               # 300ms trailing quiet before we consider a segment "ready"

def segment_ready(text: str, now_ts: float, last_ts: float) -> bool:
    """Return True if we've had enough trailing quiet and content is long enough."""
    quiet_enough = (now_ts - last_ts) >= DEBOUNCE_SEC
    # "long enough": ≥6 ascii chars OR contains jp punctuation OR ≥4 CJK chars
    long_enough = (
        len(text) >= 6
        or any(p in text for p in "。！？!?、")
        or sum(1 for c in text if ord(c) >= 0x4E00) >= 4
    )
    return quiet_enough and long_enough

# --- GPT hallucination/CTA filter --------------------------------------------
async def hallucination_check(text: str) -> bool:
    """
    Conservative filler/CTA detector.
    Returns True only if the string is clearly a stock outro/CTA-style filler.
    Unknown or ambiguous phrases default to NO (i.e., False).
    """
    try:
        prompt = (
            "You are a strict, conservative classifier for filler/CTA phrases.\n"
            "Task: Decide if the sentence is a generic, stock, AI-ish filler line commonly used as video/podcast/stream outros or engagement CTAs.\n\n"
            "Positive categories (say YES only if a clear match or close paraphrase):\n"
            "- Thanks/see-you outros (e.g., 'Thanks for watching', 'See you in the next video').\n"
            "- Engagement CTAs (e.g., 'Like and subscribe', 'Hit the bell', 'Share with your friends').\n"
            "- Follow/subscribe requests across platforms (e.g., 'Follow me on X/Instagram', 'Subscribe for more').\n"
            "- Generic sign-offs with no content (e.g., 'That's all for today', 'Stay tuned for more').\n\n"
            "Negative indicators (say NO):\n"
            "- Any sentence containing specific information, claims, instructions, or context (facts, names, numbers, steps, dates, product features, agendas).\n"
            "- Opinions about content, analysis, or summaries that reference specifics.\n"
            "- Sponsor reads or calls-to-action that include concrete details (codes, URLs, product names) → NO unless the entire line is a bare generic CTA.\n"
            "- Greetings/intros ('Hi everyone, welcome back') → NO (not an outro/engagement CTA).\n\n"
            "Decision rules:\n"
            "- Be conservative: if unfamiliar, ambiguous, or partly specific → NO.\n"
            "- Only YES if the sentence fits a Positive category without additional specific content.\n"
            "- Language-agnostic: if not in English, judge by meaning; still output YES/NO in English.\n"
            "- Output EXACTLY 'YES' or 'NO'. No punctuation or extra words.\n\n"
            "Positive examples (YES):\n"
            "- Thanks for watching\n"
            "- Don't forget to subscribe\n"
            "- Click the bell icon\n"
            "- See you in the next video\n"
            "- Like and share\n"
            "- Follow me for more\n"
            "- That's all for today\n"
            "- Subscribe for more content\n\n"
            "Negative examples (NO):\n"
            "- In 2024, we grew revenue by 18% across APAC.\n"
            "- Click the link in the description to access the dataset we covered today (v3.2, updated May 5).\n"
            "- Thanks to Acme Corp for sponsoring this episode with code ACME20.\n"
            "- We'll compare BERT and GPT architectures focusing on attention mechanisms.\n"
            "- お時間ある方は資料の3ページ目をご確認ください。\n"
            "- 次回は6月15日に大阪でワークショップを開催します。\n\n"
            "Answer strictly with YES or NO.\n\n"
            f"Sentence:\n{text}"
        )

        result = await client.chat.completions.create(
            model="gpt-5-nano",  # bump to gpt-5-mini if you need more nuance
            messages=[
                {"role": "system", "content": "Reply ONLY with YES or NO. Be conservative; unknowns default to NO."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            top_p=0,
            max_tokens=1
        )
        return result.choices[0].message.content.strip().upper() == "YES"
    except Exception:
        return False

# --- Natural translation (non-streaming; used for context-refine) -------------
async def translate_text(text, source_lang: str, target_lang: str, mode: str = "default"):
    """
    Bidirectional EN↔JA translation with optional context refinement.
    - mode="default": one-shot natural translation
    - mode="context": refine translation of the PREVIOUS sentence using the NEW sentence
    """
    system_prompt = (
        f"You are a professional bidirectional {source_lang} ↔ {target_lang} translator and simultaneous interpreter.\n"
        f"Goals: produce natural, fluent, culturally appropriate {target_lang}; preserve meaning, tone, and intent; keep terminology consistent across turns.\n\n"
        f"Policy:\n"
        f"- Output ONLY the translation in {target_lang} (no quotes, no commentary).\n"
        f"- If input is already in {target_lang}, return it unchanged.\n"
        f"- Preserve proper names, numbers, and units; use established translations when widely known.\n"
        f"- Prefer idiomatic equivalents over literal calques.\n"
        f"- Handle partial or interrupted speech gracefully; produce the most natural fragment possible.\n\n"
        f"When translating INTO Japanese:\n"
        f"- Default to polite です・ます unless the source is clearly casual or quoted; keep honorifics/titles; omit pronouns when natural; use Japanese punctuation; transliterate unknown names inカタカナ.\n\n"
        f"When translating INTO English:\n"
        f"- Use idiomatic, concise phrasing; make implicit subjects explicit when needed; avoid awkward literalness."
    )

    if mode == "context":
        previous, current = text
        user_prompt = (
            f"Refine the translation of the PREVIOUS sentence using the NEW sentence as context.\n\n"
            f"PREVIOUS ({source_lang}): {previous}\n"
            f"NEW ({source_lang}): {current}\n\n"
            f"Rules:\n"
            f"- Return ONLY the improved translation of the PREVIOUS sentence in {target_lang}.\n"
            f"- Update wording ONLY if NEW changes meaning, tone, terminology, named entities, or register; otherwise keep the translation unchanged.\n"
            f"- Do NOT translate or repeat the NEW sentence.\n"
            f"- Do NOT repeat content already translated unless required for naturalness.\n"
            f"- If PREVIOUS and NEW are near-duplicates, translate PREVIOUS once.\n"
            f"- If PREVIOUS is incomplete, produce the most natural fragment consistent with NEW."
        )
    else:
        user_prompt = (
            f"Translate the following from {source_lang} to {target_lang}.\n\n"
            f"Sentence:\n{text}\n\n"
            f"Constraints:\n"
            f"- Output ONLY the translation in {target_lang}; no extra words.\n"
            f"- If the sentence is already in {target_lang}, return it unchanged."
        )

    try:
        response = await client.chat.completions.create(
            model="gpt-5",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2,
            max_tokens=200
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print("❌ Translation error:", e)
        return text if isinstance(text, str) else text[0]

# --- Streaming translator (for live feel) ------------------------------------
async def stream_translate(websocket: WebSocket, text: str, source_lang: str, target_lang: str) -> str:
    """
    Streams a translation for 'text' and sends [PARTIAL] events during generation,
    then a [FINAL] event at completion. Returns the final translated string.
    """
    system_prompt = (
        f"You are a professional bidirectional {source_lang} ↔ {target_lang} translator and simultaneous interpreter.\n"
        f"Goals: produce natural, fluent, culturally appropriate {target_lang}; preserve meaning, tone, and intent; keep terminology consistent across turns.\n\n"
        f"Policy:\n"
        f"- Output ONLY the translation in {target_lang} (no quotes, no commentary).\n"
        f"- If input is already in {target_lang}, return it unchanged.\n"
        f"- Prefer idiomatic equivalents over literal calques; handle partial speech gracefully."
    )
    user_prompt = (
        f"Translate the following from {source_lang} to {target_lang}.\n\n"
        f"Sentence:\n{text}\n\n"
        f"Constraints:\n"
        f"- Output ONLY the translation in {target_lang}; no extra words.\n"
        f"- If the sentence is already in {target_lang}, return it unchanged."
    )

    try:
        stream = await client.chat.completions.create(
            model="gpt-5",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2,
            max_tokens=200,
            stream=True,
        )

        partial_buf = []
        last_flush = time.monotonic()

        async for chunk in stream:
            try:
                delta = chunk.choices[0].delta.get("content") or ""
            except Exception:
                delta = ""
            if not delta:
                continue

            partial_buf.append(delta)
            s = "".join(partial_buf)

            now = time.monotonic()
            # flush policy: punctuation boundary OR 180ms elapsed OR ≥8 (space-delimited) tokens
            boundary = s.endswith(("。", "、", ".", "!", "?", "！", "？"))
            elapsed = (now - last_flush) > 0.18
            tokish = (len(s.split()) >= 8)

            if boundary or elapsed or tokish:
                await websocket.send_text(f"[PARTIAL]{json.dumps({'text': s}, ensure_ascii=False)}")
                last_flush = now

        final_text = "".join(partial_buf).strip()
        await websocket.send_text(f"[FINAL]{json.dumps({'text': final_text}, ensure_ascii=False)}")
        return final_text
    except Exception as e:
        print("❌ Streaming translate error:", e)
        # fall back to non-streaming
        try:
            final_text = await translate_text(text, source_lang, target_lang)
            await websocket.send_text(f"[FINAL]{json.dumps({'text': final_text}, ensure_ascii=False)}")
            return final_text
        except Exception:
            return ""

# --- WebSocket: receive config then audio frames ------------------------------
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("🔌 WebSocket connected")

    global LAST_AUDIO_TS

    try:
        # config message (JSON text)
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

        # main loop: audio bytes
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

                try:
                    subprocess.run(
                        [
                            "ffmpeg", "-y",
                            "-i", raw_path,
                            "-af", "silenceremove=1:0:-40dB",
                            "-ar", "16000", "-ac", "1",
                            wav_path
                        ],
                        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True
                    )
                except Exception:
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

            text = result.get("text", "").strip()
            print("📝 Transcribed:", text)
            if not text:
                continue

            # quick thank-you filter
            tl = text.lower()
            if ("thank you" in tl or "thanks" in tl or
                "ありがとう" in text or "ありがとうございます" in text or "ありがと" in text):
                print("🚫 Skipping thank-you/ありがとう phrase:", text)
                continue

            # GPT-based hallucination filter
            if await hallucination_check(text):
                print("🧠 GPT flagged as hallucination:", text)
                continue

            # debounce + min-length gate
            now_ts = time.monotonic()
            if not segment_ready(text, now_ts, LAST_AUDIO_TS):
                # Not "stable" yet; send a soft partial echo so UI can show pending STT if you like
                await websocket.send_text(f"[PARTIAL_ASR]{json.dumps({'text': text}, ensure_ascii=False)}")
                continue

            # assign ID and stream-translate
            segment_id = str(uuid.uuid4())
            transcript_history.append((segment_id, text))

            final_translation = await stream_translate(websocket, text, source_lang, target_lang)
            await websocket.send_text(f"[DONE]{json.dumps({'id': segment_id, 'text': final_translation}, ensure_ascii=False)}")

            # context-refine the previous translation (non-streaming)
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
