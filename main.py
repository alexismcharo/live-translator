import tempfile, subprocess, uvicorn, openai, whisper, os, json, uuid
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = openai.AsyncOpenAI(api_key=api_key)

# Pre-warm ffmpeg & audio stack
try:
    subprocess.run([
        "ffmpeg", "-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono",
        "-t", "0.5", "-ar", "16000", "-ac", "1",
        "-y", "/tmp/warm.wav"
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
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

transcript_history = []  # [(segment_id, original_text)]

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

Answer with exactly YES or NO.
""".strip()

        result = await client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ],
            temperature=0,
            max_tokens=1
        )
        return result.choices[0].message.content.strip().upper() == "YES"
    except:
        return False

# --- WebSocket for streaming translation ---
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

            with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as raw:
                raw.write(audio)
                raw.flush()

                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as wav:
                    try:
                        subprocess.run([
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

                    result = model.transcribe(
                        wav.name,
                        fp16=True,
                        temperature=0.0,
                        beam_size=1,
                        condition_on_previous_text=True,
                        hallucination_silence_threshold=0.2,
                        no_speech_threshold=0.3,
                        language="en" if source_lang == "English" else "ja",
                        compression_ratio_threshold=2.4,
                        logprob_threshold=-1.0
                    )

                    text = result["text"].strip()
                    print("📝 Transcribed:", text)
                    if not text:
                        continue

                    # GPT-based hallucination filter (keeps normal conversational turns)
                    if await hallucination_check(text):
                        print("🧠 GPT flagged as broadcast/CTA filler:", text)
                        continue

                    # assign ID and translate
                    segment_id = str(uuid.uuid4())
                    transcript_history.append((segment_id, text))

                    translated = await translate_text(text, source_lang, target_lang)
                    await websocket.send_text(f"[DONE]{json.dumps({'id': segment_id, 'text': translated})}")

                    # update previous translation using new context
                    if len(transcript_history) >= 2:
                        prev, curr = transcript_history[-2][1], transcript_history[-1][1]
                        improved = await translate_text((prev, curr), source_lang, target_lang, mode="context")
                        await websocket.send_text(f"[UPDATE]{json.dumps({'id': transcript_history[-2][0], 'text': improved})}")

    except Exception as e:
        print("❌ WebSocket error:", e)
        await websocket.close()

# --- Main translation function (natural, live-caption style) ---
async def translate_text(text, source_lang, target_lang, mode="default"):
    """
    Natural, idiomatic live captions; mirrors fragment completeness; minimal edits on refinements.
    """
    target_register = "polite" if target_lang == "Japanese" else "neutral"

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

<rules>
- Output: ONLY the improved translation of <previous>. No quotes, no commentary.
- Prefer minimal edits to avoid visual 'jumping' in captions.
- Make phrasing natural in {target_lang} (not literal), but add no new information.
- If <previous> was a fragment, keep it a natural fragment; don't invent endings.
- Resolve pronouns, names, tense, or ellipsis only if <current> makes them clear.
- Remove filler like uh/um/えっと/あの unless meaningful.
- Keep numbers as digits and preserve proper nouns/terminology.
- Register: for Japanese use {"です・ます" if target_register=="polite" else "casual speech"}; for English use {target_register} spoken style.
</rules>

<previous>
{previous}
</previous>

<current>
{current}
</current>
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

<rules>
- Natural over literal: use target-language word order and phrasing; keep meaning faithful.
- Mirror completeness: if the source is a fragment, keep a natural fragment; do not guess the rest.
- Remove pure filled pauses (uh/um/えっと/あの) unless they carry meaning.
- Keep numbers as digits; preserve names, technical terms, and units.
- Do not add greetings/sign-offs/explanations or YouTube-style CTAs.
- If input is already in {target_lang}, return it unchanged.
- Register: for Japanese use {"です・ます" if target_register=="polite" else "casual speech"}; for English use {target_register} spoken style.
</rules>

<input>
{text}
</input>
""".strip()

    try:
        response = await client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ],
            temperature=0.3,   # slightly higher for more natural phrasing
            max_tokens=60
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print("❌ Translation error:", e)
        return text

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
