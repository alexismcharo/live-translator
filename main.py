import tempfile, subprocess, uvicorn, openai, whisper, os, json, uuid, deepl
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
deepl_auth_key = os.getenv("DEEPL_API_KEY")
client = openai.AsyncOpenAI(api_key=api_key)
translator = deepl.Translator(deepl_auth_key)

try:
    subprocess.run([
        "ffmpeg", "-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono",
        "-t", "0.5", "-ar", "16000", "-ac", "1",
        "-y", "/tmp/warm.wav"
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
except:
    pass

model = whisper.load_model("large-v3")

try:
    model.transcribe("warmup.wav", language="en", fp16=True)
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

# hallucination filter via GPT
async def hallucination_check(text):
    try:
        prompt = (
            "You are a strict hallucination filter. Your task is to detect whether a sentence sounds like a generic, fabricated, or AI-generated filler phrase.\n\n"
            "Examples:\n"
            "- 'Thanks for watching'\n"
            "- 'Don't forget to subscribe'\n"
            "- 'Click the bell icon'\n"
            "- 'See you in the next video'\n"
            "- 'Like and share'\n\n"
            "If the sentence contains any such content, return only: YES\nOtherwise: NO\n\n"
            f"Sentence:\n{text}"
        )

        result = await client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You judge if the sentence is hallucinated filler. Only reply YES or NO."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=1
        )
        return result.choices[0].message.content.strip().upper() == "YES"
    except:
        return False

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

                    # filter thank-you phrases
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

                    # hallucination check
                    if await hallucination_check(text):
                        print("🧠 GPT flagged as hallucination:", text)
                        continue

                    # assign ID and translate
                    segment_id = str(uuid.uuid4())
                    transcript_history.append((segment_id, text))

                    translated = await translate_text(text, source_lang, target_lang)
                    await websocket.send_text(f"[DONE]{json.dumps({'id': segment_id, 'text': translated})}")

                    # update previous with new context
                    if len(transcript_history) >= 2:
                        prev, curr = transcript_history[-2][1], transcript_history[-1][1]
                        improved = await translate_text((prev, curr), source_lang, target_lang, mode="context")
                        await websocket.send_text(f"[UPDATE]{json.dumps({'id': transcript_history[-2][0], 'text': improved})}")

    except Exception as e:
        print("❌ WebSocket error:", e)
        await websocket.close()

# GPT + DeepL translation combo
async def translate_text(text, source_lang, target_lang, mode="default"):
    try:
        if mode == "context":
            previous, current = text
            prompt = (
                f"You are a strict, literal translation engine. Refine the translation of the previous sentence based on new context.\n\n"
                f"Previous: {previous}\n"
                f"Current: {current}\n\n"
                f"Rules:\n"
                f"- Do NOT repeat previous sentences unless their meaning changes.\n"
                f"- Merge or rephrase ONLY if new information adds clarity.\n"
                f"- Return the improved translation of the 'Previous' sentence only.\n"
                f"- Do NOT repeat phrases that were already translated unless absolutely necessary.\n"
                f"- If a sentence is identical or near-identical to the previous one, translate it only once.\n"
            )
            refined = await call_gpt(prompt)
        else:
            prompt = (
                f"You are a strict, literal translation engine. Translate the sentence below from {source_lang} to {target_lang}.\n\n"
                f"Rules:\n"
                f"- Return only the translated sentence. No commentary, no meta info.\n"
                f"- Never say 'already in English/Japanese'. If it's already translated, return it as-is.\n"
                f"- No thank yous, greetings, or filler.\n\n"
                f"Sentence: {text}"
            )
            refined = await call_gpt(prompt)
        
        final = await translate_with_deepl(refined, source_lang, target_lang)
        return final

    except Exception as e:
        print("❌ Translation error:", e)
        return text

# call OpenAI GPT-4o
async def call_gpt(prompt):
    response = await client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a strict and literal translator."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
        max_tokens=60
    )
    return response.choices[0].message.content.strip()

# use DeepL to polish the translation
async def translate_with_deepl(text, source_lang, target_lang):
    try:
        result = translator.translate_text(
            text,
            source_lang="EN-GB" if source_lang == "English" else "JA",
            target_lang="JA" if target_lang == "Japanese" else "EN-GB",
            formality="default"
        )
        return result.text
    except Exception as e:
        print("❌ DeepL translation failed:", e)
        return text

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
