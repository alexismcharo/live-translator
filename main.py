import os, json, uuid, tempfile, subprocess, torch, whisper, uvicorn, openai
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from transformers import AutoTokenizer, AutoModelForCausalLM
from dotenv import load_dotenv

# === ENV & MODELS ===
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = openai.AsyncOpenAI(api_key=api_key)  # Using AsyncOpenAI client (requires openai v1.x)

# Load Whisper model for transcription
model = whisper.load_model("large-v3")

# Load DeepSeek R1 (chat model) with proper trust_remote_code
device = "cuda" if torch.cuda.is_available() else "cpu"
deepseek_tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-moe-16b-chat", trust_remote_code=True)
deepseek_model = AutoModelForCausalLM.from_pretrained(
    "deepseek-ai/deepseek-moe-16b-chat",
    trust_remote_code=True,  # Fix: allow loading custom code for this model
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)

# === FastAPI Setup ===
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

# No global transcript_history – will use per-connection history

# === Warmup ===
try:
    subprocess.run([
        "ffmpeg", "-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono",
        "-t", "0.5", "-ar", "16000", "-ac", "1",
        "-y", "/tmp/warm.wav"
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
except:
    pass

try:
    model.transcribe("/tmp/warm.wav", language="en", fp16=True)
except:
    pass

@app.get("/")
async def serve_index():
    return FileResponse(os.path.join("frontend", "index.html"))

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("🔌 WebSocket connected")

    # Initialize transcript history for this connection
    transcript_history = []  # Use a fresh list for each websocket session

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

            # Write audio chunk to a temporary file for processing
            with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as raw:
                raw.write(audio)
                raw.flush()
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as wav:
                    try:
                        subprocess.run([
                            "ffmpeg", "-y",
                            "-i", raw.name,
                            "-af", "silenceremove=1:0:-40dB",  # remove silence
                            "-ar", "16000", "-ac", "1",
                            wav.name
                        ], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True)
                    except:
                        continue  # if ffmpeg fails, skip this chunk

                    # Transcribe (and possibly translate) the audio chunk
                    if source_lang == "Japanese":
                        # Whisper translates Japanese speech to English text
                        result = model.transcribe(
                            wav.name, fp16=True, task="translate", language="ja",
                            temperature=0.0, beam_size=1, condition_on_previous_text=True,
                            hallucination_silence_threshold=0.2, no_speech_threshold=0.3,
                            compression_ratio_threshold=2.4, logprob_threshold=-1.0
                        )
                        text = result["text"].strip()
                        print("🎌 Whisper-translated from Japanese:", text)
                    else:
                        # Transcribe English speech to English text
                        result = model.transcribe(
                            wav.name, fp16=True, task="transcribe", language="en",
                            temperature=0.0, beam_size=1, condition_on_previous_text=True,
                            hallucination_silence_threshold=0.2, no_speech_threshold=0.3,
                            compression_ratio_threshold=2.4, logprob_threshold=-1.0
                        )
                        text = result["text"].strip()
                        print("🇬🇧 Transcribed English:", text)

            # Clean up temp files to avoid accumulation
            try:
                os.remove(raw.name)
                os.remove(wav.name)
            except:
                pass

            if not text:
                continue

            # Skip polite/filler phrases to reduce clutter
            text_lower = text.lower()
            if any(x in text_lower for x in ["thank you", "thanks"]) or "ありがとう" in text:
                print("🚫 Skipping polite filler:", text)
                continue

            if await hallucination_check(text):
                print("🧠 Skipping hallucinated filler:", text)
                continue

            # Save the transcribed text (in original language or English if translated by Whisper)
            segment_id = str(uuid.uuid4())
            transcript_history.append((segment_id, text))

            # Translate the text to the target language
            translated = await translate_text(text, source_lang, target_lang)
            await websocket.send_text(f"[DONE]{json.dumps({'id': segment_id, 'text': translated})}")

            # If we have context (previous segment), refine the previous translation with new context
            if len(transcript_history) >= 2:
                prev_text = transcript_history[-2][1]
                curr_text = transcript_history[-1][1]
                improved_prev = await translate_text((prev_text, curr_text), source_lang, target_lang, mode="context")
                await websocket.send_text(f"[UPDATE]{json.dumps({'id': transcript_history[-2][0], 'text': improved_prev})}")

    except Exception as e:
        print("❌ WebSocket error:", e)
    finally:
        await websocket.close()

async def translate_text(text, source_lang, target_lang, mode="default"):
    try:
        # Step 1: Use GPT-4 (gpt-4o) for initial translation or refinement
        if mode == "context":
            previous, current = text
            prompt = (
                f"You are a strict, literal translation engine. Refine the translation of the previous sentence based on new context.\n\n"
                f"Previous: {previous}\n"
                f"Current: {current}\n\n"
                f"Rules:\n"
                f"- Do NOT repeat the previous sentence unless the meaning changes.\n"
                f"- Merge or rephrase ONLY if new information adds clarity.\n"
                f"- Return the improved translation of the 'Previous' sentence only.\n"
                f"- Do NOT repeat phrases that were already translated unless absolutely necessary.\n"
                f"- If a sentence is identical or nearly identical to the previous one, translate it only once.\n"
            )
        else:
            prompt = (
                f"You are a strict, literal translation engine. Translate the sentence below from {source_lang} to {target_lang}.\n\n"
                f"Rules:\n"
                f"- Return only the translated sentence. No commentary, no extra info.\n"
                f"- Never say 'already in English/Japanese'. If it's already translated, return it as-is.\n"
                f"- No thank-yous, greetings, or filler.\n\n"
                f"Sentence: {text}"
            )

        gpt_response = await client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a strict and literal translator."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=100
        )
        refined_text = gpt_response.choices[0].message.content.strip()

        # Step 2: Final polish with DeepSeek model
        instruction = f"Translate this from {source_lang} to {target_lang}: \"{refined_text}\""
        inputs = deepseek_tokenizer(instruction, return_tensors="pt").to(device)
        outputs = deepseek_model.generate(
            **inputs, max_new_tokens=256, do_sample=False, temperature=0.7,
            pad_token_id=deepseek_tokenizer.eos_token_id
        )
        output_text = deepseek_tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Remove the prompt from the output if present
        output_text = output_text.replace(instruction, "").strip()

        return output_text if output_text else refined_text

    except Exception as e:
        print("❌ Translation error:", e)
        return str(text)  # return original text on failure

async def hallucination_check(text):
    try:
        prompt = (
            "You are a strict hallucination filter. Determine if a sentence is a generic AI-generated filler phrase.\n\n"
            "Examples of filler:\n"
            "- 'Thanks for watching'\n"
            "- 'Don't forget to subscribe'\n"
            "- 'See you in the next video'\n\n"
            "ONLY answer 'YES' if the sentence is a known filler phrase, or 'NO' otherwise.\n\n"
            f"Sentence:\n{text}"
        )
        result = await client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "Respond only with YES or NO."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=1
        )
        reply = result.choices[0].message.content.strip()
        return reply.upper() == "YES"
    except:
        return False

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
