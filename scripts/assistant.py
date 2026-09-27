import queue
import sys
import threading
import time
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf
import torch
from openwakeword.model import Model as WakeWordModel
from peft import PeftModel
from transformers import WhisperForConditionalGeneration

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.audio.vad import VoiceActivityDetector
from src.config import ModelConfig, PathConfig
from src.data.features import feature_extractor, tokenizer
from src.nlu.keyword_matcher import KeywordMatcher

SAMPLE_RATE = 16000
CHUNK_SIZE = 512  # Silero
WAKE_WORD_THRESHOLD = 0.5
WAKE_WORD = "alexa"


def run_assistant_loop(
    model_config: ModelConfig | None = None,
    paths: PathConfig | None = None,
):
    vad = VoiceActivityDetector()
    model_config = model_config or ModelConfig()
    paths = paths or PathConfig()
    debug_dir = paths.output_dir / "debug_utterances"
    debug_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    base_model = WhisperForConditionalGeneration.from_pretrained(
        model_config.model_identifier
    )
    model = PeftModel.from_pretrained(
        base_model, "output/whisper-finetuned/checkpoint-35"
    )
    model.generation_config.language = model_config.language
    model.generation_config.task = model_config.task
    model.generation_config.forced_decoder_ids = model_config.forced_decoder_ids
    model.to(device)
    model.eval()

    wake_word_model = WakeWordModel()
    matcher = KeywordMatcher()

    utterance_queue: queue.Queue = queue.Queue()
    listening_for_command = False

    def callback(indata, frames, time, status):
        nonlocal listening_for_command
        chunk = indata[:, 0]

        if not listening_for_command:
            chunk_int16 = (chunk * 32767).astype(np.int16)
            prediction = wake_word_model.predict(chunk_int16)
            if prediction[WAKE_WORD] >= WAKE_WORD_THRESHOLD:
                print(f"Wake word detected ({prediction[WAKE_WORD]:.2f})")
                listening_for_command = True
                vad.reset()
            return

        utterance = vad.process_chunk(chunk)
        if utterance is not None:
            listening_for_command = False
            utterance_queue.put(utterance)

    def worker():
        while True:
            utterance = utterance_queue.get()
            ts = int(time.time() * 1000)
            debug_path = debug_dir / f"utterance_{ts}.wav"
            sf.write(debug_path, utterance, SAMPLE_RATE)
            print(
                f"Saved utterance: {debug_path} ({len(utterance) / SAMPLE_RATE:.2f}s)"
            )

            inputs = feature_extractor(
                utterance, sampling_rate=SAMPLE_RATE, return_tensors="pt"
            )
            input_features = inputs.input_features.to(device)
            with torch.no_grad():
                predicted_ids = model.generate(
                    input_features,
                    repetition_penalty=1.3,
                    no_repeat_ngram_size=3,
                    max_new_tokens=64,
                )
            transcript = tokenizer.batch_decode(
                predicted_ids, skip_special_tokens=True
            )[0].strip()
            print(f"Transcript: {transcript}")

            action = matcher.match(transcript)
            print(f"Action: {action}" if action else "Action: no match")

    threading.Thread(target=worker, daemon=True).start()

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        blocksize=CHUNK_SIZE,
        callback=callback,
    ):
        print("Listening... Ctrl+C to stop.")
        try:
            while True:
                sd.sleep(1000)
        except KeyboardInterrupt:
            print("Stopped.")


if __name__ == "__main__":
    run_assistant_loop()
