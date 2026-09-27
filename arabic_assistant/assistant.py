import queue
import threading

import sounddevice as sd

from arabic_assistant.config import AssistantConfig
from arabic_assistant.listener import CHUNK_SIZE, SAMPLE_RATE, CommandListener
from arabic_assistant.nlu.keyword_matcher import KeywordMatcher
from arabic_assistant.sinks.base import Sink
from arabic_assistant.sinks.stdout_sink import StdoutSink
from arabic_assistant.transcriber import Transcriber


class Assistant:
    def __init__(self, config: AssistantConfig, sink: Sink | None = None):
        self.config = config

        self.listener = CommandListener(
            wake_word=config.wake_word,
            wake_word_threshold=config.wake_word_threshold,
        )
        self.transcriber = Transcriber(
            repo_id=config.model_repo_id,
            repetition_penalty=config.repetition_penalty,
            no_repeat_ngram_size=config.no_repeat_ngram_size,
            max_new_tokens=config.max_new_tokens,
        )
        self.matcher = KeywordMatcher()
        self.sink = sink or StdoutSink()

        self.utterance_queue: queue.Queue = queue.Queue()

    def run(self):
        threading.Thread(target=self._worker_loop, daemon=True).start()

        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=CHUNK_SIZE,
            callback=self._on_audio_chunk,
        ):
            print("Ready.")
            try:
                while True:
                    sd.sleep(1000)
            except KeyboardInterrupt:
                print("Stopped.")

    def _on_audio_chunk(self, indata, frames, time, status):
        chunk = indata[:, 0].copy()
        utterance = self.listener.process_chunk(chunk)
        if utterance is not None:
            self.utterance_queue.put(utterance)

    def _worker_loop(self):
        while True:
            utterance = self.utterance_queue.get()
            transcript = self.transcriber.transcribe(utterance)
            print(f"Transcript: {transcript}")

            action = self.matcher.match(transcript)
            if action is None:
                print("Sorry, I didn't get that.")
                continue

            self.sink.send(action)
