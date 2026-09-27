from enum import Enum, auto

import numpy as np
from openwakeword.model import Model as WakeWordModel

from arabic_assistant.vad import VoiceActivityDetector

SAMPLE_RATE = 16000
CHUNK_SIZE = 512  # Silero
GUARD_DURATION_MS = 300


class ListenerState(Enum):
    IDLE = auto()
    WAKE_WORD_DETECTED = auto()
    LISTENING = auto()


class CommandListener:
    def __init__(
        self,
        wake_word: str,
        wake_word_threshold: float = 0.5,
        vad: VoiceActivityDetector | None = None,
    ):
        self.wake_word = wake_word
        self.wake_word_threshold = wake_word_threshold
        self.wake_word_model = WakeWordModel()
        self.vad = vad or VoiceActivityDetector()

        self.state = ListenerState.IDLE
        self.guard_chunks_remaining = 0

    def process_chunk(self, chunk: np.ndarray) -> np.ndarray | None:
        chunk_int16 = (chunk * 32767).astype(np.int16)
        prediction = self.wake_word_model.predict(chunk_int16)

        if self.state == ListenerState.IDLE:
            self._check_wake_word(prediction)
            return None

        if self.state == ListenerState.WAKE_WORD_DETECTED:
            self._wait_out_guard()
            return None

        return self._listen(chunk)

    def _wait_out_guard(self) -> None:
        self.guard_chunks_remaining -= 1
        if self.guard_chunks_remaining <= 0:
            self.vad.reset()
            self.state = ListenerState.LISTENING

    def _listen(self, chunk: np.ndarray) -> np.ndarray | None:
        utterance = self.vad.process_chunk(chunk)
        if utterance is None:
            return None

        print("Processing...")
        self.state = ListenerState.IDLE
        return utterance

    def _check_wake_word(self, prediction: dict) -> None:
        score = prediction[self.wake_word]
        if score >= self.wake_word_threshold:
            print(f"Listening... (score={score:.2f})")
            self.state = ListenerState.WAKE_WORD_DETECTED
            self.guard_chunks_remaining = int(
                GUARD_DURATION_MS / 1000 * SAMPLE_RATE / CHUNK_SIZE
            )
