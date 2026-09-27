import numpy as np
from silero_vad import VADIterator, load_silero_vad

SAMPLE_RATE = 16000
CHUNK_SIZE = 512  # Silero

class VoiceActivityDetector:
    def __init__(
        self,
        threshold: float = 0.6,
        min_silence_duration_ms: int = 450,
        speech_pad_ms: int = 150,
    ):
        model = load_silero_vad()
        self.vad_iterator = VADIterator(
            model,
            sampling_rate=SAMPLE_RATE,
            threshold=threshold,
            min_silence_duration_ms=min_silence_duration_ms,
            speech_pad_ms=speech_pad_ms,
        )
        self.buffer = []
        self.speaking = False

    def reset(self):
        self.vad_iterator.reset_states()
        self.buffer = []
        self.speaking = False

    def process_chunk(self, chunk: np.ndarray):
        event = self.vad_iterator(chunk)

        if event and "start" in event:
            self.speaking = True

        if self.speaking:
            self.buffer.append(chunk.copy())

        if event and "end" in event:
            utterance = np.concatenate(self.buffer)
            self.buffer = []
            self.speaking = False

            return utterance

        return None


if __name__ == "__main__":
    import sounddevice as sd

    vad = VoiceActivityDetector()

    def callback(indata, frames, time, status):
        chunk = indata[:, 0]
        utterance = vad.process_chunk(chunk)
        if utterance is not None:
            print(f"Utterance detected: {len(utterance) / SAMPLE_RATE:.2f}s")

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
