import numpy as np
import torch
from transformers import (
    WhisperFeatureExtractor,
    WhisperForConditionalGeneration,
    WhisperTokenizer,
)
from transformers.utils import logging as hf_logging

hf_logging.set_verbosity_error()


class Transcriber:
    def __init__(
        self,
        repo_id: str,
        repetition_penalty: float = 1.3,
        no_repeat_ngram_size: int = 3,
        max_new_tokens: int = 64,
    ):
        self.repo_id = repo_id
        self.repetition_penalty = repetition_penalty
        self.no_repeat_ngram_size = no_repeat_ngram_size
        self.max_new_tokens = max_new_tokens

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.feature_extractor, self.tokenizer = self._load_model()

    def _load_model(self):
        model = WhisperForConditionalGeneration.from_pretrained(self.repo_id)
        model.to(self.device)
        model.eval()

        model.generation_config.max_new_tokens = self.max_new_tokens
        model.generation_config.repetition_penalty = self.repetition_penalty
        model.generation_config.no_repeat_ngram_size = self.no_repeat_ngram_size

        feature_extractor = WhisperFeatureExtractor.from_pretrained(self.repo_id)
        tokenizer = WhisperTokenizer.from_pretrained(self.repo_id)

        return model, feature_extractor, tokenizer

    def transcribe(self, audio: np.ndarray) -> str:
        inputs = self.feature_extractor(audio, sampling_rate=16000, return_tensors="pt")
        input_features = inputs.input_features.to(self.device)

        with torch.no_grad():
            predicted_ids = self.model.generate(input_features)

        transcript = self.tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        return transcript.strip()


if __name__ == "__main__":
    SAMPLE_RATE = 16000
    duration_s = 2.0
    frequency_hz = 440.0

    t = np.linspace(0, duration_s, int(SAMPLE_RATE * duration_s), endpoint=False)
    audio = (0.5 * np.sin(2 * np.pi * frequency_hz * t)).astype(np.float32)

    transcriber = Transcriber(repo_id="YahyaMujahed/whisper-small-arabic-smart-home")
    print(transcriber.transcribe(audio))
