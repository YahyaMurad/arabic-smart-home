import dataclasses
from dataclasses import dataclass
from pathlib import Path

import yaml

DEFAULT_CONFIG_PATH = Path("assistant.yml")


@dataclass
class AssistantConfig:
    model_repo_id: str = "YahyaMujahed/whisper-small-arabic-smart-home"
    wake_word: str = "alexa"
    wake_word_threshold: float = 0.5
    vad_threshold: float = 0.6
    vad_min_silence_duration_ms: int = 450
    vad_speech_pad_ms: int = 150
    repetition_penalty: float = 1.3
    no_repeat_ngram_size: int = 3
    max_new_tokens: int = 64

    @classmethod
    def load(cls, path: Path | None = None) -> "AssistantConfig":
        config = cls()

        path = path or DEFAULT_CONFIG_PATH
        if path.exists():
            overrides = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            config = dataclasses.replace(config, **overrides)

        return config
