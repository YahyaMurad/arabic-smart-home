from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class PathConfig:
    root: Path = ROOT
    data_dir: Path = ROOT / "data"
    raw_audio_dir: Path = ROOT / "data" / "raw"
    recordings_csv: Path = ROOT / "data" / "recordings.csv"
    output_dir: Path = ROOT / "output"
    data_collection_dir: Path = ROOT / "data_collection"
    prompts_config: Path = ROOT / "data_collection" / "config.yml"


@dataclass(frozen=True)
class ModelConfig:
    openai_tag: str = "openai/"
    model_name: str = "whisper-small"
    model_identifier: str = field(init=False)

    language: str = "arabic"
    task: str = "transcribe"
    forced_decoder_ids: list[str] = None

    def __post_init__(self):
        object.__setattr__(self, "model_identifier", self.openai_tag + self.model_name)

@dataclass(frozen=True)
class TrainingConfig:
    lora_r: int = 32
    lora_alpha: int = 64
    lora_target_modules: list[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    lora_dropout: float = 0.05

    output_dir: str = "output/whisper-finetuned"
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 2
    learning_rate: float = 1e-3
    num_train_epochs: int = 10
    eval_strategy: str = "epoch"
    save_strategy: str = "epoch"
    logging_steps: int = 5
    predict_with_generate: bool = True
    generation_max_length: int = 128
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "wer"
    greater_is_better: bool = False
    fp16: bool = True