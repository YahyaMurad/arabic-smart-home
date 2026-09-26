import argparse
import sys
from pathlib import Path

from datasets import DatasetDict
from peft import LoraConfig, get_peft_model
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import ModelConfig, TrainingConfig
from src.data.collator import DataCollatorSpeechSeq2SeqWithPadding
from src.data.dataset import build_dataset
from src.data.features import feature_extractor, prepare_features, tokenizer
from src.metrics import build_compute_metrics

data_collator = DataCollatorSpeechSeq2SeqWithPadding(
    feature_extractor=feature_extractor,
    tokenizer=tokenizer,
)


def train(
    dataset: DatasetDict | None = None,
    model_config: ModelConfig | None = None,
    training_config: TrainingConfig | None = None,
):
    dataset = dataset or build_dataset()
    model_config = model_config or ModelConfig()
    training_config = training_config or TrainingConfig()

    dataset = dataset.map(
        prepare_features, remove_columns=dataset.column_names["train"]
    )

    model = WhisperForConditionalGeneration.from_pretrained(
        model_config.model_identifier
    )
    model.generation_config.language = model_config.language
    model.generation_config.task = model_config.task
    model.generation_config.forced_decoder_ids = model_config.forced_decoder_ids

    lora_config = LoraConfig(
        r=training_config.lora_r,
        lora_alpha=training_config.lora_alpha,
        target_modules=training_config.lora_target_modules,
        lora_dropout=training_config.lora_dropout,
    )

    model = get_peft_model(model, lora_config)

    training_args_dict = {
        k: v for k, v in training_config.__dict__.items() if not k.startswith("lora_")
    }
    args = Seq2SeqTrainingArguments(**training_args_dict)

    trainer = Seq2SeqTrainer(
        args=args,
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=data_collator,
        compute_metrics=build_compute_metrics(tokenizer),
        processing_class=feature_extractor,
    )

    trainer.train()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    args = ap.parse_args()

    train()
