import argparse
import sys
from pathlib import Path

from datasets import DatasetDict
from peft import LoraConfig, PeftModel, get_peft_model
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
    repo_id: str | None = None,
):
    dataset = dataset or build_dataset()
    model_config = model_config or ModelConfig()
    training_config = training_config or TrainingConfig()

    dataset = dataset.map(
        prepare_features, remove_columns=dataset.column_names["train"]
    )

    base_model = WhisperForConditionalGeneration.from_pretrained(
        model_config.model_identifier
    )
    base_model.generation_config.language = model_config.language
    base_model.generation_config.task = model_config.task
    base_model.generation_config.forced_decoder_ids = model_config.forced_decoder_ids

    lora_config = LoraConfig(
        r=training_config.lora_r,
        lora_alpha=training_config.lora_alpha,
        target_modules=training_config.lora_target_modules,
        lora_dropout=training_config.lora_dropout,
    )

    lora_model = get_peft_model(base_model, lora_config)

    training_args_dict = {
        k: v for k, v in training_config.__dict__.items() if not k.startswith("lora_")
    }
    args = Seq2SeqTrainingArguments(**training_args_dict)

    trainer = Seq2SeqTrainer(
        args=args,
        model=lora_model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=data_collator,
        compute_metrics=build_compute_metrics(tokenizer),
        processing_class=feature_extractor,
    )

    trainer.train()

    if repo_id is None:
        return

    best_checkpoint = trainer.state.best_model_checkpoint
    print(f"Merging best checkpoint: {best_checkpoint}")

    fresh_base_model = WhisperForConditionalGeneration.from_pretrained(
        model_config.model_identifier
    )
    best_model = PeftModel.from_pretrained(fresh_base_model, best_checkpoint)
    merged_model = best_model.merge_and_unload()

    merged_model.generation_config.language = model_config.language
    merged_model.generation_config.task = model_config.task
    merged_model.generation_config.forced_decoder_ids = (
        model_config.forced_decoder_ids
    )

    print(f"Publishing to https://huggingface.co/{repo_id}")
    merged_model.push_to_hub(repo_id)
    feature_extractor.push_to_hub(repo_id)
    tokenizer.push_to_hub(repo_id)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--repo-id",
        type=str,
        default=None,
        help="If set, merge the best checkpoint and publish it to this HF Hub repo (e.g. your-username/whisper-small-arabic-smart-home)",
    )
    args = ap.parse_args()

    train(repo_id=args.repo_id)
