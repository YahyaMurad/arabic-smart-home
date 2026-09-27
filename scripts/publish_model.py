import argparse
import sys
from pathlib import Path

from peft import PeftModel
from transformers import WhisperForConditionalGeneration

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import ModelConfig
from src.data.features import feature_extractor, tokenizer


def publish_model(
    checkpoint_dir: str,
    repo_id: str,
    model_config: ModelConfig | None = None,
):
    model_config = model_config or ModelConfig()

    base_model = WhisperForConditionalGeneration.from_pretrained(
        model_config.model_identifier
    )
    best_model = PeftModel.from_pretrained(base_model, checkpoint_dir)
    merged_model = best_model.merge_and_unload()

    merged_model.generation_config.language = model_config.language
    merged_model.generation_config.task = model_config.task
    merged_model.generation_config.forced_decoder_ids = model_config.forced_decoder_ids

    print(f"Publishing to https://huggingface.co/{repo_id}")
    merged_model.push_to_hub(repo_id)
    feature_extractor.push_to_hub(repo_id)
    tokenizer.push_to_hub(repo_id)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to a PEFT checkpoint dir, e.g. output/whisper-finetuned/checkpoint-35",
    )
    ap.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="HF Hub repo to publish to, e.g. your-username/whisper-small-arabic-smart-home",
    )
    args = ap.parse_args()

    publish_model(checkpoint_dir=args.checkpoint, repo_id=args.repo_id)
