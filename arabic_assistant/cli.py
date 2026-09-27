import argparse
from pathlib import Path

from .assistant import Assistant
from .config import AssistantConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Arabic smart-home voice assistant")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to a config YAML file (default: ./assistant.yml if it exists)",
    )
    parser.add_argument("--model-repo-id", type=str, default=None)
    parser.add_argument("--wake-word", type=str, default=None)
    parser.add_argument("--wake-word-threshold", type=float, default=None)
    parser.add_argument("--vad-threshold", type=float, default=None)
    parser.add_argument("--vad-min-silence-duration-ms", type=int, default=None)
    parser.add_argument("--vad-speech-pad-ms", type=int, default=None)
    parser.add_argument("--repetition-penalty", type=float, default=None)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    config = AssistantConfig.load(args.config)

    overrides = {
        "model_repo_id": args.model_repo_id,
        "wake_word": args.wake_word,
        "wake_word_threshold": args.wake_word_threshold,
        "vad_threshold": args.vad_threshold,
        "vad_min_silence_duration_ms": args.vad_min_silence_duration_ms,
        "vad_speech_pad_ms": args.vad_speech_pad_ms,
        "repetition_penalty": args.repetition_penalty,
        "no_repeat_ngram_size": args.no_repeat_ngram_size,
        "max_new_tokens": args.max_new_tokens,
    }
    for field, value in overrides.items():
        if value is not None:
            setattr(config, field, value)

    Assistant(config).run()
