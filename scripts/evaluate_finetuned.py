import argparse
import sys
from pathlib import Path

import jiwer
import pandas as pd
import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import WhisperForConditionalGeneration

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import ModelConfig, PathConfig
from src.data.dataset import build_dataset
from src.data.features import feature_extractor, tokenizer
from src.transformation.normalization import normalize_arabic


def evaluate_finetuned(
    checkpoint_dir: str,
    dataset=None,
    model_config: ModelConfig | None = None,
    paths: PathConfig | None = None,
):
    paths = paths or PathConfig()
    model_config = model_config or ModelConfig()
    dataset = dataset or build_dataset()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    base_model = WhisperForConditionalGeneration.from_pretrained(
        model_config.model_identifier
    )
    model = PeftModel.from_pretrained(base_model, checkpoint_dir)
    model.generation_config.language = model_config.language
    model.generation_config.task = model_config.task
    model.generation_config.forced_decoder_ids = model_config.forced_decoder_ids
    model.to(device)
    model.eval()

    test_set = dataset["test"]

    rows = []
    for example in tqdm(test_set, desc="Evaluating finetuned model"):
        audio = example["audio"]
        inputs = feature_extractor(
            audio["array"], sampling_rate=audio["sampling_rate"], return_tensors="pt"
        )
        input_features = inputs.input_features.to(device)

        with torch.no_grad():
            predicted_ids = model.generate(input_features)
        prediction = tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)[0]

        rows.append(
            {
                "transcript": example["transcript"],
                "prediction": prediction.strip(),
                "speaker": example["speaker"],
            }
        )

    df = pd.DataFrame(rows)
    df["transcript_norm"] = df["transcript"].apply(normalize_arabic)
    df["prediction_norm"] = df["prediction"].apply(normalize_arabic)

    df["wer"] = [
        jiwer.wer(r, h) for r, h in zip(df.transcript_norm, df.prediction_norm)
    ]
    df["cer"] = [
        jiwer.cer(r, h) for r, h in zip(df.transcript_norm, df.prediction_norm)
    ]

    model_name = f"finetuned-{Path(checkpoint_dir).name}"
    df["model"] = model_name

    paths.output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(paths.output_dir / f"preds_{model_name}.csv", index=False)

    corpus_wer = jiwer.wer(list(df.transcript_norm), list(df.prediction_norm))
    corpus_cer = jiwer.cer(list(df.transcript_norm), list(df.prediction_norm))
    print(f"Corpus WER: {corpus_wer:.4f}")
    print(f"Corpus CER: {corpus_cer:.4f}")

    summary_path = paths.output_dir / "summary.csv"
    summary_row = pd.DataFrame(
        [
            {
                "model": model_name,
                "n_samples": len(df),
                "corpus_wer": corpus_wer,
                "corpus_cer": corpus_cer,
                "mean_row_wer": df["wer"].mean(),
                "mean_row_cer": df["cer"].mean(),
            }
        ]
    )
    summary_row.to_csv(
        summary_path, mode="a", header=not summary_path.exists(), index=False
    )

    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to a PEFT checkpoint dir, e.g. output/whisper-finetuned/checkpoint-35",
    )
    args = ap.parse_args()

    evaluate_finetuned(checkpoint_dir=args.checkpoint)
