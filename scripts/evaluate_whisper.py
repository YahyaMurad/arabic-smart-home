import argparse
import sys
from pathlib import Path

import jiwer
import pandas as pd
import whisper
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import PathConfig
from src.transformation.normalization import normalize_arabic


def evaluate_whisper(
    model_name: str = "base", paths: PathConfig | None = None, limit: int | None = None
):
    paths = paths or PathConfig()
    paths.output_dir.mkdir(parents=True, exist_ok=True)

    model = whisper.load_model(model_name)

    data = pd.read_csv(paths.recordings_csv)
    if limit is not None:
        data = data.head(limit)

    results = []
    for item in tqdm(data.itertuples(), total=len(data), desc="Whisper Evaluation"):
        file_path = paths.data_dir / item.file
        model_transcription = model.transcribe(str(file_path), language="ar")["text"]

        results.append(model_transcription.strip())

    data["whisper_results"] = results
    data["model"] = model_name

    transcript_norm = data["transcript"].apply(normalize_arabic)
    whisper_results_norm = data["whisper_results"].apply(normalize_arabic)

    data["wer"] = [
        jiwer.wer(r, h) for r, h in zip(transcript_norm, whisper_results_norm)
    ]
    data["cer"] = [
        jiwer.cer(r, h) for r, h in zip(transcript_norm, whisper_results_norm)
    ]
    data.to_csv(paths.output_dir / f"preds_whisper_{model_name}.csv", index=False)

    corpus_wer = jiwer.wer(list(transcript_norm), list(whisper_results_norm))
    corpus_cer = jiwer.cer(list(transcript_norm), list(whisper_results_norm))
    print(f"Corpus WER: {corpus_wer:.4f}")
    print(f"Corpus CER: {corpus_cer:.4f}")

    summary_path = paths.output_dir / "summary.csv"
    summary_row = pd.DataFrame(
        [
            {
                "model": model_name,
                "n_samples": len(data),
                "corpus_wer": corpus_wer,
                "corpus_cer": corpus_cer,
                "mean_row_wer": data["wer"].mean(),
                "mean_row_cer": data["cer"].mean(),
            }
        ]
    )
    summary_row.to_csv(
        summary_path, mode="a", header=not summary_path.exists(), index=False
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model",
        type=str,
        nargs="+",
        default=["base"],
        choices=["tiny", "base", "small", "medium", "large-v2", "large-v3", "turbo"],
        help="One or more Whisper models to evaluate",
    )
    ap.add_argument(
        "--limit", type=int, default=None, help="Evaluate on first N rows only"
    )
    args = ap.parse_args()

    for model_name in args.model:
        print(f"=== {model_name} ===")
        evaluate_whisper(model_name=model_name, limit=args.limit)
