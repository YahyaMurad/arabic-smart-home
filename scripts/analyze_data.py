import argparse
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.audio.stats import clipping_ratio, load_audio, peak_amplitude, silence_ratio
from src.config import PathConfig


def analyze_data(paths: PathConfig | None = None, limit: int | None = None):
    paths = paths or PathConfig()
    paths.output_dir.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(paths.recordings_csv)
    if limit is not None:
        data = data.head(limit)

    peaks = []
    clip_ratios = []
    silence_ratios = []
    for item in tqdm(data.itertuples(), total=len(data), desc="Analyzing Audio"):
        audio, sample_rate = load_audio(paths.data_dir / item.file)
        peaks.append(peak_amplitude(audio))
        clip_ratios.append(clipping_ratio(audio))
        silence_ratios.append(silence_ratio(audio))

    data["peak"] = peaks
    data["clipping_ratio"] = clip_ratios
    data["silence_ratio"] = silence_ratios
    data.to_csv(paths.output_dir / "audio_analysis.csv", index=False)

    n_any_clipping = (data["clipping_ratio"] > 0).sum()
    n_sustained_clipping = (data["clipping_ratio"] > 0.001).sum()
    print(f"Files with any clipped samples: {n_any_clipping} / {len(data)}")
    print(
        f"Files with sustained clipping (>0.1% of samples): {n_sustained_clipping} / {len(data)}"
    )

    n_excessive_silence = (data["silence_ratio"] > 0.9).sum()
    print(
        f"Files with excessive silence (>0.1% of samples): {n_excessive_silence} / {len(data)}"
    )

    return data


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--limit", type=int, default=None, help="Evaluate on first N rows only"
    )
    args = ap.parse_args()
    analyze_data(limit=args.limit)
