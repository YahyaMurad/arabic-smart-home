import pandas as pd
from datasets import Audio, Dataset

from src.config import PathConfig


def build_dataset(paths: PathConfig | None = None, test_size: float = 0.15):
    paths = paths or PathConfig()
    df = pd.read_csv(paths.recordings_csv)
    df["audio"] = df["file"].apply(lambda f: str(paths.data_dir / f))

    cols = ["audio", "transcript", "speaker", "dialect", "distance", "noise"]
    ds = Dataset.from_dict(df[cols].to_dict(orient="list"))
    ds = ds.cast_column("audio", Audio(sampling_rate=16000))
    return ds.train_test_split(test_size=test_size, seed=42)
