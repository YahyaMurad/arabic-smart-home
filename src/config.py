from dataclasses import dataclass
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
