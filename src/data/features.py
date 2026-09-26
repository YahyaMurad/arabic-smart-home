from transformers import WhisperFeatureExtractor, WhisperTokenizer

from src.config import ModelConfig

model = ModelConfig()

feature_extractor = WhisperFeatureExtractor.from_pretrained(model.model_identifier)
tokenizer = WhisperTokenizer.from_pretrained(
    model.model_identifier, language="Arabic", task="transcribe"
)

def prepare_features(batch):
    audio = batch["audio"]
    batch["input_features"] = feature_extractor(
        audio["array"], sampling_rate=audio["sampling_rate"]
    ).input_features[0]
    batch["labels"] = tokenizer(batch["transcript"]).input_ids
    return batch
