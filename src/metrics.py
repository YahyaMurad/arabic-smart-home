import jiwer

from src.transformation.normalization import normalize_arabic


def build_compute_metrics(tokenizer):
    def compute_metrics(pred):
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = tokenizer.pad_token_id

        pred_str = tokenizer.batch_decode(pred.predictions, skip_special_tokens=True)
        label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        pred_norm = [normalize_arabic(p) for p in pred_str]
        label_norm = [normalize_arabic(l) for l in label_str]

        return {
            "wer": jiwer.wer(label_norm, pred_norm),
            "cer": jiwer.cer(label_norm, pred_norm),
        }

    return compute_metrics
