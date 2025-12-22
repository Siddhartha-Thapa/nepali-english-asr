from evaluate import load
import torch
import numpy as np

# Load WER metric
wer_metric = load("wer")
cer_metric = load("cer")

def compute_metrics(pred, processor):
    """
    Compute Word Error Rate (WER) and Character Error Rate (CER)
    """
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)
    
    # Replace -100 with pad_token_id
    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id
    
    # Decode predictions and labels
    pred_str = processor.batch_decode(pred_ids)
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)
    
    # Compute metrics
    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    cer = cer_metric.compute(predictions=pred_str, references=label_str)
    
    return {
        "wer": wer,
        "cer": cer
    }