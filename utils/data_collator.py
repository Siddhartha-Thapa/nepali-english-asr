from dataclasses import dataclass
from typing import Dict, List, Union
import torch

@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator for CTC training
    Pads inputs and labels dynamically
    """
    processor: any
    padding: Union[bool, str] = True
    max_length: Union[int, None] = None
    max_length_labels: Union[int, None] = None
    pad_to_multiple_of: Union[int, None] = None
    pad_to_multiple_of_labels: Union[int, None] = None
    
    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # Separate inputs and labels
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        
        # Pad inputs
        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        
        # Pad labels
        labels_batch = self.processor.pad(
            labels=label_features,
            padding=self.padding,
            max_length=self.max_length_labels,
            pad_to_multiple_of=self.pad_to_multiple_of_labels,
            return_tensors="pt",
        )
        
        # Replace padding with -100 (ignored by loss function)
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        
        batch["labels"] = labels
        
        # Add attention mask if present
        if "attention_mask" in batch:
            batch["attention_mask"] = batch["attention_mask"]
        
        return batch