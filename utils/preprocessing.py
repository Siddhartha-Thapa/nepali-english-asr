from transformers import Wav2Vec2Processor
import torch

def prepare_dataset(batch, processor):
    """
    Preprocess batch for training
    """
    # Get audio
    audio = batch["audio"]
    
    # Process audio to input values
    batch["input_values"] = processor(
        audio["array"],
        sampling_rate=audio["sampling_rate"]
    ).input_values[0]
    
    # Encode text labels
    with processor.as_target_processor():
        batch["labels"] = processor(batch["romanized_transcript"]).input_ids
    
    return batch

def preprocess_datasets(dataset_dict, processor, num_proc=4):
    """
    Apply preprocessing to all dataset splits
    """
    print("Preprocessing datasets...")
    
    # Columns to remove after preprocessing
    remove_columns = dataset_dict["train"].column_names
    
    # Apply preprocessing
    processed_dataset = dataset_dict.map(
        lambda batch: prepare_dataset(batch, processor),
        remove_columns=remove_columns,
        num_proc=num_proc,
        desc="Preprocessing datasets"
    )
    
    print("✓ Preprocessing complete")
    
    return processed_dataset