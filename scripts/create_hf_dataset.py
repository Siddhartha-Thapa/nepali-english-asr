import pandas as pd
from datasets import Dataset, DatasetDict, Audio, Features, Value
import json
from pathlib import Path

def create_huggingface_dataset():
    """
    Convert CSV files to HuggingFace Dataset format
    """
    print("Creating HuggingFace Dataset...")
    print("=" * 60)
    
    # Load CSV files
    train_df = pd.read_csv('../data/train.csv')
    val_df = pd.read_csv('../data/validation.csv')
    test_df = pd.read_csv('../data/test.csv')
    
    # Define features
    features = Features({
        'audio_id': Value('string'),
        'audio': Audio(sampling_rate=16000),
        'romanized_transcript': Value('string'),
        'bilingual_transcript': Value('string'),
        'duration': Value('float32'),
    })
    train_df = train_df.rename(columns={'audio_path': 'audio'})
    val_df   = val_df.rename(columns={'audio_path': 'audio'})
    test_df  = test_df.rename(columns={'audio_path': 'audio'})
    # Create datasets
    print("\nCreating train dataset...")
    train_dataset = Dataset.from_pandas(
        train_df[['audio_id', 'audio', 'romanized_transcript', 
                  'bilingual_transcript', 'duration']],
        features=features
    )

    
    print("Creating validation dataset...")
    val_dataset = Dataset.from_pandas(
        val_df[['audio_id', 'audio', 'romanized_transcript', 
                'bilingual_transcript', 'duration']],
        features=features
    )
  
    print("Creating test dataset...")
    test_dataset = Dataset.from_pandas(
        test_df[['audio_id', 'audio', 'romanized_transcript', 
                 'bilingual_transcript', 'duration']],
        features=features
    )

    # Combine into DatasetDict
    dataset_dict = DatasetDict({
        'train': train_dataset,
        'validation': val_dataset,
        'test': test_dataset
    })
    
    # Save to disk
    output_path = '../data/hf_dataset'
    dataset_dict.save_to_disk(output_path)
    
    print(f"\n✓ Dataset saved to {output_path}")
    print("\nDataset structure:")
    print(dataset_dict)
    
    print("\nSample from train set:")
    print(dataset_dict['train'][0])
    
    print("=" * 60)
    
    return dataset_dict

if __name__ == "__main__":
    dataset = create_huggingface_dataset()