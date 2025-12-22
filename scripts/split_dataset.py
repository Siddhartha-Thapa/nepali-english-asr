# Create: scripts/split_dataset.py
import pandas as pd
from sklearn.model_selection import train_test_split
import json

def split_dataset(test_size=0.1, val_size=0.1, random_state=42):
    """
    Split dataset into train/validation/test sets
    
    With 3000 augmented samples:
    - Train: ~2400 (80%)
    - Validation: ~300 (10%)
    - Test: ~300 (10%)
    """
    print("Splitting dataset...")
    print("=" * 60)
    
    # Load data
   
    df = pd.read_csv('../data/augmented_metadata.csv')
    print("Using augmented dataset")

    
    # First split: separate test set
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        shuffle=True
    )
    
    # Second split: separate validation from training
    val_size_adjusted = val_size / (1 - test_size)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_size_adjusted,
        random_state=random_state,
        shuffle=True
    )
    
    # Save splits
    train_df.to_csv('../data/train.csv', index=False)
    val_df.to_csv('../data/validation.csv', index=False)
    test_df.to_csv('../data/test.csv', index=False)
    
    # Create split info
    split_info = {
        'total_samples': len(df),
        'train_samples': len(train_df),
        'validation_samples': len(val_df),
        'test_samples': len(test_df),
        'train_duration': float(train_df['duration'].sum()),
        'validation_duration': float(val_df['duration'].sum()),
        'test_duration': float(test_df['duration'].sum()),
        'random_state': random_state
    }
    
    with open('../data/split_info.json', 'w') as f:
        json.dump(split_info, f, indent=2)
    
    # Print statistics
    print("\nDataset Split Summary:")
    print("-" * 60)
    print(f"Total samples: {len(df)}")
    print(f"\nTrain set:")
    print(f"  Samples: {len(train_df)} ({len(train_df)/len(df)*100:.1f}%)")
    print(f"  Duration: {train_df['duration'].sum()/60:.2f} minutes")
    
    print(f"\nValidation set:")
    print(f"  Samples: {len(val_df)} ({len(val_df)/len(df)*100:.1f}%)")
    print(f"  Duration: {val_df['duration'].sum()/60:.2f} minutes")
    
    print(f"\nTest set:")
    print(f"  Samples: {len(test_df)} ({len(test_df)/len(df)*100:.1f}%)")
    print(f"  Duration: {test_df['duration'].sum()/60:.2f} minutes")
    
    print("\n✓ Split files saved:")
    print("  - data/train.csv")
    print("  - data/validation.csv")
    print("  - data/test.csv")
    print("  - data/split_info.json")
    print("=" * 60)
    
    return train_df, val_df, test_df

if __name__ == "__main__":
    from pathlib import Path
    split_dataset()