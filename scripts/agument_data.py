import librosa
import soundfile as sf
import numpy as np
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import random

def time_stretch(y, rate):
    """Change speed of audio"""
    return librosa.effects.time_stretch(y, rate=rate)

def pitch_shift(y, sr, n_steps):
    """Shift pitch up or down"""
    return librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)

def add_noise(y, noise_factor):
    """Add random noise"""
    noise = np.random.randn(len(y))
    return y + noise_factor * noise

def augment_audio_file(input_path, output_dir, audio_id, sr=16000):
    """
    Create augmented versions of a single audio file
    Returns list of (output_path, augmentation_type)
    """
    # Load audio
    y, _ = librosa.load(input_path, sr=sr)
    
    augmentations = []
    
    # Original (no augmentation)
    original_path = output_dir / f"{audio_id}_original.wav"
    sf.write(original_path, y, sr)
    augmentations.append((str(original_path), "original"))
    
    # Time stretch variations
    for rate, suffix in [(0.9, "slow"), (1.1, "fast")]:
        aug_path = output_dir / f"{audio_id}_speed_{suffix}.wav"
        y_stretched = time_stretch(y, rate)
        sf.write(aug_path, y_stretched, sr)
        augmentations.append((str(aug_path), f"speed_{suffix}"))
    
    # Pitch shift variations
    for n_steps, suffix in [(-2, "lower"), (2, "higher")]:
        aug_path = output_dir / f"{audio_id}_pitch_{suffix}.wav"
        y_shifted = pitch_shift(y, sr, n_steps)
        sf.write(aug_path, y_shifted, sr)
        augmentations.append((str(aug_path), f"pitch_{suffix}"))
    
    # Add noise
    aug_path = output_dir / f"{audio_id}_noisy.wav"
    y_noisy = add_noise(y, noise_factor=0.005)
    # Normalize
    y_noisy = y_noisy / np.max(np.abs(y_noisy))
    sf.write(aug_path, y_noisy, sr)
    augmentations.append((str(aug_path), "noisy"))
    
    return augmentations

def create_augmented_dataset():
    """
    Create augmented versions of all audio files
    This will multiply your 500 samples to ~3000 samples
    """
    print("Data Augmentation Pipeline")
    print("=" * 60)
    
    # Load metadata
    df = pd.read_csv('../data/metadata.csv')
    
    # Create output directory
    augmented_dir = Path("../data/augmented_audio")
    augmented_dir.mkdir(exist_ok=True)
    
    augmented_metadata = []
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Augmenting audio"):
        audio_id = row['audio_id']
        input_path = row['processed_audio_path']
        romanized = row['romanized_transcript']
        bilingual = row['bilingual_transcript']
        
        # Create augmentations
        augmentations = augment_audio_file(
            input_path, 
            augmented_dir, 
            audio_id
        )
        
        # Add to metadata
        for aug_path, aug_type in augmentations:
            # Get duration
            y, sr = librosa.load(aug_path, sr=None)
            duration = len(y) / sr
            
            augmented_metadata.append({
                'audio_id': f"{audio_id}_{aug_type}",
                'original_audio_id': audio_id,
                'audio_path': aug_path,
                'romanized_transcript': romanized,
                'bilingual_transcript': bilingual,
                'duration': round(duration, 2),
                'augmentation_type': aug_type
            })
    
    # Save augmented metadata
    aug_df = pd.DataFrame(augmented_metadata)
    aug_df.to_csv('../data/augmented_metadata.csv', index=False)
    
    # Statistics
    print("\n" + "=" * 60)
    print("AUGMENTATION SUMMARY")
    print("=" * 60)
    print(f"Original samples: {len(df)}")
    print(f"Augmented samples: {len(aug_df)}")
    print(f"Multiplication factor: {len(aug_df) / len(df):.1f}x")
    print(f"Total duration: {aug_df['duration'].sum() / 60:.2f} minutes")
    print("=" * 60)
    
    return aug_df

if __name__ == "__main__":
    create_augmented_dataset()