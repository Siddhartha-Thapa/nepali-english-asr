import os
import pandas as pd
import librosa
from pathlib import Path
from tqdm import tqdm

def create_metadata_csv():
    """
    Create metadata.csv with information about all audio files
    """
    print("Creating metadata.csv...")
    
    raw_audio_dir = Path("../data/raw_audio")
    romanized_dir = Path("../data/transcripts/romanized")
    bilingual_dir = Path("../data/transcripts/bilingual")
    
    metadata = []
    
    # Get all audio files
    audio_files = sorted(raw_audio_dir.glob("*.wav"))
    
    for audio_path in tqdm(audio_files, desc="Processing files"):
        audio_id = audio_path.stem  # filename without extension
        
        # Get corresponding transcript files
        romanized_path = romanized_dir / f"{audio_id}.txt"
        bilingual_path = bilingual_dir / f"{audio_id}.txt"
        
        # Read transcripts
        try:
            with open(romanized_path, 'r', encoding='utf-8') as f:
                romanized_text = f.read().strip()
        except FileNotFoundError:
            print(f"⚠️  Missing romanized transcript for {audio_id}")
            romanized_text = ""
        
        try:
            with open(bilingual_path, 'r', encoding='utf-8') as f:
                bilingual_text = f.read().strip()
        except FileNotFoundError:
            print(f"⚠️  Missing bilingual transcript for {audio_id}")
            bilingual_text = ""
        
        # Get audio duration
        try:
            y, sr = librosa.load(audio_path, sr=None)
            duration = len(y) / sr
        except Exception as e:
            print(f"⚠️  Error loading {audio_id}: {e}")
            duration = 0.0
        
        metadata.append({
            'audio_id': audio_id,
            'audio_path': str(audio_path),
            'romanized_transcript': romanized_text,
            'bilingual_transcript': bilingual_text,
            'duration': round(duration, 2),
            'sample_rate': sr
        })
    
    # Create DataFrame
    df = pd.DataFrame(metadata)
    
    # Save to CSV
    df.to_csv('../data/metadata.csv', index=False)
    
    # Print statistics
    print("\n" + "=" * 60)
    print("DATASET STATISTICS")
    print("=" * 60)
    print(f"Total samples: {len(df)}")
    print(f"Total duration: {df['duration'].sum() / 60:.2f} minutes")
    print(f"Average duration: {df['duration'].mean():.2f} seconds")
    print(f"Min duration: {df['duration'].min():.2f} seconds")
    print(f"Max duration: {df['duration'].max():.2f} seconds")
    print(f"\nEmpty romanized transcripts: {(df['romanized_transcript'] == '').sum()}")
    print(f"Empty bilingual transcripts: {(df['bilingual_transcript'] == '').sum()}")
    print("=" * 60)
    
    # Show sample
    print("\nSample entries:")
    print(df.head())
    
    return df

if __name__ == "__main__":
    df = create_metadata_csv()