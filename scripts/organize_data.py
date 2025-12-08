# Create: scripts/organize_data.py
import os
import shutil
from pathlib import Path

def organize_audio_files():
    """
    Organize your 500 audio files with their transcripts
    
    Expected input format:
    - You have 500 .wav audio files
    - You have corresponding transcripts (one per audio file)
    """
    
    # Define paths
    raw_audio_dir = "data/raw_audio"
    romanized_dir = "data/transcripts/romanized"
    bilingual_dir = "data/transcripts/bilingual"
    
    print("Data Organization Helper")
    print("=" * 60)
    print("\nExpected structure:")
    print("  data/raw_audio/")
    print("    - audio_001.wav")
    print("    - audio_002.wav")
    print("    - ...")
    print("\n  data/transcripts/romanized/")
    print("    - audio_001.txt (content: 'ma dherai happy chu')")
    print("    - audio_002.txt")
    print("    - ...")
    print("\n  data/transcripts/bilingual/")
    print("    - audio_001.txt (content: 'म धेरै happy छु')")
    print("    - audio_002.txt")
    print("    - ...")
    print("=" * 60)
    
    # Check if files exist
    audio_files = list(Path(raw_audio_dir).glob("*.wav"))
    print(f"\n✓ Found {len(audio_files)} audio files")
    
    romanized_files = list(Path(romanized_dir).glob("*.txt"))
    print(f"✓ Found {len(romanized_files)} romanized transcripts")
    
    bilingual_files = list(Path(bilingual_dir).glob("*.txt"))
    print(f"✓ Found {len(bilingual_files)} bilingual transcripts")
    
    # Verify matching
    if len(audio_files) == len(romanized_files):
        print("\n✓ Audio and transcript counts match!")
    else:
        print("\n⚠️  WARNING: Mismatch in file counts!")
        
    return len(audio_files)

if __name__ == "__main__":
    organize_audio_files()