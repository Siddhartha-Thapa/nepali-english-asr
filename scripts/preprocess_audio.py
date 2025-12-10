# Create: scripts/preprocess_audio.py
import os
import librosa
import soundfile as sf
import numpy as np
from pathlib import Path
from tqdm import tqdm
import pandas as pd

def validate_audio(audio_path):
    """Check if audio file is valid"""
    try:
        y, sr = librosa.load(audio_path, sr=None)
        
        # Checks
        if len(y) == 0:
            return False, "Empty audio"
        if sr not in [8000, 16000, 22050, 44100, 48000]:
            return False, f"Unusual sample rate: {sr}"
        if np.max(np.abs(y)) == 0:
            return False, "Silent audio"
            
        return True, "Valid"
    except Exception as e:
        return False, str(e)

def preprocess_audio_file(input_path, output_path, target_sr=16000):
    """
    Preprocess single audio file:
    - Resample to 16kHz (required by wav2vec2)
    - Normalize volume
    - Convert to mono if stereo
    """
    try:
        # Load audio
        y, sr = librosa.load(input_path, sr=None, mono=True)
        
        # Resample to target sample rate
        if sr != target_sr:
            y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        
        # Normalize to [-1, 1]
        if np.max(np.abs(y)) > 0:
            y = y / np.max(np.abs(y))
        
        # Remove silence from beginning and end
        y_trimmed, _ = librosa.effects.trim(y, top_db=20)
        
        # Save processed audio
        sf.write(output_path, y_trimmed, target_sr)
        
        return True, len(y_trimmed) / target_sr
        
    except Exception as e:
        return False, str(e)

def preprocess_all_audio():
    """Preprocess all audio files"""
    print("Audio Preprocessing Pipeline")
    print("=" * 60)
    
    # Load metadata
    df = pd.read_csv('data/metadata.csv')
    
    output_dir = Path("data/processed_audio")
    output_dir.mkdir(exist_ok=True)
    
    results = []
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing audio"):
        audio_id = row['audio_id']
        input_path = row['audio_path']
        output_path = output_dir / f"{audio_id}.wav"
        
        # Validate
        is_valid, message = validate_audio(input_path)
        if not is_valid:
            print(f"\n⚠️  Invalid audio {audio_id}: {message}")
            results.append({
                'audio_id': audio_id,
                'status': 'invalid',
                'message': message
            })
            continue
        
        # Preprocess
        success, result = preprocess_audio_file(input_path, output_path)
        
        if success:
            results.append({
                'audio_id': audio_id,
                'status': 'success',
                'processed_duration': result
            })
        else:
            print(f"\n⚠️  Error processing {audio_id}: {result}")
            results.append({
                'audio_id': audio_id,
                'status': 'error',
                'message': result
            })
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv('data/preprocessing_results.csv', index=False)
    
    # Update metadata with processed paths
    df['processed_audio_path'] = df['audio_id'].apply(
        lambda x: str(output_dir / f"{x}.wav")
    )
    df.to_csv('data/metadata.csv', index=False)
    
    # Summary
    print("\n" + "=" * 60)
    print("PREPROCESSING SUMMARY")
    print("=" * 60)
    print(f"Total files: {len(results)}")
    print(f"Successful: {len([r for r in results if r['status'] == 'success'])}")
    print(f"Failed: {len([r for r in results if r['status'] != 'success'])}")
    print("=" * 60)

if __name__ == "__main__":
    preprocess_all_audio()