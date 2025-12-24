import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import librosa
import sys
from pathlib import Path

def load_trained_model():
    base_dir = Path(__file__).resolve().parent.parent
    model_path = base_dir / "models" / "final_model"

    print(f"Loading model from {model_path}")

    processor = Wav2Vec2Processor.from_pretrained(str(model_path))
    model = Wav2Vec2ForCTC.from_pretrained(str(model_path))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    return model, processor, device

def transcribe_audio(audio_path, model, processor, device):
    """Transcribe a single audio file"""
    # Load audio
    audio, sr = librosa.load(audio_path, sr=16000)
    
    # Process audio
    inputs = processor(
        audio,
        sampling_rate=16000,
        return_tensors="pt",
        padding=True
    )
    
    input_values = inputs.input_values.to(device)
    attention_mask = inputs.attention_mask.to(device) if "attention_mask" in inputs else None
    
    # Get predictions
    with torch.no_grad():
        if attention_mask is not None:
            logits = model(input_values, attention_mask=attention_mask).logits
        else:
            logits = model(input_values).logits
    
    # Decode
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]
    
    return transcription

def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/inference.py <audio_file.wav>")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    
    # Load model
    model, processor, device = load_trained_model()
    
    # Transcribe
    print(f"\nTranscribing: {audio_file}")
    result = transcribe_audio(audio_file, model, processor, device)
    
    print(f"\nTranscription: {result}")

if __name__ == "__main__":
    main()