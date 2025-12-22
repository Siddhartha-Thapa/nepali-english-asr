from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch
import json

def initialize_model():
    """
    Load pre-trained Wav2Vec2 model and prepare for fine-tuning
    """
    print("Initializing Wav2Vec2 Model...")
    print("=" * 60)
    
    # Load processor
    processor = Wav2Vec2Processor.from_pretrained('../models/processor')
    
    # Load vocabulary size
    with open('../data/vocab.json', 'r') as f:
        vocab_dict = json.load(f)
    vocab_size = len(vocab_dict)
    
    print(f"Vocabulary size: {vocab_size}")
    
    # Choose pre-trained model
    # Options:
    # 1. facebook/wav2vec2-base (95M params) - faster, less accurate
    # 2. facebook/wav2vec2-large-xlsr-53 (317M params) - RECOMMENDED for multilingual
    # 3. facebook/wav2vec2-xls-r-300m (300M params) - newer, good balance
    
    model_name = "facebook/wav2vec2-large-xlsr-53"
    print(f"\nLoading pre-trained model: {model_name}")
    
    model = Wav2Vec2ForCTC.from_pretrained(
        model_name,
        attention_dropout=0.1,
        hidden_dropout=0.1,
        feat_proj_dropout=0.0,
        mask_time_prob=0.05,
        layerdrop=0.1,
        ctc_loss_reduction="mean",
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=vocab_size,
        ignore_mismatched_sizes=True  # Important: allows vocab size mismatch
    )
    
    # Freeze feature encoder (only train CTC head initially)
    model.freeze_feature_encoder()
    
    print("✓ Model loaded successfully")
    print(f"✓ Feature encoder frozen")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nModel Statistics:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Frozen parameters: {total_params - trainable_params:,}")
    print(f"  Trainable percentage: {trainable_params/total_params*100:.2f}%")
    
    # Save initial model
    model_dir = '../models/initial_model'
    model.save_pretrained(model_dir)
    
    print(f"\n✓ Initial model saved to {model_dir}")
    print("=" * 60)
    
    return model, processor

if __name__ == "__main__":
    model, processor = initialize_model()