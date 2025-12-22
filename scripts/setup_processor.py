from transformers import (
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Processor
)
import json
from pathlib import Path


def setup_processor():
    """
    Create Wav2Vec2 processor (tokenizer + feature extractor)
    """
    print("Setting up Wav2Vec2 Processor...")
    print("=" * 60)

    # Load vocabulary
    vocab_path = Path("../data/vocab.json")
    with vocab_path.open("r", encoding="utf-8") as f:
        vocab_dict = json.load(f)

    print(f"✓ Loaded vocabulary: {len(vocab_dict)} tokens")

    # Create tokenizer
    tokenizer = Wav2Vec2CTCTokenizer(
        "../data/vocab.json",
        unk_token="[UNK]",
        pad_token="[PAD]",
        word_delimiter_token="|",
        do_lower_case=True
    )

    print("✓ Created tokenizer")

    # Create feature extractor
    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1,
        sampling_rate=16000,
        padding_value=0.0,
        do_normalize=True,
        return_attention_mask=True
    )

    print("✓ Created feature extractor")

    # Combine into processor
    processor = Wav2Vec2Processor(
        feature_extractor=feature_extractor,
        tokenizer=tokenizer
    )

    # Save processor
    processor_dir = Path("../models/processor")
    processor_dir.mkdir(parents=True, exist_ok=True)
    processor.save_pretrained(processor_dir)

    print(f"\n✓ Processor saved to {processor_dir}")
    print("=" * 60)

    return processor


if __name__ == "__main__":
    processor = setup_processor()
