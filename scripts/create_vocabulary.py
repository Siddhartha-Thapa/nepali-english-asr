import json
import pandas as pd
from collections import Counter
from pathlib import Path

def create_vocabulary(transcript_column='romanized_transcript'):
    """
    Create character-level vocabulary from transcripts
    """
    print(f"Creating vocabulary from '{transcript_column}'...")
    print("=" * 60)
    
    # Load augmented metadata 
    if Path('../data/augmented_metadata.csv').exists():
        df = pd.read_csv('../data/augmented_metadata.csv')
        print("Using augmented dataset")
    else:
        df = pd.read_csv('data/metadata.csv')
        print("Using original dataset")
    
    # Extract all characters
    all_chars = []
    char_freq = Counter()
    
    for text in df[transcript_column]:
        if pd.notna(text):
            # Convert to lowercase
            text = text.lower()
            chars = list(text)
            all_chars.extend(chars)
            char_freq.update(chars)
    
    # Get unique characters
    unique_chars = sorted(set(all_chars))
    
    print(f"\nTotal characters in corpus: {len(all_chars)}")
    print(f"Unique characters: {len(unique_chars)}")
    
    # Create vocabulary dictionary
    vocab = {}
    for idx, char in enumerate(unique_chars):
        vocab[char] = idx
    
    # Add special tokens
    special_tokens = {
        '[PAD]': len(vocab),
        '[UNK]': len(vocab) + 1,
        '[CTC]': len(vocab) + 2,  # CTC blank token
    }
    
    vocab.update(special_tokens)
    
    # Save vocabulary
    output_path = '../data/vocab.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    
    print(f"\n✓ Vocabulary saved to {output_path}")
    print(f"✓ Vocabulary size: {len(vocab)}")
    
    # Print character frequency (top 20)
    print("\nTop 20 most frequent characters:")
    print("-" * 40)
    for char, freq in char_freq.most_common(20):
        char_display = repr(char) if char in [' ', '\n', '\t'] else char
        print(f"  {char_display:10} : {freq:6} times")
    
    # Print special tokens
    print("\nSpecial tokens:")
    print("-" * 40)
    for token, idx in special_tokens.items():
        print(f"  {token:10} : {idx}")
    
    # Save character frequency
    freq_df = pd.DataFrame([
        {'character': char, 'frequency': freq, 'percentage': freq/len(all_chars)*100}
        for char, freq in char_freq.most_common()
    ])
    freq_df.to_csv('../data/character_frequency.csv', index=False)
    
    print("\n✓ Character frequency saved to data/character_frequency.csv")
    print("=" * 60)
    
    return vocab

if __name__ == "__main__":
    vocab = create_vocabulary()
    
    # Display sample
    print("\nSample vocabulary entries:")
    sample_items = list(vocab.items())[:10]
    for char, idx in sample_items:
        print(f"  '{char}' -> {idx}")