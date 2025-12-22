import torch
from transformers import (
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from datasets import load_from_disk
import sys
import os

# Add utils to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.data_collator import DataCollatorCTCWithPadding
from utils.preprocessing import preprocess_datasets
from utils.metrics import compute_metrics

def train():
    """
    Main training function
    """
    print("=" * 80)
    print("WAV2VEC2 TRAINING PIPELINE")
    print("=" * 80)
    
    # Check GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n✓ Using device: {device}")
    if torch.cuda.is_available():
        print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
        print(f"✓ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    


    # Load processor
    print("\n1. Loading processor...")
    processor = Wav2Vec2Processor.from_pretrained('../models/processor')
    print("✓ Processor loaded")
    
    # Load datasets
    print("\n2. Loading datasets...")
    dataset = load_from_disk('../data/hf_dataset')
    print("✓ Datasets loaded")
    print(dataset)
    
    # Preprocess datasets
    print("\n3. Preprocessing datasets...")
    dataset = preprocess_datasets(dataset, processor, num_proc=1)

    print("✓ Preprocessing complete")
    
    # Load model
    print("\n4. Loading model...")
    model = Wav2Vec2ForCTC.from_pretrained('../models/initial_model')
    model.to(device)
    print("✓ Model loaded and moved to device")
    
    # Create data collator
    print("\n5. Creating data collator...")
    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)
    print("✓ Data collator created")
    
    # Training arguments
    print("\n6. Setting up training arguments...")
    training_args = TrainingArguments(
        output_dir="../models/checkpoints",
        
        # Training hyperparameters
        per_device_train_batch_size=8,  # Adjust based on GPU memory
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=2,  # Effective batch size = 8 * 2 = 16
        
        # Learning rate
        learning_rate=3e-4,
        warmup_steps=500,
        
        # Epochs and evaluation
        num_train_epochs=50,  # More epochs for small dataset
        eval_strategy="steps",
        eval_steps=100,
        save_steps=100,
        logging_steps=25,
        
        # Save best model
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        save_total_limit=3,
        
        # Optimization
        fp16=True,  # Mixed precision (faster on GPU)
        dataloader_num_workers=4,
        group_by_length=True,
        
        # Reporting
        report_to="tensorboard",
        logging_dir="logs",
        
        # Other
        push_to_hub=False,
        remove_unused_columns=False,
        label_names=["labels"],
    )
    
    print("✓ Training arguments configured")
    print(f"\n  Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
    print(f"  Total epochs: {training_args.num_train_epochs}")
    print(f"  Learning rate: {training_args.learning_rate}")
    
    # Initialize trainer
    print("\n7. Initializing trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=data_collator,
        compute_metrics=lambda pred: compute_metrics(pred, processor),
        tokenizer=processor.feature_extractor,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
    )
    
    print("✓ Trainer initialized")
    
    # Start training
    print("\n" + "=" * 80)
    print("STARTING TRAINING")
    print("=" * 80)
    print("\nMonitor training with: tensorboard --logdir=logs")
    print("\n" + "=" * 80 + "\n")
    
    # Train
    train_result = trainer.train()
    
    # Save final model
    print("\n8. Saving final model...")
    final_model_dir = "../models/final_model"
    trainer.save_model(final_model_dir)
    processor.save_pretrained(final_model_dir)
    
    print(f"✓ Final model saved to {final_model_dir}")
    
    # Print training summary
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"\nTraining loss: {train_result.training_loss:.4f}")
    print(f"Training time: {train_result.metrics['train_runtime']:.2f} seconds")
    print(f"Samples per second: {train_result.metrics['train_samples_per_second']:.2f}")
    
    # Evaluate on test set
    print("\n9. Evaluating on test set...")
    test_results = trainer.evaluate(dataset["test"])
    
    print("\nTest Results:")
    print(f"  WER: {test_results['eval_wer']:.4f} ({test_results['eval_wer']*100:.2f}%)")
    print(f"  CER: {test_results['eval_cer']:.4f} ({test_results['eval_cer']*100:.2f}%)")
    
    print("\n" + "=" * 80)
    
    return trainer, test_results

if __name__ == "__main__":
    trainer, results = train()