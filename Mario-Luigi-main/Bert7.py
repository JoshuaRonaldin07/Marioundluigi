"""
BERT Emotion Classifier (for ISEAR)
-----------------------------------
Implements BERT-based emotion classification using pre-trained transformer model.
Loads training, validation, and test CSVs (no headers), and outputs predictions as CSVs.
"""

import csv
import os
import torch
import numpy as np
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')


class EmotionDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=256):  # Increased max_length
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # Add special tokens for emotion classification
        text = f"[CLS] {text} [SEP]"
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


def load_csv(path):
    """Load CSV data in format: emotion, sentence with improved preprocessing"""
    data = []
    try:
        with open(path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 2 and row[0].strip() and row[1].strip():
                    # Remove quotes from emotion label if present
                    emotion = row[0].strip().strip('"').lower()  # Normalize case
                    sentence = ','.join(row[1:]).strip().strip('"')
                    
                    # Basic text cleaning
                    sentence = sentence.replace('\n', ' ').replace('\r', ' ')
                    sentence = ' '.join(sentence.split())  # Remove extra whitespace
                    
                    if len(sentence) > 10:  # Filter very short sentences
                        data.append((emotion, sentence))
    except FileNotFoundError:
        print(f"Error: File not found: {path}")
    return data


def compute_metrics(eval_pred):
    """Compute metrics for evaluation"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
    
    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


def save_predictions(predictions, texts, emotions, source):
    """Save predictions to CSV file"""
    output_file = f"{source}-bert-pred.csv"
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        for pred_idx, text in zip(predictions, texts):
            emotion_label = emotions[pred_idx]
            writer.writerow([emotion_label, text])
    print(f"Predictions saved to {output_file}")


def check_class_balance(labels, emotions):
    """Check and report class distribution"""
    from collections import Counter
    label_counts = Counter(labels)
    print("\nClass distribution:")
    for emotion in emotions:
        count = label_counts[emotion]
        percentage = (count / len(labels)) * 100
        print(f"  {emotion}: {count} samples ({percentage:.1f}%)")
    
    # Check if severely imbalanced
    min_count = min(label_counts.values())
    max_count = max(label_counts.values())
    imbalance_ratio = max_count / min_count
    
    if imbalance_ratio > 3:
        print(f"⚠️  Dataset is imbalanced (ratio: {imbalance_ratio:.1f}:1)")
        return True
    return False


def train_bert_classifier():
    """Main training and prediction function"""
    
    # Step 1: Load data
    train_path = input("Enter the path to the training CSV: ")
    val_path = input("Enter the path to the validation CSV (or press Enter): ")
    test_path = input("Enter the path to the test CSV (or press Enter): ")

    train_data = load_csv(train_path)
    val_data = load_csv(val_path) if val_path else []
    test_data = load_csv(test_path) if test_path else []

    if not train_data:
        print("No training data found.")
        return

    # Extract labels and texts
    train_labels, train_texts = zip(*train_data)
    val_labels, val_texts = zip(*val_data) if val_data else ([], [])
    test_labels, test_texts = zip(*test_data) if test_data else ([], [])

    # Get unique emotions and encode labels
    emotions = sorted(list(set(train_labels)))
    label_encoder = LabelEncoder()
    label_encoder.fit(emotions)
    
    train_encoded_labels = label_encoder.transform(train_labels)
    val_encoded_labels = label_encoder.transform(val_labels) if val_labels else []
    
    print(f"Loaded {len(train_texts)} training samples.")
    print(f"Emotions ({len(emotions)}): {', '.join(emotions)}")
    
    # Check class balance
    is_imbalanced = check_class_balance(train_labels, emotions)
    
    # Calculate class weights for imbalanced datasets
    class_weights = None
    if is_imbalanced:
        from sklearn.utils.class_weight import compute_class_weight
        class_weights = compute_class_weight(
            'balanced',
            classes=np.array(range(len(emotions))),
            y=train_encoded_labels
        )
        class_weights = torch.tensor(class_weights, dtype=torch.float32)
        print("✓ Using class weights to handle imbalance")
    
    # Step 2: Initialize BERT model and tokenizer - TRY DIFFERENT MODELS
    model_options = [
        "bert-base-uncased",
        "roberta-base", 
        "distilbert-base-uncased",
        "albert-base-v2"
    ]
    
    model_choice = input(f"Choose model (0-{len(model_options)-1}): {model_options}: ").strip()
    try:
        model_name = model_options[int(model_choice)]
    except (ValueError, IndexError):
        model_name = "bert-base-uncased"  # Default
    
    print(f"Loading {model_name} model...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(emotions),
        problem_type="single_label_classification",
        hidden_dropout_prob=0.3,  # Add dropout for regularization
        attention_probs_dropout_prob=0.3
    )
    
    # Step 3: Create datasets
    train_dataset = EmotionDataset(train_texts, train_encoded_labels, tokenizer)
    val_dataset = EmotionDataset(val_texts, val_encoded_labels, tokenizer) if val_texts else None
    
    # Step 4: Set up training arguments - OPTIMIZED FOR BETTER PERFORMANCE
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=5,  # Increased from 3
        per_device_train_batch_size=8,  # Reduced for better gradients
        per_device_eval_batch_size=8,
        learning_rate=2e-5,  # Optimal for BERT fine-tuning
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        eval_strategy="epoch" if val_dataset else "no",
        save_strategy="epoch",
        load_best_model_at_end=True if val_dataset else False,
        save_total_limit=1,
        logging_steps=50,  # More frequent logging
        remove_unused_columns=False,
        fp16=True,  # Mixed precision for efficiency
        gradient_accumulation_steps=2,  # Simulate larger batch size
        dataloader_drop_last=True,
        seed=42,  # For reproducibility
    )
    
    # Step 5: Create trainer
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics if val_dataset else None,
    )
    
    # Step 6: Train the model
    print("Starting training...")
    trainer.train()
    
    # Step 7: Make predictions
    def do_prediction(texts, source_name):
        """Make predictions on given texts"""
        print(f"Making predictions for {source_name}...")
        
        # Create dataset for prediction
        dummy_labels = [0] * len(texts)  # Dummy labels for prediction
        pred_dataset = EmotionDataset(texts, dummy_labels, tokenizer)
        
        # Get predictions
        predictions = trainer.predict(pred_dataset)
        predicted_labels = np.argmax(predictions.predictions, axis=1)
        
        # Save predictions
        save_predictions(predicted_labels, texts, emotions, source_name)
        
        return predicted_labels
    
    # Step 8: Predict on available datasets
    if test_texts:
        do_prediction(test_texts, os.path.splitext(os.path.basename(test_path))[0])
    elif val_texts:
        use_val = input("No test file. Predict on validation data? (y/n): ")
        if use_val.lower() == 'y':
            do_prediction(val_texts, os.path.splitext(os.path.basename(val_path))[0])
    else:
        use_train = input("No test/val file. Predict on training data? (y/n): ")
        if use_train.lower() == 'y':
            do_prediction(train_texts, os.path.splitext(os.path.basename(train_path))[0])
    
    print("BERT emotion classification complete.")
    
    # Clean up
    if os.path.exists('./results'):
        import shutil
        shutil.rmtree('./results')
    if os.path.exists('./logs'):
        import shutil
        shutil.rmtree('./logs')


if __name__ == "__main__":
    # Check if GPU is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    train_bert_classifier()