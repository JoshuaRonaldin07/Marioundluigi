"""
Emotion Classifier Perceptron (for ISEAR)
-------------------------
This script implements a simple perceptron model for emotion classification.
It trains on normalized data, extracts bag-of-words features, and predicts emotions.
Ultimately produces a CSV file with predicted emotions that can be evaluated.
"""

import csv
import os
import random
import math
from collections import Counter, defaultdict

def train_perceptron():
    # Step 1: Get paths to the training, validation and test files
    train_path = input("Enter the path to the training CSV file: ")
    val_path = input("Enter the path to the validation CSV file (or press Enter to skip): ")
    test_path = input("Enter the path to the test CSV file (or press Enter to skip): ")
    
    # Step 2: Read the training CSV file
    train_data = []
    try:
        with open(train_path, 'r', encoding='utf-8') as file:
            csv_reader = csv.reader(file)
            for row in csv_reader:
                if len(row) >= 2 and row[0] and row[1]:  # Ensure valid row
                    train_data.append([row[0].strip(), row[1].strip()])
        
        print(f"Loaded {len(train_data)} training examples")
    except FileNotFoundError:
        print(f"Error: The file {train_path} was not found.")
        return
    
    # Step 3: Extract vocabulary, emotion classes, and word statistics from training data
    all_words = set()
    emotions = set()
    
    # Count word occurrences per emotion and overall
    word_counts_per_emotion = defaultdict(Counter)
    docs_per_emotion = defaultdict(int)
    word_doc_counts = defaultdict(int)  # How many documents contain each word
    
    wlen = 1
    topw = 7
    fnum = 250

    for emotion, text in train_data:
        emotions.add(emotion)
        # Filter out words shorter than 3 letters
        words = set(word for word in text.split() if len(word) >= wlen)
        all_words.update(words)
        
        for word in words:
            word_doc_counts[word] += 1
        
        docs_per_emotion[emotion] += 1
        word_counts_per_emotion[emotion].update(words)
    
    emotions = sorted(list(emotions))
    
    print(f"Found {len(emotions)} emotions: {', '.join(emotions)}")
    print(f"Initial vocabulary size (words ≥ {wlen} letters): {len(all_words)}")
    
    # Step 4: Select discriminative features for each emotion
    discriminative_words = set()
    
    # For each emotion, find words that appear frequently in that emotion's documents
    # but rarely in other emotions' documents
    for emotion in emotions:
        # Calculate a score for each word based on its discriminative power
        word_scores = {}
        
        for word in all_words:
            # Count how many documents with this emotion contain this word
            emotion_count = word_counts_per_emotion[emotion][word]
            
            # Count how many documents with other emotions contain this word
            other_count = word_doc_counts[word] - emotion_count
            
            # Skip words that don't appear in this emotion
            if emotion_count == 0:
                continue
            
            # Calculate tf-idf inspired discriminative score
            # Higher when word is common in this emotion but rare in others
            tf = emotion_count / docs_per_emotion[emotion]  # Term frequency in this emotion

            # Calculate inverse document frequency for other emotions
            if other_count == 0:
                idf = 10.0  # High value for words exclusive to this emotion
            else:
                total_other_docs = sum(docs_per_emotion[e] for e in emotions if e != emotion)
                idf = math.log(total_other_docs / (other_count + 1))
            
            word_scores[word] = tf * idf
        
        # Select top discriminative words for this emotion (top 250 or fewer)
        num_features_per_emotion = min(fnum, len(word_scores))
        top_words = sorted(word_scores.items(), key=lambda x: x[1], reverse=True)[:num_features_per_emotion]
        
        print(f"Top {topw} discriminative words for '{emotion}': " + 
              ", ".join([f"{word} ({score:.2f})" for word, score in top_words[:topw]]))
        
        # Add these words to our discriminative vocabulary
        discriminative_words.update([word for word, _ in top_words])

    # Use discriminative words as our vocabulary
    vocabulary = sorted(list(discriminative_words))
    print(f"Final discriminative vocabulary size: {len(vocabulary)}")
    
    # Create feature vectors using the discriminative vocabulary
    def extract_features(text):
        # Filter out words shorter than 3 letters
        words = set(word for word in text.split() if len(word) >= wlen)
        
        # Create feature vector (binary word presence)
        features = {}
        for word in vocabulary:
            features[word] = 1 if word in words else 0
                
        # Add bias term
        features['bias'] = 1
        return features
    
    # Step 5: Initialize perceptron weights for each emotion
    weights = {}
    for emotion in emotions:
        weights[emotion] = {word: 0 for word in vocabulary}
        weights[emotion]['bias'] = 0  # Add bias term
    
    # Step 6: Train the perceptron
    max_epochs = 25
    learning_rate = 0.01
    best_val_accuracy = 0
    best_weights = None
    
    print("\nTraining perceptron...")
    
    # If validation path was provided, use it for early stopping
    val_data = []
    if val_path:
        try:
            with open(val_path, 'r', encoding='utf-8') as file:
                csv_reader = csv.reader(file)
                for row in csv_reader:
                    if len(row) >= 2 and row[0] and row[1]:
                        val_data.append([row[0].strip(), row[1].strip()])
            print(f"Loaded {len(val_data)} validation examples")
        except FileNotFoundError:
            print(f"Warning: Validation file {val_path} not found. Training without validation.")
    
    for epoch in range(max_epochs):
        # Shuffle training data
        random.shuffle(train_data)
        
        # Track metrics
        correct = 0
        total = 0
        
        # Process each training example
        for emotion, text in train_data:
            # Extract features
            features = extract_features(text)
            
            # Calculate scores for each emotion
            scores = {}
            for e in emotions:
                scores[e] = sum(weights[e][feat] * value for feat, value in features.items() if feat in weights[e])
            
            # Find predicted emotion
            predicted_emotion = max(scores, key=scores.get)
            
            # Update if prediction is incorrect
            if predicted_emotion != emotion:
                # Increase weights for correct emotion
                for feat, value in features.items():
                    if feat in weights[emotion]:
                        weights[emotion][feat] += learning_rate * value
                
                # Decrease weights for predicted emotion
                for feat, value in features.items():
                    if feat in weights[predicted_emotion]:
                        weights[predicted_emotion][feat] -= learning_rate * value
            else:
                correct += 1
            
            total += 1
        
        train_accuracy = correct / total if total > 0 else 0
        print(f"Epoch {epoch + 1}/{max_epochs}, Training accuracy: {train_accuracy:.4f}")
        
        # Validation step if validation data is available
        if val_data:
            val_correct = 0
            for true_emotion, text in val_data:
                features = extract_features(text)
                scores = {}
                for e in emotions:
                    scores[e] = sum(weights[e][feat] * value for feat, value in features.items() if feat in weights[e])
                predicted_emotion = max(scores, key=scores.get)
                if predicted_emotion == true_emotion:
                    val_correct += 1
            
            val_accuracy = val_correct / len(val_data) if val_data else 0
            print(f"Validation accuracy: {val_accuracy:.4f}")
            
            # Save best model based on validation accuracy
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                best_weights = {e: w.copy() for e, w in weights.items()}
                print(f"New best model saved!")
    
    # Use the best weights if validation was performed
    if best_weights:
        weights = best_weights
        print(f"Using best model with validation accuracy: {best_val_accuracy:.4f}")
    
    # Step 7: Ask which dataset to make predictions on if not specified
    prediction_data = None
    prediction_source = ""
    
    # Function to make predictions on a dataset
    def make_predictions(data_to_predict, source_name):
        print(f"\nMaking predictions on {len(data_to_predict)} examples from {source_name}")
        
        # Create predictions
        predictions = []
        for _, text in data_to_predict:  # Ignore the first column (true emotion if present)
            features = extract_features(text)
            scores = {}
            for e in emotions:
                scores[e] = sum(weights[e][feat] * value for feat, value in features.items() if feat in weights[e])
            predicted_emotion = max(scores, key=scores.get)
            predictions.append([predicted_emotion, text])
        
        # Save predictions to file
        output_filename = f"{source_name}-p.csv"
        with open(output_filename, 'w', newline='', encoding='utf-8') as file:
            csv_writer = csv.writer(file)
            csv_writer.writerows(predictions)
        
        print(f"Predictions saved to {output_filename}")
        print(f"You can now run the evaluator to compare {output_filename} with the true labels.")
    
    # If test path was provided, use it for predictions
    if test_path:
        try:
            test_data = []
            with open(test_path, 'r', encoding='utf-8') as file:
                csv_reader = csv.reader(file)
                for row in csv_reader:
                    if len(row) >= 2 and row[0] and row[1]:
                        test_data.append([row[0].strip(), row[1].strip()])
            
            prediction_data = test_data
            prediction_source = os.path.splitext(os.path.basename(test_path))[0]
        except FileNotFoundError:
            print(f"Error: The file {test_path} was not found.")
    
    # If no test file but validation file exists, offer to use validation data
    elif val_data:
        use_val = input("No test file provided. Make predictions on validation data? (y/n): ").lower()
        if use_val == 'y':
            prediction_data = val_data
            prediction_source = os.path.splitext(os.path.basename(val_path))[0]
    
    # If no test or validation, offer to use training data or input a new file
    if not prediction_data:
        options = "1. Training data\n2. Enter a new file path\n3. Skip predictions"
        choice = input(f"No prediction data selected. Choose an option:\n{options}\nEnter choice (1-3): ")
        
        if choice == '1':
            prediction_data = train_data
            prediction_source = os.path.splitext(os.path.basename(train_path))[0]
        elif choice == '2':
            new_path = input("Enter the path to the file for predictions: ")
            try:
                new_data = []
                with open(new_path, 'r', encoding='utf-8') as file:
                    csv_reader = csv.reader(file)
                    for row in csv_reader:
                        if len(row) >= 2 and row[0] and row[1]:
                            new_data.append([row[0].strip(), row[1].strip()])
                prediction_data = new_data
                prediction_source = os.path.splitext(os.path.basename(new_path))[0]
            except FileNotFoundError:
                print(f"Error: The file {new_path} was not found.")
        else:
            print("Skipping predictions.")
    
    # Make predictions if we have data
    if prediction_data:
        make_predictions(prediction_data, prediction_source)
    
    print("\nPerceptron training completed.")

if __name__ == "__main__":
    train_perceptron()