"""
Emotion Classifier ML-KNN (for ISEAR)
-------------------------------------
Implements ML-KNN algorithm for emotion classification using bag-of-words features.
Loads training, validation, and test CSVs (no headers), and outputs predictions as CSVs.
"""

import csv
import os
import math
import random
from collections import Counter, defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def load_csv(path):
    data = []
    try:
        with open(path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 2 and row[0].strip() and row[1].strip():
                    data.append((row[0].strip(), ','.join(row[1:]).strip()))

    except FileNotFoundError:
        print(f"Error: File not found: {path}")
    return data


def extract_features(train_texts, val_texts=None, test_texts=None):
    vectorizer = TfidfVectorizer(binary=True, max_features=3000)
    X_train = vectorizer.fit_transform(train_texts)
    X_val = vectorizer.transform(val_texts) if val_texts else None
    X_test = vectorizer.transform(test_texts) if test_texts else None
    return X_train, X_val, X_test, vectorizer


def compute_neighbors(X_train, k):
    sim_matrix = cosine_similarity(X_train)
    neighbors = []
    for i in range(sim_matrix.shape[0]):
        sim_scores = list(enumerate(sim_matrix[i]))
        sim_scores.sort(key=lambda x: x[1], reverse=True)
        top_k = [idx for idx, score in sim_scores[1:k+1]]
        neighbors.append(top_k)
    return neighbors


def estimate_priors(train_labels, neighbors, emotions):
    prior = defaultdict(float)
    conditional = defaultdict(lambda: defaultdict(float))

    for idx, neigh_idxs in enumerate(neighbors):
        label = train_labels[idx]
        counts = Counter(train_labels[j] for j in neigh_idxs)
        for emo in emotions:
            conditional[emo][label == emo] += counts[emo]

    for emo in emotions:
        total = sum(conditional[emo].values())
        if total == 0:
            prior[emo] = 0.0
        else:
            prior[emo] = conditional[emo][True] / total
    return prior


def predict_knn(X_train, train_labels, X_test, emotions, k):
    sim_matrix = cosine_similarity(X_test, X_train)
    predictions = []

    for i in range(sim_matrix.shape[0]):
        sims = list(enumerate(sim_matrix[i]))
        sims.sort(key=lambda x: x[1], reverse=True)
        top_k = [idx for idx, _ in sims[:k]]
        label_counts = Counter(train_labels[j] for j in top_k)
        predicted = label_counts.most_common(1)[0][0]
        predictions.append(predicted)

    return predictions


def save_predictions(predictions, texts, source):
    output_file = f"{source}-mlknn-pred.csv"
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        for pred, text in zip(predictions, texts):
            writer.writerow([pred, text])
    print(f"Predictions saved to {output_file}")


def train_mlknn():
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

    train_labels, train_texts = zip(*train_data)
    val_labels, val_texts = zip(*val_data) if val_data else ([], [])
    test_labels, test_texts = zip(*test_data) if test_data else ([], [])

    emotions = sorted(set(train_labels))
    print(f"Loaded {len(train_texts)} training samples. Emotions: {', '.join(emotions)}")

    # Step 2: Extract features
    X_train, X_val, X_test, vectorizer = extract_features(train_texts, val_texts, test_texts)

    # Step 3: Train ML-KNN
    k = 10
    print(f"Computing neighbors with k={k}...")
    neighbors = compute_neighbors(X_train, k)

    print("Estimating class priors and conditionals...")
    priors = estimate_priors(train_labels, neighbors, emotions)

    # Step 4: Predict
    def do_prediction(X_target, texts, source_name):
        preds = predict_knn(X_train, train_labels, X_target, emotions, k)
        save_predictions(preds, texts, source_name)

    if X_test is not None:
        do_prediction(X_test, test_texts, os.path.splitext(os.path.basename(test_path))[0])
    elif X_val is not None:
        use_val = input("No test file. Predict on validation data? (y/n): ")
        if use_val.lower() == 'y':
            do_prediction(X_val, val_texts, os.path.splitext(os.path.basename(val_path))[0])
    else:
        use_train = input("No test/val file. Predict on training data? (y/n): ")
        if use_train.lower() == 'y':
            do_prediction(X_train, train_texts, os.path.splitext(os.path.basename(train_path))[0])

    print("ML-KNN classification complete.")


if __name__ == "__main__":
    train_mlknn()
