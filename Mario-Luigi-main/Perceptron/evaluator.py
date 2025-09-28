"""
Emotion Classifier Evaluator (for isear)
-------------------------
This script takes two CSV files (true and predicted labels) and compares them.
It neatly calculates TP, FP, FN, Precision, and Recall.
Ultimately outputs the F1-score (as a float).
"""

import csv
import os

def evaluate_emotion_predictions():
    # Step 1: Get the path to the true labels CSV file
    true_labels_path = input("Enter the path to the true labels CSV file: ")
    
    # Step 2: Get the path to the predicted labels CSV file
    predictions_path = input("Enter the path to the predicted labels CSV file: ")
    
    try:
        # Step 3: Read the gold CSV file
        true_labels = []
        with open(true_labels_path, 'r', encoding='utf-8') as file:
            csv_reader = csv.reader(file)
            for row in csv_reader:
                if len(row) >= 2 and row[0] and row[1]:  # Just in case
                    true_labels.append([row[0].strip(), row[1].strip()])
        
        # Step 4: Read the predicted CSV file
        predictions = []
        with open(predictions_path, 'r', encoding='utf-8') as file:
            csv_reader = csv.reader(file)
            for row in csv_reader:
                if len(row) >= 2 and row[0] and row[1]:  # Just in case
                    predictions.append([row[0].strip(), row[1].strip()])
        
        # Step 5: Check if both datasets read have the same number of rows
        if len(true_labels) != len(predictions):
            print("Warning: Number of rows not matching!")
            print(f"True labels: {len(true_labels)}, Predictions: {len(predictions)}")
            print("Will evaluate based on the common texts only.")
        
        # Step 6: Create dictionaries to map text to emotion for an easier comparison
        true_dict = {row[1]: row[0] for row in true_labels}
        pred_dict = {row[1]: row[0] for row in predictions}
        
        # Step 7: Calculate TP, FP, and FN for every single emotion
        all_emotions = set(emotion for emotion, _ in true_labels + predictions)
        
        # Create the confusion matrix components
        tp_counts = {emotion: 0 for emotion in all_emotions}
        fp_counts = {emotion: 0 for emotion in all_emotions}
        fn_counts = {emotion: 0 for emotion in all_emotions}
        
        # Find common texts to evaluate
        common_texts = set(true_dict.keys()) & set(pred_dict.keys())
        
        for text in common_texts:
            true_emotion = true_dict[text]
            pred_emotion = pred_dict[text]
            
            if true_emotion == pred_emotion:
                # True Positive for the true emotion
                tp_counts[true_emotion] += 1
            else:
                # False Negative for the true emotion
                fn_counts[true_emotion] += 1
                # False Positive for the predicted emotion
                fp_counts[pred_emotion] += 1
        
        # Step 8: Calculate Precision, Recall, and F1-score for each emotion and also overall
        total_tp = sum(tp_counts.values())
        total_fp = sum(fp_counts.values())
        total_fn = sum(fn_counts.values())
        
        # Calculate macro-averaged metrics
        precision_values = []
        recall_values = []
        f1_values = []
        
        # Print per-emotion metrics
        print("\nPer-emotion metrics:")
        print("-" * 90)
        print(f"{'Emotion':<23} {'TP':<11} {'FP':<11} {'FN':<11} {'Precision':<10} {'Recall':<10} {'F1-score':<10}")
        print("-" * 90)
        
        for emotion in sorted(all_emotions):
            tp = tp_counts[emotion]
            fp = fp_counts[emotion]
            fn = fn_counts[emotion]
            
            # Calculate precision and recall, while handling division by zero as usual
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            
            # Calculate F1-score
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            # Store for macro-averaging
            precision_values.append(precision)
            recall_values.append(recall)
            f1_values.append(f1)
            
            # Print metrics for this emotion
            print(f"{emotion:<23} {tp:<11} {fp:<11} {fn:<11} {precision:.4f}     {recall:.4f}     {f1:.4f}")
        
        # Calculate macro-averaged metrics
        macro_precision = sum(precision_values) / len(precision_values) if precision_values else 0
        macro_recall = sum(recall_values) / len(recall_values) if recall_values else 0
        macro_f1 = sum(f1_values) / len(f1_values) if f1_values else 0
        
        # Calculate micro-averaged metrics (global counts)
        micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0
        
        # Print overall metrics
        print("\nOverall metrics:")
        print("-" * 90)
        print(f"Total TP: {total_tp}, Total FP: {total_fp}, Total FN: {total_fn}")
        print(f"Micro-averaged - Precision: {micro_precision:.4f}, Recall: {micro_recall:.4f}, F1-score: {micro_f1:.4f}")
        print(f"Macro-averaged - Precision: {macro_precision:.4f}, Recall: {macro_recall:.4f}, F1-score: {macro_f1:.4f}")
        
        # Return the micro-averaged F1-score as the final output of the program
        return micro_f1
        
    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
        return 0
    except Exception as e:
        print(f"An error occurred: {e}")
        return 0

if __name__ == "__main__":
    f1_score = evaluate_emotion_predictions()
    print(f"\nFinal F1-score: {f1_score:.4f}")