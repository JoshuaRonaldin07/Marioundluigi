CL Team Labs (SuSe 2025)
Mario&Liuigi
Mehryar Lessani & Joshua Ronaldino Peter

Perceptron:

This is a simple perceptron for text classification that we built from scratch.
We have tried to stick to a modular approach.
As such, in this version (1.0) we have 3 main modules:

    1. normaliser.py : takes any single-label (single- or multi- class) CSV dataset and
        performs a simple normalisation, enforcing lowercase and removing all
        non-alphabetical characters.

    2. evaluator.py : takes two CSV files (one with true labels and one with predicted
        labels) and performs a full comparison returning emotion-wise and overall
        precision, recall, and F1-score (micro and macro).

    3. perceptron.py : takes a training set, a validation set, and a test set (the
        latter two being optional) and performs feature selection, weight training, and
        making predictions.

Notes:
    1. A tf-idf inspired approach is used to rank the features for each emotion so that
        the top most discriminative features can be used for training the model.

    2. The variables wlen, topw, and fnum can be altered to change the minimum length for
        tokens to consider, the number of top discriminative words to display for each
        emotion, and the number of top features to use.

    3. We have also developed a version 1.1 which features the ability to save/load a
        features.csv with the features selected for each emotion [and their weights]
        but it is currently in BETA.

        

KNN: ML-KNN Classifier (MLKNN5.py)

The KNN classifier implemented in MLKNN5.py is an Emotion Classifier ML-KNN designed for the ISEAR dataset. It uses a bag-of-words approach and is structured for multi-label classification using the ML-KNN algorithm, although the provided predict_knn function appears to implement a standard K-Nearest Neighbors for single-label classification based on the majority label among the K neighbors.

Key Features and Implementation Details
Feature Extraction: TF-IDF Vectorization is used, with the TfidfVectorizer set to binary=True and a maximum of 3000 features. This creates a high-dimensional, sparse feature representation for the text.

Similarity Metric: Cosine similarity is used to determine the distance (or similarity) between texts.

Neighbor Computation: The compute_neighbors function pre-computes the K nearest neighbors for each training sample.

Prediction (predict_knn):

It calculates the cosine similarity between the test samples and all training samples.

It identifies the K closest training samples for each test sample (where K is set to 10 by default).

The final prediction is the most common label (majority vote) among the K nearest neighbors.

ML-KNN Specific Functions: Functions like estimate_priors are present to estimate class priors and conditionals, which are components of the full ML-KNN algorithm, although the main predict_knn function seems to use a simpler majority voting approach.

Modularity: The process is broken down into clear steps: load_csv, extract_features, compute_neighbors, estimate_priors, and predict_knn.

BERT: BERT Emotion Classifier (Bert7.py)
The BERT classifier implemented in Bert7.py is a BERT-based emotion classification model, utilizing a pre-trained transformer model for single-label classification.

Key Features and Implementation Details
Model: Uses Hugging Face Transformers library for fine-tuning pre-trained models.

Model Options: The user can choose from several pre-trained models: bert-base-uncased, roberta-base, distilbert-base-uncased, or albert-base-v2.

Input Handling (EmotionDataset):

Texts are tokenized and preprocessed, including adding [CLS] and [SEP] special tokens.

The maximum sequence length is set to 256.

Labels are first encoded using LabelEncoder.

Handling Imbalance:

The code checks for class balance and prints an imbalance warning if the ratio of the largest to the smallest class is greater than 3:1.

If imbalanced, it computes and uses class weights to ensure fair learning across all emotions.

Training Arguments (Fine-Tuning): The model is fine-tuned with optimized parameters:

Epochs: 5.

Learning Rate: 2e−5.

Batch Size: Per device train/eval batch size is 8.

Gradient Accumulation: 2 steps (simulating a larger batch size).

Regularization: Dropout is added to hidden_dropout_prob and attention_probs_dropout_prob at 0.3.

Efficiency: Uses fp16=True (mixed precision).

Evaluation Metrics: The compute_metrics function calculates accuracy, precision, recall, and F1 score (weighted average) for evaluation.

Reproducibility: A seed of 42 is set for reproducibility.

Update: There is also a perceptronnumpy.py which is the exact same except for using the
         numpy library to significantly increase the speed of calculations.


