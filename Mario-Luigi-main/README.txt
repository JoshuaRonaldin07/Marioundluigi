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

Update: There is also a perceptronnumpy.py which is the exact same except for using the
         numpy library to significantly increase the speed of calculations.



KNN:



BERT:

