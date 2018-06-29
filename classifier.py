"""
===================================================
     Introduction to Machine Learning (67577)
             IML HACKATHON, June 2018

            **  Code Classifier  **

Author(s):
Carl Veksler
Hagar Dolev
Hadar Dotan

===================================================
"""
import numpy as np
import math
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
import sklearn.tree

NUM_OF_SAMPLES = 8000
NUM_FEAT = 20000
PROJECT_LIST = ['Sonar', 'Dragonfly', 'tensorflow', 'devilution', 'flutter', 'react', 'spritejs']
TAGS = {'Sonar': 0, 'Dragonfly': 1, 'tensorflow': 2, 'devilution': 3, 'flutter': 4, 'react': 5, 'spritejs': 6}
STOP_WORDS = ["i", "me", "my", "myself", "we", "our", "ours",
              "ourselves", "you", "your", "yours", "yourself", "yourselves", "he",
              "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself",
              "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom",
              "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been",
              "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an",
              "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by",
              "for", "with", "about", "against", "between", "into", "through", "during", "before", "after",
              "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again",
              "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both",
              "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same",
              "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"]


class Classifier(object):

    def classify(self, X):
        """
        Recieves a list of m unclassified pieces of code, and predicts for each
        one the Github project it belongs to.
        :param X: A list of length m containing the code segments (strings)
        :return: y_hat - a list where each entry is a number between 0 and 6
        0 - Sonar
        1 - Dragonfly
        2 - tensorflow
        3 - devilution
        4 - flutter
        5 - react
        6 - spritejs
        """
        text_data = []
        y = []
        for p in PROJECT_LIST:
            with open(p + '_all_data.txt') as f:
                content = f.readlines()
                content = [x.strip() for x in content]
                text_data.extend(content)
                y.extend(([TAGS[p]] * len(content)))
        y = np.asarray(y)
        vectorizer = TfidfVectorizer(max_features=NUM_FEAT, stop_words=STOP_WORDS, ngram_range=(1, 3))
        X = vectorizer.fit_transform(text_data)
        X = normalize(X, norm='l1', axis=1)
        perm = np.random.permutation(X.shape[0])
        X = X[perm]
        y = y[perm]
        train_X, train_y = X[:NUM_OF_SAMPLES], y[:NUM_OF_SAMPLES]
        dt = sklearn.tree.DecisionTreeClassifier()
        dt.fit(train_X, train_y)
        return dt.predict(train_X)


