import numpy as np
import math
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


PROJECT_LIST = ['Sonar', 'Dragonfly', 'tensorflow', 'devilution', 'flutter', 'react', 'spritejs']
TAGS = {'Sonar': 0, 'Dragonfly': 1, 'tensorflow': 2, 'devilution': 3, 'flutter': 4, 'react': 5, 'spritejs': 6}


# Possible knobs - ngrams, different regex
def get_data(n_features, split):
    text_data = []
    y = []
    for p in PROJECT_LIST:
        with open(p + '_all_data.txt') as f:
            content = f.readlines()
            content = [x.strip() for x in content]
            text_data.extend(content)
            y.extend(([TAGS[p]] * len(content)))
    y = np.asarray(y)
    vectorizer = TfidfVectorizer(max_features=n_features)
    X = vectorizer.fit_transform(text_data)
    X = normalize(X, norm='l1', axis=1)
    perm = np.random.permutation(X.shape[0])
    X = X[perm]
    y = y[perm]
    train_size = math.floor(X.shape[0] * split[0])
    test_size = math.floor(X.shape * (split[0] + split[1]))
    train_X, test_X, validation_X = X[:train_size], X[train_size:test_size], X[test_size:]
    train_y, test_y, validation_y = y[:train_size], y[train_size:test_size], y[test_size:]
    return (train_X, train_y), (test_X, test_y), (validation_X, validation_y)

