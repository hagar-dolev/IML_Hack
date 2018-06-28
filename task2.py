import numpy as np
import math
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
import sklearn.cluster
import sklearn.neighbors
import sklearn.tree
import matplotlib.pyplot as plt



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
    return train_X, train_y, test_X, test_y, validation_X, validation_y




def find_best_model(train_X, train_y, test_X, test_y, val_X, val_y):
    """"""
    kmeans = sklearn.cluster.KMeans(n_clusters=7)


    # KNN
    knn_training_error, knn_validation_error , knn_test_error = [], [], []
    knn = sklearn.neighbors.KNeighborsClassifier(n_neighbors=100)
    knn.fit(train_X, train_y)


    knn_training_error.append((1/train_X.shape[0]) * np.count_nonzero(train_y - knn.predict(train_X)))
    knn_validation_error.append(((1/val_X.shape[0]) * np.count_nonzero(val_y - knn.predict(val_X))))
    knn_test_error.append(
        ((1 / test_X.shape[0]) * np.count_nonzero(test_y - knn.predict(test_X))))


    # desicion tree

    dt_training_error, dt_validation_error, dt_test_error = [], [], []
    dt = sklearn.tree.DecisionTreeClassifier()
    dt.fit(train_X, train_y)

    dt_training_error.append((1 / train_X.shape[0]) * np.count_nonzero(
        train_y - dt.predict(train_X)))
    dt_validation_error.append(
        ((1 / val_X.shape[0]) * np.count_nonzero(val_y - dt.predict(val_X))))
    dt_test_error.append(
        ((1 / test_X.shape[0]) * np.count_nonzero(
            test_y - dt.predict(test_X))))



    # SVC



    #plot

    plt.plot(knn_training_error, label='knn training error', color='magenta')
    plt.plot(knn_validation_error, label='validation error',color='deepskyblue')
    plt.plot(knn_test_error, label='test error',color='deepskyblue')
    plt.title('knn')
    plt.legend(loc='best')
    # plt.xlabel('T - number of base learners to learn')
    # plt.ylabel('Error')
    plt.show()

    plt.plot(dt_training_error, label='training error', color='magenta')
    plt.plot(dt_validation_error, label='validation error',
             color='deepskyblue')
    plt.plot(dt_test_error, label='test error', color='deepskyblue')
    plt.title('dt')
    plt.legend(loc='best')
    # plt.xlabel('T - number of base learners to learn')
    # plt.ylabel('Error')
    plt.show()


def main():

    for n_feat in [1000, 2000,3000, 4000, 5000, 10000, 20000]:
        find_best_model(get_data(n_features=n_feat,split=(0.7,0.2,0.1)))




main()