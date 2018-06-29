import numpy as np
import math
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
import sklearn.neighbors
import sklearn.tree
import matplotlib.pyplot as plt
import scipy


NUM_OF_SAMPLES = 5000
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
    vectorizer = TfidfVectorizer(max_features=n_features, stop_words=STOP_WORDS, ngram_range=(1, 3))
    X = vectorizer.fit_transform(text_data)
    X = normalize(X, norm='l1', axis=1)
    perm = np.random.permutation(X.shape[0])
    X = X[perm]
    y = y[perm]
    X, y = X[:NUM_OF_SAMPLES], y[:NUM_OF_SAMPLES]
    train_size = math.floor(X.shape[0] * split[0])
    test_size = math.floor(X.shape[0] * (split[0] + split[1]))
    train_X, test_X, validation_X = X[:train_size], X[train_size:test_size], X[test_size:]
    train_y, test_y, validation_y = y[:train_size], y[train_size:test_size], y[test_size:]
    return train_X, train_y, test_X, test_y, validation_X, validation_y




def find_best_model(train_X, train_y, test_X, test_y, val_X, val_y):
    """"""
    print('In find best model')

    # KNN
    print('KNN')
    # knn_training_error, knn_validation_error , knn_test_error = [], [], []
    knn = sklearn.neighbors.KNeighborsClassifier(n_neighbors=20)
    knn.fit(train_X, train_y)


    # knn_training_error.append((1/train_X.shape[0]) * np.sum(train_y != knn.predict(test_X)))
    # knn_validation_error.append(((1/val_X.shape[0]) * np.sum(val_y != knn.predict(val_X))))
    # knn_test_error.append(((1/test_X.shape[0]) * np.sum(test_y != knn.predict(test_X))))



    knn_train_err = (1 / train_X.shape[0]) * np.sum(train_y != knn.predict(train_X))
    knn_test_err = ((1 / test_X.shape[0]) * np.sum(test_y != knn.predict(test_X)))
    knn_val_err = ((1 / val_X.shape[0]) * np.sum(val_y != knn.predict(val_X)))
    print('KNN train err:' + str(knn_train_err))
    print('KNN test err:' + str(knn_test_err))
    print('KNN val err:' + str(knn_val_err))


    # desicion tree
    print('Desicion Tree')
    # dt_training_error, dt_validation_error, dt_test_error = [], [], []
    dt = sklearn.tree.DecisionTreeClassifier()
    dt.fit(train_X, train_y)

    # dt_training_error.append((1 / train_X.shape[0]) * np.sum(train_y != dt.predict(train_X)))
    # dt_validation_error.append(((1 / val_X.shape[0]) * np.sum(val_y != dt.predict(val_X))))
    # dt_test_error.append(((1 / test_X.shape[0]) * np.sum(test_y != dt.predict(test_X))))

    dt_training_error = (1 / train_X.shape[0]) * np.sum(train_y != dt.predict(train_X))
    dt_test_error = ((1 / test_X.shape[0]) * np.sum(test_y != dt.predict(test_X)))
    dt_validation_error = ((1 / val_X.shape[0]) * np.sum(val_y != dt.predict(val_X)))
    print('DT train err:' + str(dt_training_error))
    print('DT test err:' + str(dt_test_error))
    print('DT val err:' + str(dt_validation_error))

    # SVC

# plot

    # plt.plot(knn_training_error, label='knn training error', color='magenta')
    # plt.plot(knn_validation_error, label='validation error',color='deepskyblue')
    # plt.plot(knn_test_error, label='test error',color='deepskyblue')
    # plt.title('knn')
    # plt.legend(loc='best')
    # # plt.xlabel('T - number of base learners to learn')
    # # plt.ylabel('Error')
    # plt.show()
    #
    # plt.plot(dt_training_error, label='training error', color='magenta')
    # plt.plot(dt_validation_error, label='validation error',
    #          color='deepskyblue')
    # plt.plot(dt_test_error, label='test error', color='deepskyblue')
    # plt.title('dt')
    # plt.legend(loc='best')
    # # plt.xlabel('T - number of base learners to learn')
    # # plt.ylabel('Error')
    # plt.show()

    return knn_train_err, knn_test_err, knn_val_err, dt_training_error, dt_test_error, dt_validation_error


def main():
    knn_training_error, knn_validation_error, knn_test_error = [], [], []
    dt_training_error, dt_validation_error, dt_test_error = [], [], []
    for n_feat in [50, 1000, 3000, 5000, 10000, 20000, 50000]:
        print(n_feat)
        ret = get_data(n_features=n_feat, split=(0.7, 0.2, 0.1))
        train_X, train_y = (ret[0]).todense(), (ret[1])
        test_X, test_y = (ret[2]).todense(), (ret[3])
        validation_X, validation_y = (ret[4]).todense(), (ret[5])
        k1, k2, k3, d1, d2, d3 = find_best_model(train_X, train_y, test_X, test_y, validation_X, validation_y)
        knn_training_error.append(k1)
        knn_test_error.append(k2)
        knn_validation_error.append(k3)
        dt_training_error.append(d1)
        dt_test_error.append(d2)
        dt_validation_error.append(d3)
    #plot
    plt.plot(knn_training_error, label='knn training error', color='magenta')
    plt.plot(knn_validation_error, label='validation error', color='yellowgreen')
    plt.plot(knn_test_error, label='test error', color='deepskyblue')
    plt.title('knn')
    plt.legend(loc='best')
    # plt.xlabel('T - number of base learners to learn')
    # plt.ylabel('Error')
    plt.show()

    plt.plot(dt_training_error, label='training error', color='magenta')
    plt.plot(dt_validation_error, label='validation error', color='yellowgreen')
    plt.plot(dt_test_error, label='test error', color='deepskyblue')
    plt.title('dt')
    plt.legend(loc='best')
    # plt.xlabel('T - number of base learners to learn')
    # plt.ylabel('Error')
    plt.show()


main()
