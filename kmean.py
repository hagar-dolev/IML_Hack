import sklearn.cluster
import matplotlib.pyplot as plt


def to_main(X, y):

    k_means = sklearn.cluster.KMeans(n_clusters=7, init='k-means++')
    k_means.fit_predict(X)

    labels = k_means.labels_
    centers = k_means.cluster_centers_

    plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='viridis')
    plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)


    plt.scatter(X[:, 0], X[:, 1], c=labels , s=50, cmap='viridis')
    plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
