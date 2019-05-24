rom
sklearn.manifold
import TSNE
import matplotlib.pyplot as plt
from matplotlib import offsetbox
from matplotlib.pyplot import figure

import numpy as np


def dist_plot(pairs_lbs, y_pred, name='_'):
    col = []
    theta = []

    for label in [[label[0], label[1]] for label in pairs_lbs]:
        theta.append(0.2 * np.pi * label[0])
        if label[0] == label[1]:
            col.append(0.3)
        else:
            col.append(0.1)

    figure(num=None, figsize=(8, 8))
    N = 17820
    r = y_pred.ravel()[:N]

    # theta = 2 * np.pi *np.random.rand(N)
    area = 250
    # colors = -0.42*tr_y[:N]+0.2
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='polar')
    c = ax.scatter(theta, r, c=col, s=area, edgecolor='black', cmap='Set3', alpha=0.3)
    # ax.set_rorigin(-0.12)
    ax.set_yticklabels([])
    fig.savefig('/images/dist_plot' + str(name) + '.png')


def reshape(x):  # goes from (m, 28, 28) to (m,28*28)=(m,784)
    return np.array([x[i].flatten() for i in range(len(x))])


def plot_emb(x, x_or, y, size, name):
    ind = np.random.choice(len(x), size, replace=False)

    x = np.array([x[i] for i in ind])
    x_or = np.array([x_or[i] for i in ind])
    y = np.array([y[i] for i in ind])

    tsne = TSNE(n_components=2, random_state=0)
    X_tsne = tsne.fit_transform(reshape(x))
    plot_embedding(X_tsne, x_or, y, figsize=(10, 10), zoom=0.35, name=name)


def plot_embedding(X, X_or, y, figsize=(10, 10), zoom=0.3, title=None, name='loss'):
    # X is tsne transformed embedding (128 dim vector), X_or is the original X array

    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    plt.figure(figsize=figsize)
    ax = plt.subplot(111)
    plt.scatter(X[:, 0], X[:, 1], s=25, c=y / 10, cmap='Set3', linewidths=0.3, edgecolors='black', alpha=0.7)

    if hasattr(offsetbox, 'AnnotationBbox'):
        shown_images = np.array([[1., 1.]])  # just something big
        for i in range(X.shape[0]):
            dist = np.sum((X[i] - shown_images) ** 2, 1)
            if np.min(dist) < 4e-3:  # 4e-3: # don't show points that are too close
                continue
            shown_images = np.r_[shown_images, [X[i]]]
            imagebox = offsetbox.AnnotationBbox(
                offsetbox.OffsetImage(X_or[i], cmap=plt.cm.gray_r, zoom=zoom), X[i])
            ax.add_artist(imagebox)
    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)
    plt.savefig('images/tSNE_' + str(name) + '.png')




