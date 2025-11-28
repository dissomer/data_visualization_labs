import matplotlib.pyplot as plt
from sklearn.manifold import Isomap, LocallyLinearEmbedding, MDS, TSNE
from sklearn.preprocessing import StandardScaler
import pandas as pd


def load_dataset(path='transfusion.data'):
    try:
        df = pd.read_csv(path)
    except:
        df = pd.read_csv(path, sep=';')

    print('Dataset shape:', df.shape)
    return df.values


def normalize_data(X):
    return StandardScaler().fit_transform(X)


def plot_embedding_2d(Y, labels, title):
    plt.figure(figsize=(6, 5))
    colors = ['red' if lbl == 0 else 'green' for lbl in labels]
    plt.scatter(Y[:, 0], Y[:, 1], c=colors, s=10)
    plt.title(title)
    plt.grid(True)
    plt.show()


def plot_embedding_3d(Y, labels, title):
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection='3d')
    colors = ['red' if lbl == 0 else 'green' for lbl in labels]
    ax.scatter(Y[:, 0], Y[:, 1], Y[:, 2], c=colors, s=10)
    plt.title(title)
    plt.show()


def get_methods_2d():
    return {
        'Isomap': Isomap(n_neighbors=10, n_components=2),
        'LLE': LocallyLinearEmbedding(n_neighbors=10, n_components=2, method='standard'),
        'MDS': MDS(n_components=2, n_init=4, max_iter=300, normalized_stress='auto'),
        't-SNE': TSNE(n_components=2, perplexity=30, learning_rate=200)
    }


def get_methods_3d():
    return {
        'Isomap_3D': Isomap(n_neighbors=10, n_components=3),
        'LLE_3D': LocallyLinearEmbedding(n_neighbors=10, n_components=3),
        'MDS_3D': MDS(n_components=3, n_init=4, max_iter=300, normalized_stress='auto'),
        't-SNE_3D': TSNE(n_components=3, perplexity=30, learning_rate=200)
    }


def run_2d_methods(X, y):
    print('\nRunning 2D dimension reduction...')
    methods = get_methods_2d()

    for name, model in methods.items():
        print(f'{name}')
        Y = model.fit_transform(X)
        plot_embedding_2d(Y, y, f'{name} (2D)')


def run_3d_methods(X, y):
    print('\nRunning 3D dimension reduction...')
    methods = get_methods_3d()

    for name, model in methods.items():
        print(f'{name}')
        Y = model.fit_transform(X)
        plot_embedding_3d(Y, y, f'{name} (3D)')


def main():
    data = load_dataset()
    X_raw = data[:, :4]  # R, F, M, T
    y = data[:, 4]  # class 0/1
    X = normalize_data(X_raw)
    run_2d_methods(X, y)
    run_3d_methods(X, y)


if __name__ == "__main__":
    main()
