import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def load_data(path):
    data = pd.read_csv(path)
    X = data.select_dtypes(include=['number']).values
    if 'Class' in data.columns:
        y = pd.factorize(data['Class'])[0]
    else:
        y = X[:, 0]
    X_centered = X - X.mean(axis=0)
    return X_centered, y


def plot_pca(X, y):
    pca_2 = PCA(n_components=2).fit_transform(X)
    pca_3 = PCA(n_components=3).fit_transform(X)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    colors = ['green' if label == 0 else 'red' for label in y]
    plt.scatter(pca_2[:, 0], pca_2[:, 1], c=colors, s=12)
    plt.title('PCA 2D')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    ax = plt.subplot(1, 2, 2, projection='3d')
    ax.scatter(pca_3[:, 0], pca_3[:, 1], pca_3[:, 2], c=colors, s=12)
    ax.set_title('PCA 3D')
    plt.show()
    return pca_2, pca_3


def compute_svd_eigenvalues(X):
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    eigenvalues = S ** 2
    eigenvalues_sorted = np.sort(eigenvalues)[::-1]
    plt.figure(figsize=(7, 5))
    plt.plot(range(1, len(eigenvalues_sorted) + 1), eigenvalues_sorted, 'o-', linewidth=2)
    plt.title('Eigenvalues')
    plt.xlabel('Component number')
    plt.ylabel('Eigenvalue')
    plt.grid(True)
    plt.show()
    return U, S, Vt, eigenvalues_sorted


def choose_optimal_d_by_mse(X, U, S, Vt, mse_threshold=1.0):
    for d in range(1, len(S) + 1):
        S_cut = np.copy(S)
        S_cut[d:] = 0
        X_rec = (U * S_cut) @ Vt
        mse = np.mean((X - X_rec) ** 2)
        if mse <= mse_threshold:
            print(f'Optimal d by MSE = {d}, MSE = {mse:.4f}')
            return d, X_rec
    print(f'All components used, MSE = {mse:.4f}')
    return len(S), X_rec


def compare_pca_svd(X, y, pca_2, pca_3, U, S, Vt, d):
    X_svd_2 = U[:, :2] * S[:2]
    X_svd_3 = U[:, :3] * S[:3]
    colors = ['green' if label == 0 else 'red' for label in y]
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(pca_2[:, 0], pca_2[:, 1], c=colors, s=12)
    plt.title('PCA 2D')
    plt.subplot(1, 2, 2)
    plt.scatter(X_svd_2[:, 0], X_svd_2[:, 1], c=colors, s=12)
    plt.title('SVD 2D')
    plt.suptitle('PCA 2D vs SVD 2D')
    plt.show()
    fig = plt.figure(figsize=(12, 5))
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax1.scatter(pca_3[:, 0], pca_3[:, 1], pca_3[:, 2], c=colors, s=12)
    ax1.set_title('PCA 3D')
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    ax2.scatter(X_svd_3[:, 0], X_svd_3[:, 1], X_svd_3[:, 2], c=colors, s=12)
    ax2.set_title('SVD 3D')
    plt.suptitle('PCA 3D vs SVD 3D')
    plt.show()


def main():
    X, y = load_data('transfusion.data')
    pca_2, pca_3 = plot_pca(X, y)
    U, S, Vt, eigenvalues_sorted = compute_svd_eigenvalues(X)
    d, X_rec = choose_optimal_d_by_mse(X, U, S, Vt, mse_threshold=1.0)
    compare_pca_svd(X, y, pca_2, pca_3, U, S, Vt, d)


if __name__ == '__main__':
    main()
