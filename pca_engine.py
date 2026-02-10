import numpy as np

class PCAFromScratch:
    """
    Principal Component Analysis (PCA) implementation from scratch.
    """
    def __init__(self, n_components=2):
        self.n_components = n_components
        self.components = None
        self.mean = None
        self.eigenvalues = None
        self.explained_variance_ratio = None

    def fit(self, X):
        # 1. Mean centering
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean

        # 2. Covariance matrix calculation
        # Covariance = (1 / (n-1)) * (X_centered.T @ X_centered)
        cov_matrix = np.cov(X_centered.T)

        # 3. Eigen decomposition
        # eigenvalues: magnitude of variance
        # eigenvectors: direction of components
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        # 4. Sort eigenvalues and eigenvectors in descending order
        # eigh returns them sorted ascending, so we reverse
        idxs = np.argsort(eigenvalues)[::-1]
        self.eigenvalues = eigenvalues[idxs]
        self.components = eigenvectors.T[idxs]

        # 5. Calculate explained variance ratio
        total_var = np.sum(self.eigenvalues)
        self.explained_variance_ratio = self.eigenvalues / total_var

        # Store only the requested number of components
        self.components = self.components[:self.n_components]

    def transform(self, X):
        # Project data onto principal components
        X_centered = X - self.mean
        return np.dot(X_centered, self.components.T)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
