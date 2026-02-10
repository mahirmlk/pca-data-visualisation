# PCA for 2D/3D Data Visualization

This project implements **Principal Component Analysis (PCA)** from scratch using linear algebra principles and visualizes the results through an interactive web dashboard. It specifically focuses on reducing the high-dimensional Iris dataset into intuitive 2D and 3D clusters.

##  Overview
PCA is a fundamental technique in Machine Learning for dimensionality reduction. It allows us to take a dataset with many features (high dimensions) and compress it into fewer features (low dimensions) while retaining the maximum possible information (variance).

##  The Mathematics (Step-by-Step)
This project avoids high-level libraries like `scikit-learn` for the PCA logic, implementing the following mathematical steps using only `NumPy`:

1.  **Standardization (Mean Centering):**
    We shift the data so that it is centered around the origin.
    $$X_{centered} = X - \mu$$

2.  **Covariance Matrix Calculation:**
    We compute the covariance matrix to understand the relationships between different features.
    $$Cov(X) = \frac{1}{n-1} X_{centered}^T X_{centered}$$

3.  **Eigenvalue & Eigenvector Decomposition:**
    We solve for the "Principal Components":
    - **Eigenvalues:** Represent the amount of variance captured by each component.
    - **Eigenvectors:** Represent the direction of the new axes in the feature space.

4.  **Projection:**
    We select the top $k$ eigenvectors (where $k=2$ or $3$) and project the original data onto them to get the reduced coordinates.

##  Features
- **From-Scratch Implementation:** Pure NumPy logic in `pca_engine.py`.
- **Interactive Dashboard:** Built with `Streamlit` and `Plotly`.
- **2D & 3D Visualization:** Toggle between dimensions to see how clusters form.
- **Variance Analysis:** Includes a Scree Plot and Cumulative Explained Variance tracking.
- **Iris Dataset Integration:** Automatically fetches the classic Iris dataset for demonstration.

##  Project Structure
- `pca_engine.py`: The core mathematical implementation.
- `app.py`: The Streamlit web interface and visualization logic.
- `requirements.txt`: Project dependencies.
- `README.md`: Documentation.

##  How to Run

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Application:**
   ```bash
   streamlit run app.py
   ```

##  Concepts Demonstrated
- **Linear Algebra:** Matrix multiplication, Transposition, Eigendecomposition.
- **Statistics:** Mean, Variance, Covariance.
- **Data Visualization:** Mapping high-dimensional relationships to visual space.
