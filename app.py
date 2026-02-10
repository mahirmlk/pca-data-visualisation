import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pca_engine import PCAFromScratch

# Set page config
st.set_page_config(page_title="PCA from Scratch Visualization", layout="wide")

st.title("PCA from Scratch: Data Visualization")
st.markdown("""
Principal Component Analysis (PCA) is a powerful technique for dimensionality reduction. 
It transforms high-dimensional data into a lower-dimensional space while preserving as much variance as possible.
""")

# Load Dataset
@st.cache_data
def load_data():
    # Loading Iris dataset from a public URL to avoid scikit-learn dependency
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    columns = ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]
    df = pd.read_csv(url, names=columns)
    return df

try:
    df = load_data()
    
    # Sidebar configuration
    st.sidebar.header("Settings")
    n_dims = st.sidebar.radio("Target Dimensions", [2, 3], index=0)
    
    features = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    X = df[features].values
    y = df["species"]

    # Initialize and run PCA
    pca = PCAFromScratch(n_components=n_dims)
    X_projected = pca.fit_transform(X)

    # Create a results dataframe
    pca_cols = [f"PC{i+1}" for i in range(n_dims)]
    df_pca = pd.DataFrame(X_projected, columns=pca_cols)
    df_pca["species"] = y

    # Layout: Two columns
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader(f"PCA Result ({n_dims}D)")
        if n_dims == 2:
            fig = px.scatter(
                df_pca, x="PC1", y="PC2", color="species",
                title="Iris Dataset Projected to 2D",
                labels={"PC1": "First Principal Component", "PC2": "Second Principal Component"},
                template="plotly_dark"
            )
        else:
            fig = px.scatter_3d(
                df_pca, x="PC1", y="PC2", z="PC3", color="species",
                title="Iris Dataset Projected to 3D",
                labels={"PC1": "PC1", "PC2": "PC2", "PC3": "PC3"},
                template="plotly_dark"
            )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Explained Variance")
        # Show variance ratio
        var_df = pd.DataFrame({
            "Component": [f"PC{i+1}" for i in range(len(pca.explained_variance_ratio))],
            "Explained Variance Ratio": pca.explained_variance_ratio
        })
        st.dataframe(var_df.style.format({"Explained Variance Ratio": "{:.2%}"}))
        
        # Cumulative variance plot
        cum_var = np.cumsum(pca.explained_variance_ratio)
        fig_var = px.line(
            x=[f"PC{i+1}" for i in range(len(cum_var))],
            y=cum_var,
            title="Cumulative Explained Variance",
            labels={"x": "Components", "y": "Cumulative Variance Ratio"},
            markers=True,
            template="plotly_dark"
        )
        st.plotly_chart(fig_var, use_container_width=True)

    # Mathematical Steps Detail
    with st.expander("How the math works (Step-by-Step)"):
        st.write("### 1. Mean Centering")
        st.write("We subtract the mean of each feature from the data.")
        st.code("X_centered = X - np.mean(X, axis=0)")
        
        st.write("### 2. Covariance Matrix")
        st.write("We calculate how each feature varies with respect to others.")
        st.latex(r"Cov(X) = \frac{1}{n-1} X^T X")
        
        st.write("### 3. Eigendecomposition")
        st.write("We find the eigenvalues (variance magnitude) and eigenvectors (directions).")
        st.code("eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)")
        
        st.write("### 4. Projection")
        st.write("We project the original data onto the top eigenvectors.")
        st.code("X_projected = np.dot(X_centered, top_eigenvectors.T)")

    st.subheader("Raw Data Preview")
    st.dataframe(df.head())

except Exception as e:
    st.error(f"Failed to load or process data: {e}")
    st.info("Check your internet connection to fetch the Iris dataset.")
