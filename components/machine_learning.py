import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib  # For loading the saved model

def machine_Learning(data):
    st.title("🤖 Machine Learning Workflow")
    st.markdown("---")

    # Overview of Machine Learning Steps
    st.header("🔍 Overview of Machine Learning Steps")
    st.write("""
    This section outlines the key steps taken to train and evaluate machine learning models for fraud detection.
    """)

    # Two Columns for Machine Learning Steps
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("1. Model Selection")
        st.write("""
        - **Algorithms Used**: XGBoost, Random Forest, and LightGBM were selected for their ability to handle imbalanced datasets and high-dimensional data.
        - **Hyperparameters**: Default hyperparameters were used for initial training, with plans for future optimization.
        """)

        st.subheader("2. Model Training")
        st.write("""
        - **Training Data**: The dataset was split into 80% training and 20% testing sets.
        - **Feature Scaling**: Numerical features were standardized using `StandardScaler`.
        - **Training Process**: Each model was trained on the training set using 5-fold cross-validation.
        """)

    with col2:
        st.subheader("3. Model Evaluation")
        st.write("""
        - **Evaluation Metrics**: Accuracy, Precision, Recall, F1-Score, and ROC AUC were used to evaluate model performance.
        - **Confusion Matrix**: Visualized the performance of each model in classifying fraudulent and non-fraudulent transactions.
        - **Learning Curves**: Analyzed the learning curves to assess model performance with varying training set sizes.
        """)

        st.subheader("4. Model Saving")
        st.write("""
        - **Saved Models**: The trained models were saved using `joblib` for future use.
        - **Scaler and Bounds**: The scaler and winsorization bounds were also saved for consistent preprocessing of new data.
        """)

    # Model Performance Metrics
    st.header("📊 Model Performance Metrics")
    st.write("""
    Below are the performance metrics for the trained models:
    """)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("XGBoost Accuracy", "0.9997")
    with col2:
        st.metric("Random Forest Accuracy", "0.9999")
    with col3:
        st.metric("LightGBM Accuracy", "0.9997")

    # Visualizations
    st.header("📈 Visualizations")
    st.write("""
    This section provides an overview of the performance of the trained models.
    """)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.write("**Confusion Matrix for Random Forest:**")
        confusion_matrix = np.array([[56700, 12], [0, 56987]])
        fig, ax = plt.subplots(figsize=(5, 3))  # Smaller plot size
        sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_title("Confusion Matrix")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

        st.write("**Confusion Matrix for XGBoost:**")
        confusion_matrix = np.array([[56678, 34], [0, 56987]])
        fig, ax = plt.subplots(figsize=(5, 3))
        sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_title("Confusion Matrix")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

    with col2:
        st.write("**ROC Curve for Random Forest (Placeholder):**")
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.plot([0, 1], [0, 1], linestyle="--", label="Random Guess")
        ax.set_title("ROC Curve")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend()
        st.pyplot(fig)

        st.write("**Confusion Matrix for LightGBM:**")
        confusion_matrix = np.array([[56685, 27], [3, 56984]])
        fig, ax = plt.subplots(figsize=(5, 3))
        sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_title("Confusion Matrix")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)
    
    with col3:
        st.write("**Learning Curve for Random Forest :**")
        st.image("images/Rf_Learning_Curve.png", caption="Learning Curve for Random Forest", use_container_width=True)
        
        st.write("**Learning Curve for LightGBM :**")
        st.image("images/LightGBM_Learning_Curve.png", caption="Learning Curve for LightGBM", use_container_width=True)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center;">
        <p>Developed with ❤️ by [Bouba Ahmed]</p>
        <p>© 2024 - All rights reserved.</p>
    </div>
    """, unsafe_allow_html=True)

