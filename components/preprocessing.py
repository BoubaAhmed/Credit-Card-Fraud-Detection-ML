import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def preprocessing(data):
    st.title("🛠️ Data Preprocessing")
    st.markdown("---")

    # Overview of Preprocessing Steps
    st.header("🔍 Overview of Preprocessing Steps")
    st.write("""
    This section outlines the key steps taken to preprocess the dataset before training the machine learning models.
    """)

    # Two Columns for Preprocessing Steps
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("1. Data Cleaning")
        st.write("""
        - **Removed Unnecessary Columns**: Dropped the `id` column as it does not provide useful information for modeling.
        - **Handled Missing Values**: Checked for missing values and removed any rows with incomplete data.
        - **Deduplication**: Identified and removed duplicate rows to ensure data integrity.
        """)

        st.subheader("2. Handling Outliers")
        st.write("""
        - **Winsorization**: Applied winsorization to limit the impact of outliers. Values below the 1\textsuperscript{st} percentile and above the 99\textsuperscript{th} percentile were replaced with the respective percentile values.
        - **Saved Bounds**: The winsorization bounds were saved for future use on new data.
        """)

    with col2:
        st.subheader("3. Feature Engineering")
        st.write("""
        - **Feature Selection**: Selected only the relevant features for modeling (e.g., `V1-V28`, `Amount`, `Class`).
        - **Target Variable**: The target variable `Class` was kept for supervised learning.
        """)

        st.subheader("4. Data Splitting and Normalization")
        st.write("""
        - **Train-Test Split**: The dataset was split into training (80%) and testing (20%) sets.
        - **Normalization**: Numerical features were standardized using `StandardScaler` to ensure all features are on the same scale.
        - **Saved Scaler**: The scaler was saved for consistent preprocessing of new data.
        """)

    # Visualizations (Optional)
    st.header("📊 Visualizations")
    st.write("""
    Below are some visualizations of the preprocessing steps:
    """)
    numeric_features = data.select_dtypes(include=["float64", "int64"]).columns
    feature = st.selectbox("Select a Feature for Visualize", numeric_features)

    col1, col2, col3 = st.columns(3)

    with col1:
        if feature:
            st.write("**Before Winsorization:**")
            # Placeholder for a plot showing data before winsorization
            fig, ax = plt.subplots(figsize=(5, 3))
            sns.histplot(data[feature], kde=True, ax=ax)
            ax.set_title(f"Distribution of {feature} (Before Winsorization)")
            ax.set_xlabel(feature)
            ax.set_ylabel("Frequency")
            st.pyplot(fig)

    with col2:
        if feature:
            st.write("**After Winsorization:**")
            # Placeholder for a plot showing data after winsorization
            fig, ax = plt.subplots(figsize=(5, 3))
            sns.histplot(data[feature].clip(lower=data[feature].quantile(0.01), upper=data[feature].quantile(0.99)), kde=True, ax=ax)
            ax.set_title(f"Distribution of {feature} (After Winsorization)")
            ax.set_xlabel(feature)
            ax.set_ylabel("Frequency")
            st.pyplot(fig)

    with col3:
        if feature:
            st.write(f"**Outlier Detection of {feature}:**")
            fig, ax = plt.subplots(figsize=(5, 3))
            sns.boxplot(x=data[feature], ax=ax)
            ax.set_title(f"Boxplot of {feature}")
            ax.set_xlabel(feature)
            st.pyplot(fig)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center;">
        <p>Developed with ❤️ by [Bouba Ahmed]</p>
        <p>© 2024 - All rights reserved.</p>
    </div>
    """, unsafe_allow_html=True)

