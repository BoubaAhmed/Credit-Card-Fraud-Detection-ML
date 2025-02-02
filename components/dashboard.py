import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def dashboard(data):
    st.title("📊 Fraud Detection Dashboard")
    st.markdown("---")

    # Key Metrics
    st.header("🔑 Key Metrics")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Transactions", data.shape[0])
    with col2:
        st.metric("Fraudulent Transactions", data[data["Class"] == 1].shape[0])
    with col3:
        st.metric("Non-Fraudulent Transactions", data[data["Class"] == 0].shape[0])

    # Dataset Overview
    st.header("📂 Dataset Overview")
    st.write("""
    This section provides a quick overview of the dataset, including its structure and basic statistics.
    """)
    st.write("**Preview of the Dataset:**")
    st.dataframe(data.head())


    col1, col2, col3 = st.columns(3)
    with col1:
        st.write("**Dataset Shape:**")
        st.write(f"Number of Rows: {data.shape[0]}")
        st.write(f"Number of Columns: {data.shape[1]}")

    with col2:
        missing_values = data.isnull().sum().to_frame().T 
        st.write("**Missing Values per Column:**")
        st.dataframe(missing_values)

    with col3:
        column_types = data.dtypes.to_frame().T 
        st.write("**Column Names and Data Types:**")
        st.dataframe(column_types) 

    # Basic Statistics
    st.header("📈 Basic Statistics")
    st.write("""
    This section provides descriptive statistics for the numerical features in the dataset.
    """)
    st.write("**Summary Statistics:**")
    st.dataframe(data.describe())

    # Class Distribution
    st.header("📉 Class Distribution")
    st.write("""
    The target variable `Class` indicates whether a transaction is fraudulent (1) or not (0). 
    This section visualizes the distribution of the target variable.
    """)

    # Two Plots in a Row: Class Distribution and Feature Distribution
    col1, col2 = st.columns(2)
    with col1:
        class_counts = data["Class"].value_counts()
        st.write("**Class Counts:**")
        st.write(class_counts)
        

    with col2:
        fig, ax = plt.subplots(figsize=(5, 3))  # Smaller plot size
        sns.countplot(x="Class", data=data, ax=ax)
        ax.set_title("Distribution of Transactions")
        ax.set_xlabel("Class (0: Non-Fraud, 1: Fraud)")
        ax.set_ylabel("Count")
        st.pyplot(fig)


    col1, col2 = st.columns(2)
    with col1: 
        # Feature Distributions
        st.header("📊 Feature Distributions")
        st.write("""
        This section allows you to explore the distribution of individual features.
        """)
        feature = st.selectbox("Select a Feature to Visualize", data.columns)
        if feature:
            fig, ax = plt.subplots()
            sns.histplot(data[feature], kde=True, ax=ax)
            ax.set_title(f"Distribution of {feature}")
            ax.set_xlabel(feature)
            ax.set_ylabel("Frequency")
            st.pyplot(fig)

    with col2:
        # Correlation Matrix
        st.header("🔗 Correlation Matrix")
        st.write("""
        This section displays the correlation matrix for the numerical features in the dataset. 
        Correlation helps identify relationships between features.
        """)
        corr = data.corr()
        fig, ax = plt.subplots(figsize=(8, 6))  # Smaller plot size
        sns.heatmap(corr, annot=False, cmap="coolwarm", ax=ax)
        ax.set_title("Correlation Matrix")
        st.pyplot(fig)



    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center;">
        <p>Developed with ❤️ by [Bouba Ahmed]</p>
        <p>© 2024 - All rights reserved.</p>
    </div>
    """, unsafe_allow_html=True)

