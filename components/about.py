import streamlit as st

def about():
    st.title("About This Project")
    st.markdown("---")

    # Project Overview
    st.header("📌 Project Overview")
    st.write("""
    This project focuses on **detecting fraudulent credit card transactions** using machine learning. 
    Fraud detection is a critical task in the financial industry, and this application aims to provide 
    a reliable and efficient solution to identify suspicious transactions in real-time.
    """)

    # Objectives
    st.header("🎯 Objectives")
    st.write("""
    The main objectives of this project are:
    - To build and evaluate machine learning models for fraud detection.
    - To preprocess and clean the dataset to ensure high-quality input for the models.
    - To deploy the best-performing model in an interactive web application for real-time predictions.
    - To provide users with insights into the data and model performance through visualizations.
    """)

    # Dataset
    st.header("📊 Dataset")
    st.write("""
    The dataset used in this project is the **Credit Card Fraud Detection Dataset** from Kaggle. 
    It contains transactions made by credit cards in September 2013 by European cardholders. 
    """)
    st.markdown("""
    **Key Features:**
    - **V1-V28**: Principal components obtained with PCA (anonymized features).
    - **Amount**: Transaction amount.
    - **Class**: Target variable (1 for fraud, 0 for non-fraud).
    """)
    st.markdown("[Download the dataset from Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)")

    # Methodology
    st.header("🔧 Methodology")
    st.write("""
    The project follows a structured workflow:
    1. **Data Preprocessing:**
        - Handling missing values and duplicates.
        - Winsorization to treat outliers.
        - Standardization of numerical features.
    2. **Model Training:**
        - Three models were trained: **XGBoost**, **Random Forest**, and **LightGBM**.
        - Hyperparameters were tuned for optimal performance.
    3. **Model Evaluation:**
        - Metrics used: Accuracy, Precision, Recall, F1-Score, and ROC AUC.
        - Confusion matrices and learning curves were analyzed.
    4. **Deployment:**
        - The best-performing model was deployed using **Streamlit** for real-time predictions.
    """)

    # Tools and Technologies
    st.header("🛠️ Tools and Technologies")
    st.write("""
    The following tools and technologies were used in this project:
    - **Python**: Primary programming language.
    - **Pandas**: Data manipulation and analysis.
    - **NumPy**: Numerical computations.
    - **Scikit-learn**: Machine learning models and evaluation metrics.
    - **XGBoost**: Gradient boosting framework.
    - **LightGBM**: Lightweight gradient boosting framework.
    - **Matplotlib & Seaborn**: Data visualization.
    - **Streamlit**: Web application deployment.
    """)

    # Team and Credits
    st.header("👥 Team and Credits")
    st.write("""
    This project was developed as part of a machine learning course. Special thanks to:
    - **Kaggle** for providing the dataset.
    - **Streamlit** for making deployment easy and interactive.
    - **Open-source contributors** for maintaining the libraries used in this project.
    """)

    # Contact Information
    st.header("📧 Contact Information")
    st.write("""
    For questions, feedback, or collaboration opportunities, feel free to reach out:
    - **Email**: ahmedbouba383@gmail.com
    - **GitHub**: [Bouba Ahmed](https://github.com/BoubaAhmed)
    - **LinkedIn**: [Bouba Ahmed](https://www.linkedin.com/in/ahmed-bouba-861b91231/)
    """)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center;">
        <p>Developed with ❤️ by [Bouba Ahmed]</p>
        <p>© 2024 - All rights reserved.</p>
    </div>
    """, unsafe_allow_html=True)
