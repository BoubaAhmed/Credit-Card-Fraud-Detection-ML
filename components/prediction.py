import streamlit as st
import pandas as pd
import joblib

def preprocess_data(df):
    """Apply the same preprocessing as used during model training, including Winsorization and scaling."""
    # Load Winsorization bounds and apply outlier removal
    winsorization_bounds = joblib.load("models/winsorization_bounds.pkl")  # Load the saved bounds
    
    # Apply Winsorization to each numerical feature
    num_features = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    for col in num_features:
        lower, upper = winsorization_bounds[col]  # Get the bounds for the column
        df[col] = df[col].clip(lower=lower, upper=upper)  # Apply Winsorization
    
    # Load the scaler and apply it
    scaler = joblib.load("models/scaler.pkl")  # Load the same scaler used in training
    df_scaled = scaler.transform(df)  # Apply transformation
    
    return df_scaled

def prediction(data):
    # Load the trained model
    model = joblib.load("models/RandomForestClassifier.pkl")

    # Streamlit UI
    st.title("🔍 Credit Card Fraud Detection")
    st.markdown("---")

    # Toggle between Batch Prediction and Manual Prediction
    prediction_mode = st.radio(
        "Select Prediction Mode:",
        ["Batch Prediction", "Manual Prediction"],
        index=0  # Default to Batch Prediction
    )

    if prediction_mode == "Batch Prediction":
        st.header("📂 Batch Prediction")
        st.write("""
        Upload a CSV file with 29 features to detect fraudulent transactions, or use the default dataset.
        """)

        # Use default dataset or upload a new one
        use_default_data = st.checkbox("Use Default Dataset", value=True)
        if use_default_data:
            df = data.drop(columns=["Class"], errors='ignore')
        else:
            uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
            if uploaded_file:
                df = pd.read_csv(uploaded_file)
            else:
                st.warning("Please upload a CSV file or use the default dataset.")
                return

        # Check if columns match expected features
        expected_features = 29
        if df.shape[1] != expected_features:
            st.error(f"⚠️ The uploaded file must have {expected_features} columns. Found {df.shape[1]}!")
        else:
            # Preprocess data (Winsorization + scaling)
            # df_processed = preprocess_data(df)
            df_processed = df

            # Make predictions
            predictions = model.predict(df_processed)

            # Add predictions to DataFrame
            df["Fraud Prediction"] = predictions

            # Show results
            st.subheader("🔹 Prediction Results")
            st.write(df.head(20))  # Show first few rows of the DataFrame

            # Provide a download link for predictions
            csv_result = df.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download Predictions", csv_result, "predictions.csv", "text/csv")

            # Display summary counts
            fraud_count = (predictions == 1).sum()
            normal_count = (predictions == 0).sum()
            st.write(f"✅ **Normal Transactions**: {normal_count}")
            st.write(f"⚠️ **Fraudulent Transactions**: {fraud_count}")

    elif prediction_mode == "Manual Prediction":
        st.header("✍️ Manual Prediction")
        st.write("""
        Enter the feature values manually to perform a single prediction. 
        Please provide values for all 29 features (V1 to V28 and Amount).
        """)

        # Create input fields for each feature
        st.write("**Enter Feature Values:**")
        
        # Define the required features explicitly
        required_features = [f"V{i}" for i in range(1, 29)] + ["Amount"]
        
        # Create a dictionary to store user inputs
        input_data = {}
        for feature in required_features:
            input_data[feature] = st.number_input(feature, value=0.0)

        if st.button("Predict"):
            # Convert input data to a DataFrame
            input_df = pd.DataFrame([input_data])

            # Preprocess the input data
            input_processed = preprocess_data(input_df)

            # Make prediction
            prediction = model.predict(input_processed)
            prediction_proba = model.predict_proba(input_processed)[:, 1]

            # Display results
            st.subheader("🔹 Prediction Results")
            st.write(f"Predicted Class: **{'Fraud' if prediction[0] == 1 else 'Non-Fraud'}**")
            st.write(f"Prediction Probability: **{prediction_proba[0]:.4f}**")

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center;">
        <p>Developed with ❤️ by [Bouba Ahmed]</p>
        <p>© 2024 - All rights reserved.</p>
    </div>
    """, unsafe_allow_html=True)
