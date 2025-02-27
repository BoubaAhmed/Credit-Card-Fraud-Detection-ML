import streamlit as st

st.set_page_config(
    page_title="Credit Card Fraud Detection Classification", 
    page_icon="assets/icon/icon.png",
    layout="wide",
    initial_sidebar_state="expanded"
) 
import pandas as pd
import altair as alt
from components.about import about
from components.preprocessing import preprocessing
from components.prediction import prediction
from components.dashboard import dashboard
from components.machine_learning import machine_Learning
from components.conclusion import conclusion
from sklearn.utils import shuffle
# from tensorflow import ts


alt.themes.enable("dark")

if 'page_selection' not in st.session_state:
    st.session_state.page_selection = 'about' 

def set_page_selection(page):
    st.session_state.page_selection = page

# Load data
try:
    dataset = pd.read_csv('data/raw/creditcard_2023.csv', delimiter=',')
    dataset_Test = pd.read_csv('data/processed/X_test.csv', delimiter=',')
    df = shuffle(dataset)
    df = df.reset_index(drop=True)
    
except FileNotFoundError:
    st.error("Dataset not found. Ensure 'creditcard_2023.csv' is available in the app directory.")
    st.stop()

with st.sidebar:

    st.title('Credit Card Fraud Detection Classification')

    st.subheader("Pages")
    st.button("Dashboard", use_container_width=True, on_click=set_page_selection, args=('dashboard',))
    st.button("About", use_container_width=True, on_click=set_page_selection, args=('about',))
    st.button("Pre-processing", use_container_width=True, on_click=set_page_selection, args=('data_cleaning',))
    st.button("Machine Learning", use_container_width=True, on_click=set_page_selection, args=('machine_learning',))
    st.button("Prediction", use_container_width=True, on_click=set_page_selection, args=('prediction',))
    st.button("Conclusion", use_container_width=True, on_click=set_page_selection, args=('conclusion',))

    # Project Details
    st.subheader("Abstract")
    st.markdown("A Streamlit dashboard highlighting the results of training classification models using the Credit Card dataset from Kaggle.")
    st.markdown("📊 [Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)")
    st.markdown("📗 [Google Colab Notebook](https://colab.research.google.com/drive/1GK47GWhSWRis5C-mX2Ck66K0rEsuNgqV?usp=sharing)")
    st.markdown("🐙 [GitHub Repository](https://github.com/BoubaAhmed/Credit-Card-Fraud-Detection-ML)")
    st.markdown("by: [`Bouba Ahmed`]")

# Render pages based on session state
if st.session_state.page_selection == 'dashboard':
    dashboard(df)

if st.session_state.page_selection == 'about':
    about()
    
elif st.session_state.page_selection == 'data_cleaning':
    preprocessing(df)

elif st.session_state.page_selection == 'machine_learning':
    machine_Learning(df)

elif st.session_state.page_selection == 'prediction':
    prediction(dataset_Test)

elif st.session_state.page_selection == 'conclusion':
    conclusion()