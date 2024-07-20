import streamlit as st
import pandas as pd
from pandasai import Agent
from langchain_groq import ChatGroq
import matplotlib.pyplot as plt
import io
from src.preprocessing import *

def get_env_data_as_dict(path: str) -> dict:
    with open(path, 'r') as f:
        return dict(tuple(line.replace('\n', '').split('=')) for line
                    in f.readlines() if not line.startswith('#'))

env_dict = get_env_data_as_dict('.env')
model = ChatGroq(model_name='llama3-70b-8192', api_key=env_dict.get("GROQ_API_KEY", ""))



# Warning control
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

# Streamlit app
def main():
    st.title("DataWizard: Your DataScience Assistant")

    # Sidebar navigation
    st.sidebar.title("Data Science Steps")
    option = st.sidebar.selectbox("Select a step:", 
                                  ("Dataset Understanding", "Data Cleaning", "Exploratory Data Analysis (EDA)", "Feature Engineering", "Visualization", "Modeling", "Chat with Bot"))

    # File upload widget
    uploaded_file = st.file_uploader("Upload your dataset", type=["csv", "xlsx"])
    if uploaded_file:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith('.xlsx'):
                df = pd.read_excel(uploaded_file)

            st.write("Dataset Preview:")
            st.dataframe(df.head())

            if option == "Dataset Understanding":
                st.subheader("Dataset Understanding")
                st.write("### Dataset Info")
                buffer = StringIO()
                df.info(buf=buffer)
                s = buffer.getvalue()
                st.text(s)
                st.write("### Dataset Shape")
                st.write(df.shape)
                st.write("### Dataset Columns")
                st.write(df.columns)
                st.write("### Missing Values")
                st.write(df.isnull().sum())
                st.write("### Duplicate Rows")
                st.write(df.duplicated().sum())

            elif option == "Data Cleaning":
                st.subheader("Data Cleaning")

                # Missing Values Handling
                st.write("### Handle Missing Values")
                missing_method = st.selectbox("Select a method:", ["Drop Rows", "Drop Columns", "Simple", "KNN"])
                if st.button("Handle Missing Values"):
                    df = handle_missing_values(df, missing_method)
                    st.write("### Missing Values Handled Successfully!")
                    st.dataframe(df.head())

                # Remove Duplicates
                st.write("### Remove Duplicates")
                if st.button("Remove Duplicates"):
                    df = remove_duplicates(df)
                    st.write("### Duplicates Removed Successfully!")
                    st.dataframe(df.head())

                # Encode Categorical Variables
                st.write("### Encode Categorical Variables")
                encoding_method = st.selectbox("Select an encoding method:", ["Label", "OneHot"])
                if st.button("Encode Categorical Variables"):
                    df = encode_categorical(df, encoding_method)
                    st.write("### Categorical Variables Encoded Successfully!")
                    st.dataframe(df.head())

                # Handle Outliers
                st.write("### Handle Outliers")
                outlier_method = st.selectbox("Select an outlier handling method:", ["IQR", "Z-Score"])
                if st.button("Handle Outliers"):
                    df = handle_outliers(df, outlier_method)
                    st.write("### Outliers Handled Successfully!")
                    st.dataframe(df.head())

                # Scale and Normalize
                st.write("### Scale and Normalize")
                scaling_method = st.selectbox("Select a scaling method:", ["Standard", "MinMax", "Robust"])
                if st.button("Scale and Normalize"):
                    df = scale_and_normalize(df, scaling_method)
                    st.write("### Scaling and Normalization Done Successfully!")
                    st.dataframe(df.head())

            elif option == "Chat with Bot":
                st.subheader("Chat with Bot")
                user_input = st.text_input("Ask a question about data science:")
                if user_input:
                    response = chat_with_bot(user_input)
                    st.write("Bot Response:")
                    st.write(response)
        except Exception as e:
            st.error(f"Error loading dataset: {e}")
            logger.error(f"Error loading dataset: {e}")

if __name__ == "__main__":
    main()
