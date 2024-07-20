import streamlit as st 
import pandas as pd
from pandasai import Agent
from langchain_groq import ChatGroq


def get_env_data_as_dict(path: str) -> dict:
    with open(path, 'r') as f:
       return dict(tuple(line.replace('\n', '').split('=')) for line
                in f.readlines() if not line.startswith('#'))

env_dict = get_env_data_as_dict('.env') 

model = ChatGroq(model_name = 'llama3-70b-8192',api_key = env_dict.get("GROQ_API_KEY", ""))

st.title("DataWizard: Your DataScience Assistant")

uploaded_file = st.sidebar.file_uploader(
    "Upload a CSV file",
    type=["csv"]
)

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write(data.head(3))

    agent = Agent(data, config={"llm": model})
    prompt = st.text_input("Enter your prompt:")

    if st.button("Generate"):
        if prompt:
            with st.spinner("Generating response..."):
                response = agent.chat(prompt)
                if response.endswith("temp_chart.png"):
                    st.image(response)
                else:
                    st.write(response)

                