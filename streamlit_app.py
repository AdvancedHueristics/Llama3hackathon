import streamlit as st
import pandas as pd
from pandasai import Agent
from langchain_groq import ChatGroq
import matplotlib
import matplotlib.pyplot as plt
from pandasai.responses.streamlit_response import StreamlitResponse
import io
from PIL import Image

def get_env_data_as_dict(path: str) -> dict:
    with open(path, 'r') as f:
        return dict(tuple(line.replace('\n', '').split('=')) for line
                    in f.readlines() if not line.startswith('#'))

env_dict = get_env_data_as_dict('.env')
model = ChatGroq(model_name='llama3-70b-8192', api_key=env_dict.get("GROQ_API_KEY", ""))

st.title("DataWizard: Your DataScience Assistant")

# Initialize chat history and prompt input in session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'prompt_input' not in st.session_state:
    st.session_state.prompt_input = ""

if 'clicked' not in st.session_state:
    st.session_state.clicked = False

def click_button():
    st.session_state.clicked = True
    st.rerun()

# File uploader in sidebar
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

def call_agent():
    response = agent.chat(prompt)
    print(response)
    # Handle the response based on its type
    if isinstance(response, str):
        if response.endswith(".png"):
            # Save the plot to a BytesIO object
            img_buf = io.BytesIO()
            image = Image.open(response)
            image.save(img_buf, format='PNG')
            img_buf.seek(0)
            st.session_state.chat_history.insert(0, {"role": "assistant", "image": img_buf})
        else:
            st.session_state.chat_history.insert(0, {"role": "assistant", "text": response})

    else:
        st.session_state.chat_history.insert(0, {"role": "assistant", "text": str(response)})

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write(data.head(3))

    agent = Agent(data,
                  config={"llm": model,
                "save_charts": True,
                "verbose": True,
                "response_parser": StreamlitResponse})

    # Container for the chat interface
    with st.expander("Chat", expanded=True):
        # Input field for user prompt
        prompt = st.text_input("Ask questions about your data:", value=st.session_state.prompt_input, key="prompt_input")

        if st.button("Generate", on_click=call_agent):
            if prompt:
                st.session_state.chat_history.insert(0, {"role": "user", "text": prompt})
                with st.spinner("Generating response..."):
                   call_agent()

                # Clear the input field by updating session state
                #st.session_state.prompt_input = ""

        # Display chat history with latest message at the top
        if st.session_state.chat_history:
            for message in st.session_state.chat_history:
                if message['role'] == 'user':
                    st.chat_message("user").markdown(message['text'])
                elif 'text' in message:
                    st.chat_message("assistant").markdown(message['text'])
                elif 'image' in message:
                    st.image(message['image'], caption="Generated Chart")

