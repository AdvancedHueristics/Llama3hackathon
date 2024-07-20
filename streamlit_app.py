import streamlit as st
import pandas as pd
from pandasai import Agent
from langchain_groq import ChatGroq
from pandasai.responses.streamlit_response import StreamlitResponse
import io, json
from PIL import Image
from together import Together


def get_env_data_as_dict(path: str) -> dict:
    with open(path, 'r') as f:
       return dict(tuple(line.replace('\n', '').split('=')) for line
                in f.readlines() if not line.startswith('#'))

env_dict = get_env_data_as_dict('.env')
client = Together(api_key=env_dict.get("TOGETHER_API_KEY", ""))

def get_env_data_as_dict(path: str) -> dict:
    with open(path, 'r') as f:
        return dict(tuple(line.replace('\n', '').split('=')) for line in f.readlines() if not line.startswith('#'))

# Load environment variables
env_dict = get_env_data_as_dict('.env')
model = ChatGroq(model_name='llama3-70b-8192', api_key=env_dict.get("GROQ_API_KEY", ""))

SYSTEM_PROMPT_TOGETHER = '''You are an AI Query analyzer. Analyze given user query that whether it is related to any CSV, data or charts. If yes then return {"status":True} else then make a reply for the query as polite and helpful member and also state "I am a Data Wizard and I can only help you with Data Science task" return response example as {"status":false, "message":"How are you?"}. If the user query contains Hi or greeting phrases then dont return status as False with relevant reply. The data contains following columns '''


st.title("DataWizard: Your DataScience Assistant")

# Initialize session state for chat history and prompt input
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'prompt_input' not in st.session_state:
    st.session_state.prompt_input = ""

def generate_response():
    prompt = st.session_state.prompt_input
    if prompt:
        # Update chat history with the user's prompt
        st.session_state.chat_history.insert(0, {"role": "user", "text": prompt})
        with st.spinner("Generating response..."):
            try:
                is_question_relevant = client.chat.completions.create(
                    model="mistralai/Mixtral-8x7B-Instruct-v0.1",
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT_TOGETHER + str(list(data.columns))},
                        {"role": "user", "content": prompt}
                        ],
                    response_format ={"type": "json_object"}
                    )
                is_question_relevant = json.loads(is_question_relevant.choices[0].message.content)
                if not is_question_relevant['status']:
                    response = is_question_relevant['message']
                else:
                    response = agent.chat(prompt)

                if isinstance(response, str):
                    if response.endswith(".png"):
                        img_buf = io.BytesIO()
                        with Image.open(response) as image:
                            image.save(img_buf, format='PNG')
                        img_buf.seek(0)
                        st.session_state.chat_history.insert(0, {"role": "assistant", "image": img_buf})
                    else:
                        st.session_state.chat_history.insert(0, {"role": "assistant", "text": response})
                else:
                    st.session_state.chat_history.insert(0, {"role": "assistant", "text": str(response)})
            except Exception as e:
                st.error(f"Failed to generate response: {e}")

# File uploader in sidebar
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write(data.head(3))

    agent = Agent(data, config={
        "llm": model,
        "save_charts": True,
        "verbose": True,
        "response_parser": StreamlitResponse
    })

    # Container for the chat interface
    with st.expander("Chat", expanded=True):
        # Input field for user prompt
        # Input field for user prompt
        prompt = st.text_input("Ask questions about your data:", value="", key="prompt_input", on_change=None)

        # Generate button with callback
        st.button("Generate", on_click=generate_response)
        # Display chat history with the latest message at the top
        if st.session_state.chat_history:
            for message in st.session_state.chat_history:
                if message['role'] == 'user':
                    st.chat_message("user").markdown(message['text'])
                elif 'text' in message:
                    st.chat_message("assistant").markdown(message['text'])
                elif 'image' in message:
                    st.image(message['image'], caption="Generated Chart")