import streamlit as st
from main import get_response
import pandas as pd
import io

st.title('NLP Data Science Application')

# File uploader widget
uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx", "txt"])

if uploaded_file is not None:
    # Read the file based on type
    if uploaded_file.type == "text/csv":
        df = pd.read_csv(uploaded_file)
        st.write("CSV file uploaded successfully!")
    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
        df = pd.read_excel(uploaded_file)
        st.write("Excel file uploaded successfully!")
    elif uploaded_file.type == "text/plain":
        content = uploaded_file.read().decode("utf-8")
        st.write("Text file uploaded successfully!")
        df = None  # No need for DataFrame, just raw text content
    else:
        st.error("Unsupported file type.")
        df = None

    # Option to ask questions
    if df is not None:
        st.write("You can ask questions about the file.")
        question = st.text_input("Enter your question:")

        if st.button("Submit Question"):
            if question:
                # Process file content and question
                if df is not None:
                    # For demonstration, converting DataFrame to text
                    file_content = df.to_string()
                    full_input = f"File content:\n{file_content}\n\nQuestion: {question}"
                else:
                    # Use raw text content
                    full_input = f"File content:\n{content}\n\nQuestion: {question}"

                # Get response from NLP model
                response = get_response(full_input)
                st.write("Response:")
                st.write(response)
            else:
                st.warning("Please enter a question before submitting.")
