import os
#this is the correct module for csv chat
import streamlit as st
from dotenv import load_dotenv
import google.generativeai as genai
import pandas as pd
import tempfile
import time
from datetime import datetime
import random

# Load environment variables
load_dotenv()

# Configure Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Set up the model
generation_config = {
    "temperature": 0.7,
    "top_p": 1,
    "top_k": 40,
    "max_output_tokens": 4096,
}

safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
]

model = genai.GenerativeModel(
    model_name="gemini-2.0-flash",
    generation_config=generation_config,
    safety_settings=safety_settings,
)

# Human-like responses
greeting_responses = [
    "Hello! I'm ready to analyze your data. What would you like to know?",
    "Hi there! I've processed your CSV file and I'm here to help.",
    "Good {time_of_day}! Let's explore your data together."
]

redirect_responses = [
    "I'm focused on your dataset. What would you like to know about it?",
    "Let's discuss the data. What insights are you looking for?",
    "I'd be happy to answer questions about this dataset specifically."
]

# Initialize session state for CSV chat
if "csv_chat" not in st.session_state:
    st.session_state.csv_chat = {
        "messages": [],
        "file_processed": False,
        "gemini_file": None,
        "chat_session": None,
        "df": None,
        "show_data_preview": False  # Initialize here
    }

# Streamlit UI for CSV Analysis
st.title("📊 CSV Data Analysis Assistant")
st.caption("Upload a CSV file and have a conversation about your data")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

# Toggle for data preview - only show if we have data
if st.session_state.csv_chat["df"] is not None:
    st.session_state.csv_chat["show_data_preview"] = st.checkbox(
        "Show data preview", 
        value=st.session_state.csv_chat["show_data_preview"]
    )
    
    if st.session_state.csv_chat["show_data_preview"]:
        st.subheader("Data Preview")
        st.dataframe(st.session_state.csv_chat["df"].head())

# Handle CSV upload and processing
if uploaded_file and not st.session_state.csv_chat["file_processed"]:
    with st.spinner("Processing your CSV file..."):
        try:
            # Read CSV into DataFrame
            df = pd.read_csv(uploaded_file)
            st.session_state.csv_chat["df"] = df
            
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
                df.to_csv(tmp.name, index=False)
                tmp_path = tmp.name
            
            # Upload to Gemini
            st.session_state.csv_chat["gemini_file"] = genai.upload_file(tmp_path, mime_type="text/csv")
            
            # Wait for file to process
            while st.session_state.csv_chat["gemini_file"].state.name == "PROCESSING":
                time.sleep(3)
                st.session_state.csv_chat["gemini_file"] = genai.get_file(st.session_state.csv_chat["gemini_file"].name)
            
            # Initialize chat session
            st.session_state.csv_chat["chat_session"] = model.start_chat(history=[
                {
                    "role": "user",
                    "parts": [
                        st.session_state.csv_chat["gemini_file"],
                        "You are a data analysis assistant. Provide accurate answers "
                        "strictly based on the dataset. Be professional yet friendly. "
                        "For data questions, include relevant statistics. For visualization "
                        "requests, describe the chart that would best represent the answer. "
                        "If information isn't in the data, say so. For irrelevant questions "
                        "Don't provide the same responses, change the sentence if the same questions was asked repeatedly"
                        "gently redirect to the dataset content."
                    ]
                }
            ])
            
            st.session_state.csv_chat["file_processed"] = True
            
            # Get column information
            col_info = "\n".join([f"- {col}" for col in df.columns.tolist()])
            
            # Add greeting with column info
            greeting = random.choice(greeting_responses).format(
                time_of_day="morning" if datetime.now().hour < 12 else "afternoon"
            )
            greeting += f"\n\nI've identified these columns in your data:\n{col_info}"
            
            st.session_state.csv_chat["messages"].append({
                "role": "assistant",
                "content": greeting
            })
            st.rerun()
            
        except Exception as e:
            st.error(f"Error processing CSV: {e}")

# Display chat messages
for msg in st.session_state.csv_chat["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Handle user input
if prompt := st.chat_input("Ask about your data..."):
    if not st.session_state.csv_chat["file_processed"]:
        st.warning("Please upload a CSV file first")
        st.stop()
    
    # Add user message to history
    st.session_state.csv_chat["messages"].append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Analyzing data..."):
            try:
                response = st.session_state.csv_chat["chat_session"].send_message(
                    f"{prompt}\n\nProvide a detailed response based on the data. "
                    "If appropriate, suggest what kind of visualization would best "
                    "represent the answer and include relevant statistics."
                )
                response_text = response.text
                
                if not response_text.strip():
                    response_text = "I couldn't find relevant information in the data about that."
                
            except Exception as e:
                response_text = f"An error occurred: {str(e)}"
            
            st.markdown(response_text)
    
    # Add assistant response to history
    st.session_state.csv_chat["messages"].append({"role": "assistant", "content": response_text})

# Add control buttons
if st.session_state.csv_chat["messages"]:
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Clear Conversation", key="clear_csv"):
            st.session_state.csv_chat["messages"] = []
            st.rerun()

    with col2:
        if st.session_state.csv_chat["file_processed"] and st.button("Generate Summary", key="summary_csv"):
            with st.spinner("Generating comprehensive summary..."):
                try:
                    summary = st.session_state.csv_chat["chat_session"].send_message(
                        "Provide a comprehensive summary of this dataset including: "
                        "1. Overview of columns and data types\n"
                        "2. Key statistics for numerical columns\n"
                        "3. Interesting patterns or anomalies\n"
                        "4. Potential analysis directions\n"
                        "Format the response with clear headings."
                    )
                    st.session_state.csv_chat["messages"].append({
                        "role": "assistant",
                        "content": summary.text
                    })
                    st.rerun()
                except Exception as e:
                    st.error(f"Error generating summary: {str(e)}")
