import streamlit as st
# from langchain_chroma import Chroma
from query_rag import get_context, generate_prompt, query_rag
from pathlib import Path

st.title("Lords of the rings QA 📚")
st.write("This is a simple question answering app based on the Lord of the Rings (4) books including\n The Hobbit |\nThe Fellowship of the Ring |\nThe Return of the King |\n and The Two Towers..")
question = st.chat_input("Ask a question", key="question")
st.write("##### Question:")

if question:
    st.write(question)
    st.write("##### Answer:")
    with st.spinner("Searching..."):
        context_text, result = get_context(question)
        prompt = generate_prompt(question, context_text)
        response= query_rag(prompt, result)
        st.write(response)

