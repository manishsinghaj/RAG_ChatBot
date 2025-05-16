import streamlit as st
from utils.embedding import extract_text_from_pdf, chunk_text
from utils.chroma_handler import store_chunks_in_chroma, query_chroma
from utils.rag_prompt_builder import build_prompt
from config import get_gemini_response
import os

st.title("📚 Personalized RAG Chatbot")

# Step 1: User Profile
st.sidebar.header("🎯 Personalization Settings")
user_profile = {
    "tone": st.sidebar.selectbox("Tone", ["formal", "friendly", "humorous"]),
    "goal": st.sidebar.selectbox("Goal", ["educate", "summarize", "advise", "entertain"]),
    "length": st.sidebar.selectbox("Length", ["short", "detailed"]),
    "style": st.sidebar.selectbox("Style", ["storytelling", "bullet-points", "step-by-step"]),
    "persona": st.sidebar.selectbox("Persona", ["beginner", "expert", "10-year-old", "student"]),
}

# Step 2: Document Upload
uploaded_file = st.file_uploader("Upload reference PDF", type=["pdf"])
if uploaded_file:
    file_path = f"documents/{uploaded_file.name}"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())

    text = extract_text_from_pdf(file_path)
    chunks = chunk_text(text)
    store_chunks_in_chroma(chunks)
    st.success("Document uploaded and indexed.")

# Step 3: Ask Query
query = st.text_input("Ask your question")

if st.button("Get Answer"):
    with st.spinner("Generating response..."):
        context = query_chroma(query)
        prompt = build_prompt(query, context, user_profile)
        answer = get_gemini_response(prompt)
        st.markdown("### 💬 Response:")
        st.write(answer)
