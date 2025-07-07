import streamlit as st
from rag_pipline import (
    extract_text_from_pdf,
    split_text_into_chunks,
    build_faiss_vectorstore,
    load_faiss_vectorstore,
    get_context_from_query,
    generate_response
)
from langchain_google_genai import GoogleGenerativeAIEmbeddings, GoogleGenerativeAI
import os


def main():
    st.title(" RAG PDF Chatbot with Gemini")
    st.write("Upload a PDF, ask questions, and get answers powered by Gemini.")

    api_key = st.text_input(" Enter your Google API Key:", type="password")

    if not api_key:
        st.warning("Please enter your Google API key to continue.")
        return

    uploaded_pdf = st.file_uploader(" Upload PDF", type=["pdf"])

    if uploaded_pdf:
        if st.button(" Process PDF"):
            with st.spinner("Extracting text from PDF..."):
                text = extract_text_from_pdf(uploaded_pdf)

            with st.spinner("Splitting text into chunks..."):
                chunks = split_text_into_chunks(text)

            with st.spinner("Generating embeddings and building FAISS index..."):
                embeddings = GoogleGenerativeAIEmbeddings(
                    model="models/embedding-001",
                    google_api_key=api_key
                )
                build_faiss_vectorstore(chunks, embeddings)

            st.success("PDF processed and vectorstore created!")

    query = st.text_input(" Enter your question:")

    
    if query:
        if st.button(" Get Answer"):
            with st.spinner("Loading FAISS and searching for context..."):
                embeddings = GoogleGenerativeAIEmbeddings(
                    model="models/embedding-001",
                    google_api_key=api_key
                )
                vectorstore = load_faiss_vectorstore(embeddings)

                context = get_context_from_query(query, vectorstore, embeddings)

            with st.spinner("Generating response from Gemini..."):
                llm_model = GoogleGenerativeAI(
                    model="models/gemini-1.5-flash",
                    google_api_key=api_key
                )
                answer = generate_response(query, context, llm_model)

            st.subheader(" Answer:")
            st.write(answer)


if __name__ == "__main__":
    main()
