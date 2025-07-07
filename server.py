
# Import required libraries
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
# import os

#  function main Streamlit app
def main():
    st.title(" RAG PDF Chatbot with Gemini")
    st.write("Upload a PDF, ask questions, and get answers powered by Gemini.")

    # Input for Google API Key
    api_key = st.text_input(" Enter your Google API Key:", type="password")

    # Check if API key is provided
    if not api_key:
        st.warning("Please enter your Google API key to continue.")
        return

    # File uploader for PDF files
    uploaded_pdf = st.file_uploader(" Upload PDF", type=["pdf"])

 # If a PDF is uploaded and user clicks process button
    if uploaded_pdf:
        if st.button(" Process PDF"):
            with st.spinner("Extracting text from PDF..."):
            # Extract text from the uploaded PDF
                text = extract_text_from_pdf(uploaded_pdf)


            with st.spinner("Splitting text into chunks..."):
                # Split the text into chunks
                chunks = split_text_into_chunks(text)


            with st.spinner("Generating embeddings and building FAISS index..."):
                # Initialize Gemini embeddings
                embeddings = GoogleGenerativeAIEmbeddings(
                    model="models/embedding-001",
                    google_api_key=api_key
                )

                # Build and save FAISS vector store
                build_faiss_vectorstore(chunks, embeddings)

            st.success("PDF processed and vectorstore created!")

    # Input for user query
    query = st.text_input(" Enter your question:")

     # If user enters query and clicks Get Answer
    if query:
        if st.button(" Get Answer"):
            with st.spinner("Loading FAISS and searching for context..."):

                # Load the FAISS vector store
                # Note: Ensure the vectorstore is built before this step
                embeddings = GoogleGenerativeAIEmbeddings(
                    model="models/embedding-001",
                    google_api_key=api_key
                )
                vectorstore = load_faiss_vectorstore(embeddings)

            # Get relevant context from the vectorstore based on the query
                context = get_context_from_query(query, vectorstore, embeddings)

            with st.spinner("Generating response from Gemini..."):

                # Initialize Gemini chat model
                llm_model = GoogleGenerativeAI(
                    model="models/gemini-1.5-flash",
                    google_api_key=api_key
                )
                # Generate response using the context and query
                answer = generate_response(query, context, llm_model)

            st.subheader(" Answer:")
            st.write(answer)


# Run the Streamlit app
if __name__ == "__main__":
    main()
