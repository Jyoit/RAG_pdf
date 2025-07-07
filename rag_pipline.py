from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_google_genai import GoogleGenerativeAIEmbeddings, GoogleGenerativeAI
from langchain_community.vectorstores import FAISS

#  Read PDF and extract text
def extract_text_from_pdf(pdf_file) -> str:
    pdf_reader = PdfReader(pdf_file)

    # Extract text from each page
    full_text = ""
    for page in pdf_reader.pages:
        full_text += page.extract_text() + "\n"
    return full_text


#splitting the text into chunks
def split_text_into_chunks(full_text: str) -> list:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, # Size of each chunk
        chunk_overlap=200, # Overlap between chunks
    )

    # Split the text into chunks
    chunks = text_splitter.split_text(full_text)
    return chunks


#  Build and store in FAISS vectorstore
def build_faiss_vectorstore(chunks: list, embeddings, index_path: str = "faiss_index"):
    #create FAISS index from chunks and embeddings
    vectorstore = FAISS.from_texts(chunks, embeddings)
    # Save the vectorstore to a disk file
    vectorstore.save_local(index_path)
    return vectorstore


#  Load FAISS vectorstore from the disk file
def load_faiss_vectorstore(embeddings, index_path: str = "faiss_index"):
    vectorstore = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    return vectorstore


# Search for relevant context
def get_context_from_query(query: str, vectorstore, embeddings, k=3) -> str:
    # search the vectorstore
    query_embedding = embeddings.embed_query(query)
    context_docs = vectorstore.similarity_search_by_vector(query_embedding, k=k)
    
    # Combine all the context documents
    context_text = "\n\n".join([doc.page_content for doc in context_docs])
    return context_text


#  Generate response from Gemini
def generate_response(query: str, context_text: str, model, system_prompt="You are a helpful assistant.") -> str:
    # Building full prompt
    prompt = f"""
{system_prompt} Answer the question based only on the context below.
If the answer is not found in the context, say "I don't know."

Context:
{context_text}

Question:
{query}
"""
    response = model.invoke(prompt)
    return response.strip()
