import streamlit as st
import torch
import logging
import PyPDF2
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import os
from dotenv import load_dotenv

load_dotenv()  # Load environment variables

def extract_text_from_pdf(pdf_path):
    """Extract text from PDF file."""
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        full_text = ""
        for page in reader.pages:
            full_text += page.extract_text() + "\n"
    return full_text

@st.cache_resource
def initialize_services():
    try:
        pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
        
        INDEX_NAME = "financial-statement-data"
        EMBEDDING_DIMENSION = 768

        if INDEX_NAME not in [index.name for index in pc.list_indexes()]:
            pc.create_index(
                name=INDEX_NAME,
                dimension=EMBEDDING_DIMENSION,
                metric="cosine",
                spec=ServerlessSpec(cloud='aws', region='us-east-1')
            )

        index = pc.Index(INDEX_NAME)
        model = SentenceTransformer('all-mpnet-base-v2')
        tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
        seq2seq_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

        return index, model, tokenizer, seq2seq_model
    
    except Exception as e:
        st.error(f"Initialization error: {e}")
        return None, None, None, None

def prepare_embeddings(documents, index, model):
    """Generate and store embeddings in Pinecone."""
    embeddings = model.encode(documents)
    for i, embedding in enumerate(embeddings):
        index.upsert([(str(i), embedding.tolist())])

def retrieve_documents(query, index, model, top_k=2):
    """Retrieve relevant documents from Pinecone."""
    query_embedding = model.encode(query)
    results = index.query(vector=query_embedding.tolist(), top_k=top_k)
    return [result.id for result in results.matches]

def generate_answer(query, context_docs, tokenizer, seq2seq_model):
    """Generate answer using seq2seq model."""
    full_input = f"Context: {' '.join(context_docs)} Question: {query}"
    inputs = tokenizer(full_input, return_tensors="pt", max_length=512, truncation=True)
    
    with torch.no_grad():
        outputs = seq2seq_model.generate(
            input_ids=inputs.input_ids, 
            max_length=100,
            num_return_sequences=1
        )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def main():
    st.title("Financial Statement PDF Analyzer")

    # PDF Upload
    uploaded_file = st.file_uploader("Upload Financial Statement PDF", type=['pdf'])

    # Initialize services
    index, model, tokenizer, seq2seq_model = initialize_services()

    if not all([index, model, tokenizer, seq2seq_model]):
        st.warning("Failed to initialize services.")
        return

    if uploaded_file is not None:
        # Save uploaded PDF temporarily
        with open("temp_financial_statement.pdf", "wb") as f:
            f.write(uploaded_file.getvalue())

        # Extract text from PDF
        document_text = extract_text_from_pdf("temp_financial_statement.pdf")
        
        # Split text into chunks for embedding
        text_chunks = document_text.split('\n')
        text_chunks = [chunk for chunk in text_chunks if chunk.strip()]

        # Prepare embeddings
        prepare_embeddings(text_chunks, index, model)

        # Query Interface
        st.subheader("Ask Questions about the Financial Statement")
        query = st.text_input("Enter your financial query:")

        if query:
            # Retrieve relevant document chunks
            retrieved_doc_ids = retrieve_documents(query, index, model)
            context_docs = [text_chunks[int(doc_id)] for doc_id in retrieved_doc_ids]

            # Generate answer
            answer = generate_answer(query, context_docs, tokenizer, seq2seq_model)

            # Display results
            st.subheader("Retrieved Document Chunks:")
            for doc in context_docs:
                st.text(doc)
            
            st.subheader("Answer:")
            st.write(answer)

if __name__ == "__main__":
    main()