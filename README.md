# Financial Statement PDF Analyzer

A web application that analyzes financial statement PDFs and allows users to ask questions about the data using Natural Language Processing (NLP).

## Features
- Extract text from PDF financial statements.
- Generate and store document embeddings using Pinecone and Sentence Transformers.
- Retrieve contextually relevant document sections based on user queries.
- Provide concise answers to user queries using a Seq2Seq model.

## Technologies Used
- **Streamlit**: For building the interactive web interface.
- **PyPDF2**: For extracting text from PDF documents.
- **Sentence Transformers**: For creating text embeddings.
- **Pinecone**: For storing and querying document embeddings.
- **Transformers**: For generating answers using pre-trained models.

## How to Use
1. Upload a financial statement PDF.
2. Enter your financial query in the text box.
3. View the retrieved document sections and answers.

## Setup Instructions
1. Clone this repository:
   ```bash
   git clone https://github.com/surabhi-chandrakant/financial-statement-analyzer.git
