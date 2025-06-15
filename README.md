<<<<<<< HEAD
# PDF-ChatBot

## Overview

PDF-ChatBot is a powerful web application built using Streamlit that allows users to interact with multiple PDF files using natural language queries. Leveraging the capabilities of Google Generative AI and FAISS for vector-based search, this bot provides accurate responses from the content of uploaded PDFs.

The chatbot uses state-of-the-art machine learning models like Gemini and FAISS to process and retrieve relevant sections from large PDFs, making it an efficient tool for document analysis, research, and information extraction.

## Features

- **Multiple PDF Support**: Upload and process multiple PDFs simultaneously.
- **Natural Language Interaction**: Ask questions in natural language and receive detailed answers from the PDF content.
- **Vector Search**: FAISS-based search to find relevant answers efficiently.
- **Google Generative AI Integration**: Uses the Gemini model to generate responses based on the extracted content.
- **Fast Processing**: Capable of handling large documents and returning results quickly.

## How It Works

1. **Upload PDFs**: Upload one or more PDF files through the interface.
2. **PDF Parsing**: The PDFs are parsed, and text is extracted using `PyPDF2`.
3. **Text Chunking**: The extracted text is split into manageable chunks using Langchain's `RecursiveCharacterTextSplitter`.
4. **Vectorization**: Text chunks are converted into vector representations using Google's Generative AI Embeddings.
5. **Vector Store Creation**: A FAISS index is created from these vectors for fast similarity search.
6. **Ask Questions**: Users can input questions through the interface, and the chatbot will search the PDFs and respond with relevant information.
7. **Response Generation**: Based on the search results, the chatbot generates detailed answers using the Gemini model.

## Tech Stack

- **Streamlit**: Frontend interface for the PDF-ChatBot.
- **Langchain**: Used for chaining models and creating vector-based document retrieval.
- **Google Generative AI**: Handles embeddings and conversational model responses.
- **FAISS**: Efficient similarity search across large text datasets.
- **PyPDF2**: For PDF parsing and text extraction.

## Getting Started

### Prerequisites

- Python 3.8+
- Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

- Set up a `.env` file with your Google API key:

    ```makefile
    GOOGLE_API_KEY=your_api_key_here
    ```

### Running the App

1. Clone the repository:

    ```bash
    git clone https://github.com/NitinYadav1511/PDF-ChatBot.git
    ```

2. Navigate to the project directory:

    ```bash
    cd PDF-ChatBot
    ```

3. Install the required Python libraries:

    ```bash
    pip install -r requirements.txt
    ```

4. Run the Streamlit application:

    ```bash
    streamlit run main.py
    ```

5. Upload PDFs through the sidebar and start asking questions!

## Usage

- **Upload PDFs**: Click on "Upload your PDF Files" in the sidebar to upload multiple PDF documents.
- **Ask Questions**: Enter your question in the text input field, and the bot will search the PDFs and respond with relevant answers.

## Future Enhancements

- Add support for summarizing entire PDFs.
- Improve the conversational flow and context retention across multiple questions.
- Implement a feedback mechanism to refine answer quality.

## Contributor

This repository is maintained by [Saptarshi Dey](https://github.com/Saptarshi2120).

## Contributing

Contributions are welcome! Feel free to open an issue or submit a pull request.

## License

This project is licensed under the MIT License.
=======
# RAG
>>>>>>> 84547b487471857f5ed40560bc3757ce0da44775
