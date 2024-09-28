# City RAG System with Bloom and File Upload Feature

This project implements a Retrieval-Augmented Generation (RAG) system that answers questions based on a dataset of famous cities, enhanced with the ability to upload PDF and DOCX files for additional context. It uses the BLOOM model for language generation and FAISS for efficient document retrieval.

## Table of Contents
1. [Features](#features)
2. [Setup Instructions](#setup-instructions)
3. [How to Run](#how-to-run)
4. [File Structure](#file-structure)
5. [Additional Information](#additional-information)

## Features

- **Cities-based Question Answering:** The system answers questions based on stored information about cities.
- **File Upload:** Users can upload PDF or DOCX files for the system to answer questions based on those files.
- **Integration with BLOOM:** Uses BLOOM for generating language responses.

## Setup Instructions

### 1. Clone the repository

First, clone the project repository to your local machine:
```bash
git clone <repository-url>
cd <project-directory>
```
### 2. Install Dependencies
```bash 
pip install -r requirements.txt
```

### 3. Download Required Models

- SentenceTransformer: `all-MiniLM-L6-v2`
- BLOOM model: `bigscience/bloom-560m`

## How to Run

- Start the system with:

```bash
python rag.py
```

- Open your browser to the Gradio interface URL provided in the terminal output (usually `http://localhost:7860/` by default).

- In the interface:

    Enter a question in the textbox.
Optionally, upload a PDF or DOCX file for additional context.
The system will retrieve the relevant city information and generate an answer using BLOOM.

## File Structure

- cities.py: Contains the dataset of cities.
- rag.py: The main program that integrates retrieval, question answering, file uploads, and BLOOM-based language generation.

- requirements.txt: Lists all the Python dependencies required for this project.

## Additional Information

- This project utilizes FAISS for efficient search and Gradio for an interactive interface.

- It integrates with the BLOOM model for generating natural language answers.

- The file upload feature allows users to extend the context of questions with additional documents (PDF/DOCX).

