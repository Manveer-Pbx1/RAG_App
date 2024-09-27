# Required Libraries
import requests
from bs4 import BeautifulSoup
import numpy as np
from datasets import Dataset
import torch
from transformers import RagRetriever, RagTokenForGeneration, RagTokenizer, DPRQuestionEncoderTokenizer, DPRQuestionEncoder, BloomForCausalLM, BloomTokenizerFast
import os
import gradio as gr  # Gradio library
from PyPDF2 import PdfReader  # Library for PDF handling
from docx import Document  # Library for DOCX handling

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# List of cities to scrape
cities = ["Jabalpur", "Berlin", "Mumbai"]

# User-Agent string (simulate a real browser)
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

# Scrape Wikipedia articles
def scrape_wikipedia(city):
    url = f"https://en.wikipedia.org/wiki/{city}"
    response = requests.get(url, headers=headers)
    
    print(f"Scraping {city}: Status Code - {response.status_code}")
    
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Targeting the 'mw-content-text' div
        content_div = soup.find(id='mw-content-text')
        if content_div:
            # Extracting all paragraphs within the div
            paragraphs = content_div.find_all('p')
            full_text = " ".join([paragraph.get_text(strip=True) for paragraph in paragraphs if paragraph.get_text(strip=True)])
            print(len(full_text))
            return full_text if full_text else f"No content found for {city}"
        
        return f"No content found for {city}"
    else:
        return f"Failed to retrieve data for {city}"

# Chunk long text into smaller pieces
def chunk_text(text, chunk_size=500):
    words = text.split()
    return [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

# Create an empty dictionary to store articles
articles = {}

# Scrape and store articles for each city
for city in cities:
    article_text = scrape_wikipedia(city)
    # Split large texts into smaller chunks
    articles[city] = chunk_text(article_text)

# Initialize tokenizer and encoder from RAG
dpr_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
dpr_encoder = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base")

# Create lists for titles, texts, and embeddings
titles = []
texts = []
embeddings_list = []

# Generate embeddings for the article chunks using RAG's question encoder
for city, chunks in articles.items():
    for chunk in chunks:
        titles.append(city)  # Assign the city name as the title for each chunk
        texts.append(chunk)
        inputs = dpr_tokenizer(chunk, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            embeddings = dpr_encoder(**inputs).pooler_output
        embeddings_list.append(embeddings.squeeze().cpu().numpy())

# Create the dataset with title, text, and embeddings
wiki_dataset = Dataset.from_dict({
    "title": titles,  # Add the city names as the title column
    "text": texts,
    "embeddings": [embedding.tolist() for embedding in embeddings_list]
})

# Add FAISS index for efficient retrieval
wiki_dataset.add_faiss_index(column="embeddings")

# Initialize tokenizer and retriever for RAG
rag_tokenizer = RagTokenizer.from_pretrained("facebook/rag-token-base")
retriever = RagRetriever.from_pretrained("facebook/rag-token-base", indexed_dataset=wiki_dataset)

# Initialize the RAG model
rag_model = RagTokenForGeneration.from_pretrained("facebook/rag-token-base", retriever=retriever)

# Initialize Bloom tokenizer and model
bloom_tokenizer = BloomTokenizerFast.from_pretrained("bigscience/bloom-560m")
bloom_model = BloomForCausalLM.from_pretrained("bigscience/bloom-560m")

# Set the pad token to eos_token
bloom_tokenizer.pad_token = bloom_tokenizer.eos_token  # Set pad token to eos_token

# Function to extract text from uploaded PDF/DOCX files
def extract_text_from_file(uploaded_file):
    # Handle string input (file path) or file object
    if isinstance(uploaded_file, str):
        # String case (file path)
        file_path = uploaded_file
    else:
        # File object case
        file_path = uploaded_file.name  # Get the name of the uploaded file

    if file_path.endswith('.pdf'):
        # Extract text from PDF
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text.strip()  # Return extracted text
    elif file_path.endswith('.docx'):
        # Extract text from DOCX
        doc = Document(file_path)
        text = "\n".join([para.text for para in doc.paragraphs])
        return text.strip()  # Return extracted text
    return ""

# Define the function to query Bloom
def query_bloom(question, context):
    prompt = f"{context}\n\nAnswer: {question}"
    
    inputs = bloom_tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = bloom_model.generate(
            inputs.input_ids, 
            attention_mask=inputs.attention_mask,
            max_new_tokens=300,
            max_length=600,
            num_beams=2,
            do_sample=True,
            top_k=50,
            top_p=0.95
        )

    answer = bloom_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer

# Generate an answer using both RAG and Bloom
def generate_answer(question, extracted_text=""):
    if not question:
        return "Please provide a valid question."
    
    # Tokenize and encode the input question
    input_dict = rag_tokenizer(question, return_tensors="pt")
    input_ids = input_dict.input_ids.to(rag_model.device)
    attention_mask = input_dict.attention_mask.to(rag_model.device)

    try:
        # Use RAG's generate function to automatically handle retrieval and generation
        generated_ids = rag_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            num_beams=2,
            n_docs=5,
            max_length=250,
            num_return_sequences=1
        )
        
        # Decode the retrieved context (relevant chunk)
        context = rag_tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        
        # If there's extracted text from the uploaded document, use it for answering
        if extracted_text:
            context += "\n" + extracted_text  # Combine with extracted text

        # Query Bloom with the retrieved context
        answer = query_bloom(question, context)
        
        # Ensure that Bloom only returns one answer
        return answer.strip()

    except Exception as e:
        return f"An error occurred while generating the answer: {str(e)}"

# Gradio interface
def gradio_interface(question, uploaded_files):
    extracted_text = ""
    if uploaded_files:  # Check if any file is uploaded
        uploaded_file = uploaded_files[0]  # Get the first file from the list
        extracted_text = extract_text_from_file(uploaded_file)  # Extract text from uploaded file
    return generate_answer(question, extracted_text)  # Generate the answer

# Create Gradio interface with updated syntax
interface = gr.Interface(
    fn=gradio_interface,
    inputs=[
        gr.Textbox(label="Question"),  # Text input for the question
        gr.File(label="Upload Documents (PDF/DOCX)", type="filepath", file_count="multiple")  # Corrected input type
    ],
    outputs=gr.Textbox(label="Answer"),
    title="Wiki QA System",
    description="Ask questions about cities - Tokyo, New York City, Berlin, Toronto, Delhi, Mumbai, Lahore or upload a file. The model will retrieve information from Wikipedia and generate answers."
)

# Launch the interface
interface.launch()
