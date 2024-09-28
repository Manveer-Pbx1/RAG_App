# Import required libraries
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from cities import cities
import gradio as gr
import PyPDF2  # To handle PDF files
import docx2txt  # To handle DOCX files

# Load embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Generate city embeddings
city_embeddings = np.array([embedding_model.encode(cities[city]) for city in cities])

# Build the FAISS index for cities
index = faiss.IndexFlatL2(city_embeddings.shape[1])  
index.add(city_embeddings)

# Save city names for retrieval
city_names = list(cities.keys())

# Function to extract text from uploaded PDF/DOCX files
def extract_text_from_file(file_path):
    if file_path.endswith(".pdf"):
        with open(file_path, "rb") as f:
            pdf_reader = PyPDF2.PdfReader(f)
            text = ""
            for page_num in range(len(pdf_reader.pages)):
                text += pdf_reader.pages[page_num].extract_text()
        return text
    elif file_path.endswith(".docx"):
        return docx2txt.process(file_path)
    else:
        return ""

# Function to handle both city data and user-uploaded files
def gradio_interface(question, uploaded_files):
    extracted_text = ""
    
    # Check if there are uploaded files
    if uploaded_files:
        uploaded_file = uploaded_files[0] 
        extracted_text = extract_text_from_file(uploaded_file)
    
    # Generate answer based on the extracted text from uploaded file or city data
    return generate_answer(question, extracted_text)  

# Function to generate answers based on both dataset and uploaded content
def generate_answer(query, extracted_text=""):
    query_embedding = embedding_model.encode(query)
    
    # RAG system: check if there is extracted text, otherwise use the city dataset
    if extracted_text:
        # Create an embedding for the uploaded content and search within it
        file_embedding = embedding_model.encode(extracted_text)
        # For simplicity, assume single large passage in the file
        retrieved_passage = extracted_text
    else:
        # Search within city dataset if no uploaded file
        k = 1
        D, I = index.search(np.array([query_embedding]), k)
        retrieved_city = city_names[I[0][0]]
        retrieved_passage = cities[retrieved_city]
        print(f"Retrieved city: {retrieved_city}")
    
    # Use Bloom model to generate the answer
    tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-560m")
    model = AutoModelForCausalLM.from_pretrained("bigscience/bloom-560m")
    
    input_text = (
        f"Passage: {retrieved_passage}\n\n"
        f"Question: {query}\n\n"
        "Answer:"
    )
    
    inputs = tokenizer(input_text, return_tensors="pt")
    
    output = model.generate(
        **inputs, 
        max_new_tokens=250,  
        do_sample=True,      
        top_p=0.48,           
        num_return_sequences=1,  
    )
    
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    answer = response.split("Answer:")[-1].strip()
    return answer

interface = gr.Interface(
    fn=gradio_interface,
    inputs=[
        gr.Textbox(label="Question"),  
        gr.File(label="Upload Documents (PDF/DOCX)", type="filepath", file_count="multiple")  
    ],
    outputs=gr.Textbox(label="Answer"),
    title="Wiki & File QA System",
    description="Cities include: Paris, New York City, Tokyo, London, and Rome. All the data was collected from wikipedia."
)

interface.launch()

# Test the system
if __name__ == "__main__":
    query = "How many people live in Tokyo?"
    answer = generate_answer(query)
    print(f"The answer is here: {answer}")
