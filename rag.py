from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from cities import cities
import gradio as gr
import PyPDF2  
import docx2txt  

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

city_embeddings = np.array([embedding_model.encode(cities[city]) for city in cities])

index = faiss.IndexFlatL2(city_embeddings.shape[1])  
index.add(city_embeddings)

city_names = list(cities.keys())

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

def gradio_interface(question, uploaded_files):
    extracted_text = ""
    
    if uploaded_files:
        uploaded_file = uploaded_files[0] 
        extracted_text = extract_text_from_file(uploaded_file)
    
    return generate_answer(question, extracted_text)  

def generate_answer(query, extracted_text=""):
    query_embedding = embedding_model.encode(query)
    
    if extracted_text:
        file_embedding = embedding_model.encode(extracted_text)
        retrieved_passage = extracted_text
    else:
        k = 1
        D, I = index.search(np.array([query_embedding]), k)
        retrieved_city = city_names[I[0][0]]
        retrieved_passage = cities[retrieved_city]
        print(f"Retrieved city: {retrieved_city}")
    
    # Bloom
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

# For testing
# if __name__ == "__main__":
#     query = "How many people live in Tokyo?"
#     answer = generate_answer(query)
#     print(f"The answer is here: {answer}")
