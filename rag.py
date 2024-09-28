# Import required libraries
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from cities import cities
import gradio as gr


embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

city_embeddings = np.array([embedding_model.encode(cities[city]) for city in cities])

index = faiss.IndexFlatL2(city_embeddings.shape[1])  
index.add(city_embeddings)

city_names = list(cities.keys())

def gradio_interface(question, uploaded_files):
    extracted_text = ""
    if uploaded_files:  
        uploaded_file = uploaded_files[0] 
        # extracted_text = extract_text_from_file(uploaded_file)  # Extract text from uploaded file
    return generate_answer(question)  

def generate_answer(query):
    query_embedding = embedding_model.encode(query)
    k = 1
    D, I = index.search(np.array([query_embedding]), k)
    
    retrieved_city = city_names[I[0][0]]
    retrieved_passage = cities[retrieved_city]
    print(f"Retrieved city: {retrieved_city}")
    
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
    max_new_tokens=250,  # Increased the number of new tokens generated
    do_sample=True,      # Enables sampling for more creative responses
    top_p=0.48,           # Limits choices to the top 90% of probability distribution (nucleus sampling)
    num_return_sequences=1,  # Ensures a single response is generated
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
    title="Wiki QA System",
    description="Cities- Paris, London, Rome, Tokyo, and New York."
)

interface.launch()

# Test the system
if __name__ == "__main__":
    query = "How many people live in Tokyo?"
    
    answer = generate_answer(query)
    print(f"The answer is here: {answer}")
