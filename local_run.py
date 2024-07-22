import gradio as gr
import ollama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import PDFMinerLoader, CSVLoader, JSONLoader
from langchain.text_splitter import SentenceTransformersTokenTextSplitter
import os
import pickle
import asyncio
import json

# Define the prompt template
template = """Use the following information to answer the question at the end.

{context}

{question} 
"""
prompt = PromptTemplate(template=template, input_variables=["context", "question"])

# Path to cache files
CACHE_DIR = 'cache'
os.makedirs(CACHE_DIR, exist_ok=True)

def save_cache(filename, data):
    with open(os.path.join(CACHE_DIR, filename), 'wb') as f:
        pickle.dump(data, f)

def load_cache(filename):
    try:
        with open(os.path.join(CACHE_DIR, filename), 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None

async def process_files(file_paths):
    try:
        cache_key = f'{hash(tuple(file_paths))}.pkl'
        cached_data = load_cache(cache_key)

        if cached_data:
            print("Cache hit!")
            return cached_data

        print("Processing files...")
        docs = []
        for file_path in file_paths:
            if file_path.endswith('.pdf'):
                loader = PDFMinerLoader(file_path)
                docs.extend(await asyncio.to_thread(loader.load))
            elif file_path.endswith('.csv'):
                loader = CSVLoader(file_path)
                docs.extend(loader.load())
            elif file_path.endswith('.json'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                docs.append(data)

        print("Files loaded.")
        text_splitter = SentenceTransformersTokenTextSplitter(chunk_size=1024, chunk_overlap=64)
        texts = text_splitter.split_documents(docs)
        print("Text split into chunks.")

        embeddings = HuggingFaceEmbeddings(
            model_name="thenlper/gte-large",
            model_kwargs={"device": "cpu"},  # Use CPU instead of CUDA
            encode_kwargs={"normalize_embeddings": True},
        )

        db = FAISS.from_documents(texts, embeddings)
        print("FAISS index created.")
        save_cache(cache_key, (texts, db, embeddings))
        return texts, db, embeddings
    except Exception as e:
        print(f"Error: {e}")
        return None, None, str(e)

async def query_files(files, question):
    if not files or not question.strip():
        return "Please upload valid files and enter a question."

    print("Starting query processing...")
    file_paths = [file.name for file in files]

    texts, db, embeddings = await process_files(file_paths)

    if db is None:
        print("Error during processing.")
        return f"Error processing files: {embeddings}"  # embeddings will contain the error message

    print("Processing complete.")
    results = db.similarity_search(question, k=5)
    context = " ".join([result.page_content for result in results])

    prompt_text = prompt.format(context=context, question=question)

    # Use Ollama to generate text
    stream = ollama.chat(
        model='wizardlm2',
        messages=[{'role': 'user', 'content': prompt_text}],
        stream=True,
    )
    generated_text = ""
    for chunk in stream:
        generated_text += chunk['message']['content']

    return generated_text

# Set up the Gradio interface
with gr.Blocks() as interface:
    gr.Markdown("### Retrieval Augmented Generation (RAG) for LLM Local Trial")
    gr.Markdown(
        "Upload multiple files (PDF, CSV, JSON) and ask a question. The app will generate the answer based on the content of the input files.")

    with gr.Row():
        question_input = gr.Textbox(label="Enter your question", lines=3)
        files_input = gr.File(label="Upload Files", type="filepath", file_count="multiple")  # Multiple file input

    submit_button = gr.Button("Submit")
    output_text = gr.Textbox(label="LLM Response", lines=8)
    submit_button.click(lambda files, q: asyncio.run(query_files(files, q)), inputs=[files_input, question_input],
                        outputs=output_text)

if __name__ == "__main__":
    interface.launch()
