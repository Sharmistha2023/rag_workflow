import ollama
from langchain_community.embeddings import OllamaEmbeddings
import base64
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama.llms import OllamaLLM
from PIL import Image
import io
import os
import PyPDF2
import click
from transformers import pipeline
import warnings, logging

# Suppress warnings/logs from transformers
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

# Initialize Whisper ASR pipeline once for efficiency
asr_pipeline = pipeline("automatic-speech-recognition", model="openai/whisper-small")

@click.command()
@click.option('--question', '-q', help='Ask question based on the dataset')
@click.option('--dataset', '-d', type=click.Path(exists=True),
              help='provide path of direcory which you want to ingest')

def main(question, dataset):
    if dataset:
        generate_embedding(dataset)
    if question:
        rag(question)


def image_to_base64(image_path):
    """Convert the image into base64"""
    with Image.open(image_path) as img:
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')


def extract_feature_from_image_file(image_path):
    """Extract the feature from the image file"""
    try:
        # Convert image to base64
        image_base64 = image_to_base64(image_path)
        MODEL = 'minicpm-v:8b'
        response = ollama.generate(
        model=MODEL,
        prompt=(
            "You are an expert in computer vision. Analyze the provided image carefully and extract detailed features. "
            "Return the results in text format with clear headings and bullet points. Include: "
            "1. Number of objects (count and type), "
            "2. Colors of objects, "
            "3. Positions of objects in the frame, "
            "4. Additional notable patterns, textures, or anomalies. "
            "Be precise and only describe what is visible without guessing."
        ),
        images=[image_base64]
        )
        return response['response']
    except Exception as e:
        print("Error:", str(e))


def extract_text_from_audio(audio_path):
    """Transcribe audio to text using Whisper."""
    try:
        result = asr_pipeline(audio_path)
        return result["text"]
    except Exception as e:
        print("Error processing audio:", str(e))
        return ""


def generate_embedding(directory_path):
    """Generate embedding for each file"""
    file_types = {}
    if os.path.exists(directory_path) and os.path.isdir(directory_path):
        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)
            if os.path.isfile(file_path):
                _, extension = os.path.splitext(filename)
                file_types[filename] = extension
                feature = ""
                ext = extension.lower().lstrip(".")
                if ext in ["jpg", "jpeg", "png", "gif", "bmp"]:
                    print(f"This is an image file, preprocessing is needed: {file_path}")
                    feature = extract_feature_from_image_file(file_path)
                elif ext == "pdf":
                    print(f"\nThis is pdf file\n")
                    pdf = PyPDF2.PdfReader(file_path)
                    for page in pdf.pages:
                        feature += page.extract_text()
                elif ext in ["wav", "mp3", "flac", "m4a"]:
                    print(f"This is an audio file, transcribing: {file_path}\n")
                    feature = extract_text_from_audio(file_path)
            if feature:
                print(f"splitting is happening for {filename}\n")
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                texts = text_splitter.split_text(feature)
                # Create a metadata for each chunk
                print(f"filename in metadata : {filename}\n")
                metadatas = [{"source": filename, "extension": extension} for i in range(len(texts))]
                # Create a Chroma vector store
                persist_directory = "./chroma_db_new"
                embeddings = OllamaEmbeddings(model="nomic-embed-text")
                vector_store = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
                vector_store.add_texts(texts=texts, metadatas=metadatas)
                print(f"storing in db for {filename} is done\n")
                vector_store.persist()
            else:
                print(f"Skipping {filename} — no extractable text found.\n")
        print(f"ingestion is completed \n")
        print(f"Number of chunks in DB: {vector_store._collection.count()}\n")

def rag(query):
    """Perform RAG Query using this function"""
    model = OllamaLLM(model="llama3.1")
    persist_directory = "./chroma_db_new"
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vector_store = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    docs = vector_store.similarity_search(query, k=3)
    context_parts = []
    for doc in docs:
        ref = f"(Source: {doc.metadata['source']})"
        context_parts.append(f"{doc.page_content}\n{ref}")
    context = "\n\n".join(context_parts)
    prompt = (
        f"Use the following context to answer the question. "
        f"Always include the file references `(Source: ...)` in your final answer.\n\n"
        f"{context}\n\n"
        f"Question: {query}\n"
        f"Answer (with sources):"
    )
    answer = model.invoke(prompt)
    print(f"answer: {answer}")


if __name__ == "__main__":
    main()
