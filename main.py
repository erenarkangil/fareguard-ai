import os
import re
from pathlib import Path
from typing import Dict, List
from unstructured.partition.pdf import partition_pdf
from unstructured.documents.elements import Text, EmailAddress
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pytesseract
import pickle
# Set Tesseract OCR path (Windows default install)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
from openai import OpenAI
from tqdm import tqdm
from uuid import uuid4
import json
#import openai
import faiss
import numpy as np

import gradio as gr
from collections import deque
import hashlib
from transformers import pipeline
import yake

# You can use a smaller model like "t5-small" if resources are limited
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")


CLEANED_PATH = Path("cleaned_texts/cleaned.pkl")
CHUNKED_PATH = Path("chunked_output.pkl")
CACHE_PATH = Path("answer_cache.pkl")

answer_cache = {}

if CACHE_PATH.exists():
    print(f"🔁 Loading answer cache from {CACHE_PATH}")
    with open(CACHE_PATH, "rb") as f:
        answer_cache = pickle.load(f)
else:
    print("🆕 Starting with empty answer cache")


client = OpenAI(api_key="INSERT YOUR OPENAI KEY HERE")
chat_history = deque(maxlen=5)


def summarize_chat_keywords(chat_history_deque, max_turns=4, top_k=6):
    """
    Extracts keywords from the last max_turns chat messages using YAKE.
    Returns (raw_text, keyword_summary).
    """
    messages = list(chat_history_deque)[-max_turns:]

    # Combine recent meaningful messages
    chat_text = "\n".join(m["content"] for m in messages)

    print("📝 Raw Chat for Keyword Summary:")
    print(chat_text)

    try:
        # Extract keywords using YAKE
        kw_extractor = yake.KeywordExtractor(lan="en", n=2, top=top_k)
        keywords = kw_extractor.extract_keywords(chat_text)

        # Clean keyword output
        keyword_list = [kw.strip().lower() for kw, _ in keywords]
        keyword_summary = ", ".join(keyword_list)

        print("🧠 Keyword Summary:")
        print(keyword_summary)

        return chat_text, keyword_summary

    except Exception as e:
        print(f"❌ Keyword summarization failed: {e}")
        return chat_text, ""

def summarize_last_assistant_message(history, max_length=30):
    # Find last assistant reply
    for message in reversed(history):
        if message["role"] == "assistant":
            last_reply = message["content"]
            break
    else:
        return ""

    # Summarize to 1 sentence
    try:
        summary = summarizer(
            last_reply,
            max_length=max_length,
            min_length=10,
            do_sample=False
        )
        return summary[0]["summary_text"].strip()
    except Exception as e:
        print(f"❌ Summarization failed: {e}")
        return ""


def truncate_query(query, max_chars=200):
    return query.strip()[:max_chars]



def hash_query(text: str) -> str:
    return hashlib.sha256(text.strip().lower().encode()).hexdigest()

def save_pickle(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)

def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def embed_user_query(text):
    response = client.embeddings.create(
        input=[text],
        model="text-embedding-3-small"
    )
    return np.array([response.data[0].embedding], dtype="float32")




def gradio_chat_interface(query):
    print(f"Received query: {query}")

    if not query.strip():
        return "Please enter a question."


    # 1) Check cache first
    key = query.strip().lower()  # or hash it for better keys
    if key in answer_cache:
        print("!! Returning cached answer")
        return answer_cache[key]

    # Define domain keywords for strict factual queries
    domain_keywords = ["category", "fare", "stopover", "gds", "adm", "iata", "violation", "pricing", "minimum stay", "maximum stay",'penalty','airways','regulations','rules']
    is_strict_factual = any(word in query.lower() for word in domain_keywords)


    if not is_strict_factual:
        # General/greeting/chat/etc → allow model to respond freely
        # Build a brief conversation context from last few messages
        brief_context = ""
        if chat_history:  # only build context if there's chat history
            for msg in list(chat_history)[-2:]:
                role = msg["role"].capitalize()
                content = msg["content"].strip()
                brief_context += f"{role}: {content}\n"

        if brief_context:
            user_content = f"""--- BRIEF CONVERSATION CONTEXT ---
        {brief_context}--- USER QUESTION ---
        {query}
        """
        else:
            user_content = f"""--- USER QUESTION ---
        {query}
        """

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system",
                 "content": "You're a helpful and friendly assistant working on airline revenue management and agent debit memo queries. Feel free to answer general or human questions. Use analogies if needed and make the person trust you will help them to reduce penalties."},
                {"role": "user", "content": user_content}
            ]
        )

        answer = response.choices[0].message.content

        # Update only chat history, but skip cache
        chat_history.append({"role": "user", "content": query})
        chat_history.append({"role": "assistant", "content": answer})

        return answer

    # Domain query → Use FAISS + strict prompt
    try:
        embedding = embed_user_query(query)
        embedding = embedding / np.linalg.norm(embedding, axis=1, keepdims=True)

        results = search_faiss(embedding, top_k=10)
        factual_context = "\n\n".join([r["text"] for r in results])


        # STEP 2. Extract keywords (optional fallback)
        keyword_summary = summarize_chat_keywords(chat_history, max_turns=6, top_k=8)[1]
        if not keyword_summary.strip():
            keyword_summary = query  # fallback

        # STEP 2.5: Build improved brief conversation context
        last_user_question = ""
        for msg in reversed(chat_history):
            if msg["role"] == "user":
                last_user_question = msg["content"]
                break

        last_assistant_answer = summarize_last_assistant_message(chat_history)
        short_query = truncate_query(query)

        brief_context = f"""User previously asked: "{last_user_question}"
        Assistant answered: "{last_assistant_answer}"
        User now asked: "{short_query}"
        """

        # STEP 3: Build prompt content
        user_content = f"""You are an expert assistant specialized in airline fare rules helping travel agents not to get ADM penalties.

        --- FACTUAL CONTEXT FROM DOCUMENTS (highest priority) ---
        {factual_context}
        """

        if keyword_summary:
            user_content += f"""--- KEYWORDS USED IN PREVIOUS CHAT ---
        {keyword_summary}
        """

        if brief_context.strip():
            user_content += f"""--- BRIEF CONVERSATION CONTEXT ---
        {brief_context}
        """

        user_content += f"""--- CURRENT USER QUESTION ---
        {query}
        """

        system_prompt = """You are an expert assistant specialized in airline fare rules and violations (e.g., ATPCO Category 6, 7, 8), revenue management, and GDS/ADM violations.
        Only answer using the context below. Do not guess. If the answer is not clearly in the context, try to reason using any available related information, but do not guess.
        Tell the user if you are guessing.
        Use prior conversation for clarification only if it does not conflict with the retrieved context."""

        print(user_content)
        # STEP 4: OpenAI call
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages
        )
        answer = response.choices[0].message.content

        # STEP 5: Update memory & cache
        chat_history.append({"role": "user", "content": query})
        chat_history.append({"role": "assistant", "content": answer})
        answer_cache[key] = answer
        with open(CACHE_PATH, "wb") as f:
            pickle.dump(answer_cache, f)

        return answer

    except Exception as e:
        return f"❌ Error: {e}"




def launch_gradio():
    with gr.Blocks(title="✈️ Airline Fare Violation Assistant - Eren Arkangil") as demo:
        gr.Markdown("## Ask about fare rules, violations, and GDS misuse")
        chatbot = gr.Chatbot(label="FareGuard.AI Assistant")
        msg = gr.Textbox(placeholder="Ask a question...", label="Your question")

        def user_input(user_message, history):
            response = gradio_chat_interface(user_message)
            return "", history + [[user_message, response]]

        msg.submit(user_input, [msg, chatbot], [msg, chatbot])

    demo.launch(share=True, debug=True)




def search_faiss(query_embedding: np.ndarray, top_k=5):
    index = faiss.read_index("faiss.index")

    with open("faiss_metadata.pkl", "rb") as f:
        metadata = pickle.load(f)

    D, I = index.search(query_embedding, top_k)


    if len(I) == 0 or len(I[0]) == 0:
        return [{"text": "No relevant documents found.", "metadata": {}}]

    results = []
    for i in I[0]:
        results.append({
            "text": metadata["texts"][i],
            "metadata": metadata["metadatas"][i],
        })

    return results

def load_or_build_faiss_index(
    embedding_path="embedded_docs.json",
    index_path="faiss.index",
    metadata_path="faiss_metadata.pkl"
):
    if os.path.exists(index_path) and os.path.exists(metadata_path):
        print("📂 Found existing FAISS index and metadata. Loading...")

        index = faiss.read_index(index_path)
        with open(metadata_path, "rb") as f:
            metadata = pickle.load(f)
        return index, metadata

    print("🔧 No FAISS index found. Building from embeddings...")

    # Load embeddings
    with open(embedding_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    embeddings = np.array([d["embedding"] for d in data]).astype("float32")
    texts = [d["text"] for d in data]
    metadatas = [d["metadata"] for d in data]

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    faiss.write_index(index, index_path)
    with open(metadata_path, "wb") as f:
        pickle.dump({"texts": texts, "metadatas": metadatas}, f)

    print(f"✅ FAISS index saved to {index_path}")
    print(f"✅ Metadata saved to {metadata_path}")

    return index, {"texts": texts, "metadatas": metadatas}

def load_or_generate_embeddings(docs: List[Document], path="embedded_docs.json") -> List[dict]:
    if os.path.exists(path):
        print(f"📂 Found existing embeddings at {path}, loading...")
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    else:
        print("🚀 No saved embeddings found, generating...")
        embedded = embed_documents_openai(docs)
        save_embeddings_to_json(embedded, out_path=path)
        print(f"✅ Saved embeddings to {path}")
        return embedded


def save_embeddings_to_json(embeddings: List[dict], out_path: str = "embedded_docs.json"):
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(embeddings, f, ensure_ascii=False, indent=2)

def embed_documents_openai(documents: List[Document], model="text-embedding-3-small") -> List[dict]:
    embedded_docs = []

    for doc in tqdm(documents, desc="🔗 Embedding documents"):
        try:
            response = client.embeddings.create(
                input=doc.page_content,
                model=model
            )
            embedded_docs.append({
                "id": str(uuid4()),
                "embedding": response.data[0].embedding,
                "metadata": doc.metadata,
                "text": doc.page_content,
            })
        except Exception as e:
            print(f"❌ Error embedding doc: {e}")

    return embedded_docs



def is_junk_line(t: str) -> bool:
    t = t.strip().lower()
    return bool(re.match(r"^edition\s+\d+(\.\d+)?$", t)) or t in {"table of contents"}



def looks_like_toc_chunk(text: str) -> bool:
    lines = text.splitlines()
    section_count = sum(1 for l in lines if re.match(r"^Section\\s+\\d+(\\.\\d+)*\\s+", l))
    dot_leader_count = sum(1 for l in lines if re.search(r"\\.\\.+", l))
    gibberish = sum(1 for l in lines if re.search(r"[cce]{5,}", l.lower()))
    return section_count >= 2 or dot_leader_count >= 2 or gibberish >= 1


def is_datetime_line(text: str) -> bool:
    datetime_patterns = [
        r"^\d{1,2}/\d{1,2}/\d{2,4},? ?\d{1,2}:\d{2} ?[APMapm]{2}$",          # 6/3/25, 12:14 PM
        r"^\d{2}/\d{2}/\d{4} \d{2}:\d{2}$",                                  # 03/07/2024 08:45
        r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}(:\d{2})?$",                         # 2022-12-01 14:30:00
        r"^[A-Z][a-z]+day,? \d{1,2} [A-Z][a-z]+ \d{4},? \d{1,2}:\d{2} ?[APMapm]{2}$",  # Friday, 12 Jul 2024, 9:00 AM
        r"^\d{1,2} [A-Z][a-z]+ \d{2,4}$",                                    # 19 June 2025, 15 May 2011
    ]
    return any(re.match(p, text.strip()) for p in datetime_patterns)



def save_cleaned_texts(cleaned: Dict[str, str], folder: str = "cleaned_texts"):
    os.makedirs(folder, exist_ok=True)
    for filename, text in cleaned.items():
        out_path = Path(folder) / f"{filename}.txt"
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(text)
    print(f"📄 Cleaned (pre-chunk) documents saved in ./{folder}")

def save_chunked_documents(docs: List[Document], out_file: str = "chunked_output.txt"):
    with open(out_file, "w", encoding="utf-8") as f:
        for i, doc in enumerate(docs):
            f.write(f"\n\n--- Chunk {i+1} from {doc.metadata['source']} ---\n")
            f.write(doc.page_content.strip())
    print(f"📄 Chunked output saved to {out_file}")

def attach_headings_to_bullets(text_lines: List[str]) -> List[str]:
    result = []
    current_heading = ""
    for line in text_lines:
        stripped = line.strip()
        if not stripped:
            continue
        if (stripped.endswith(":") or stripped.endswith(";")) and not stripped.startswith("•"):
            # Save previous heading even if no bullets followed
            if current_heading:
                result.append(current_heading)
            current_heading = stripped
        elif stripped.startswith("•") and current_heading:
            result.append(f"{current_heading} {stripped}")
        else:
            if current_heading:
                result.append(current_heading)
                current_heading = ""  # reset so it's not reused
            result.append(stripped)

    # If last heading was not attached, still add it
    if current_heading:
        result.append(current_heading)

    return result

def is_garbage_line(line: str) -> bool:
    line = line.strip()
    if not line:
        return True
    # Don't remove if it includes technical markers
    if any(kw in line for kw in ['bytes', 'TSI', 'Geo Spec', 'value']):
        return False
    # If it's very short and has no alpha
    if len(line) < 3 and not re.search(r'[A-Za-z]', line):
        return True
    # If it's mostly non-word characters AND not technical
    if re.match(r"^[\W_\\|\.]{3,}$", line):
        return True
    return False



# Step 1: Partition all PDFs in a folder
def partition_all_pdfs(pdf_folder: Path, cache_path: Path = Path("partitioned_cache.pkl")) -> Dict[str, List[Text]]:
    # 1. Try to load from cache
    if cache_path.exists():
        print(f"🔁 Loading partitioned data from cache: {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    # 2. If not cached, run full partitioning
    partitioned = {}
    pdf_files = list(pdf_folder.glob("*.pdf"))
    for pdf_path in pdf_files:
        try:
            print(f"📄 Processing: {pdf_path.name}")
            elements = partition_pdf(filename=str(pdf_path), strategy="ocr")
            partitioned[pdf_path.name] = elements
        except Exception as e:
            print(f"❌ Failed to process {pdf_path.name}: {e}")

    # 3. Save to cache
    print(f"💾 Saving partitioned data to cache: {cache_path}")
    with open(cache_path, "wb") as f:
        pickle.dump(partitioned, f)

    return partitioned

# Step 2: Clean each partitioned document
def clean_elements(partitioned: Dict[str, List[Text]]) -> Dict[str, str]:
    cleaned = {}
    for key, elements in partitioned.items():
        # 1. Remove EmailAddress elements
        elements = [e for e in elements if not isinstance(e, EmailAddress)]

        # 2. Remove short or empty text
        elements = [e for e in elements if len(e.text.strip()) > 1]

        # 3. Rejoin hyphenated words
        texts = [e.text.replace("- ", "") for e in elements]

        # 4. Fix fake bullets like "e Duplicate Bookings"
        texts = ['• ' + t[2:].strip() if t.strip().lower().startswith("e ") else t for t in texts]

        # 5. Remove inline citations like [4, 7, 12]
        pattern = re.compile(r'\[\s*(\d+\s*,\s*)*\d+\s*\]')
        texts = [pattern.sub("", t) for t in texts]

        # 6. Remove known junk labels (optional)
        junk = {'@ thai', 'thai','AtCPEO'}
        texts = [t for t in texts if t.strip().lower() not in junk]


        # 8. Remove lines that match branding or footers
        footer_keywords = {
            "born digital", "born agile", "all rights reserved", "happiest minds",
            "the mindful company", "atcpeo", "at ) cc", "at@ce", "arxc", "you are here:",
            "arc", "axc", "click on save"
        }

        texts = [t for t in texts if not any(kw in t.lower() for kw in footer_keywords)]

        # 9. Remove lines with weird symbols or that look like formatting junk
        texts = [t for t in texts if not re.match(r"^[\W_\\|\.]{2,}$", t.strip())]

        # 10. Clean up 'mailto:' artifacts
        texts = [re.sub(r"mailto:", "", t) for t in texts]

        # Optional: Remove single non-word lines (like 'i \\By SS' or '| . ° Ye')
        texts = [t for t in texts if len(re.sub(r"\W+", "", t)) > 2]

        #some ocr garbage
        #texts = [t for t in texts if not is_garbage_line(t)]
        texts = [t for t in texts if t.strip()]

        texts = [t for t in texts if not re.match(r"^\d{1,4}$", t.strip())]

        texts = [t for t in texts if not (is_datetime_line(t) and len(t.strip().split()) <= 4)]

        texts = [t for t in texts if not is_junk_line(t)]

        # 7. Attach headings to bullets for semantic clarity
        texts = attach_headings_to_bullets(texts)

        # 8. Join into coherent text blob
        final_text = "\n\n".join(texts)
        cleaned[key] = final_text

        # 7. Join into coherent text blob
        #final_text = "\n\n".join([t.strip() for t in texts if t.strip()])
        #cleaned[key] = final_text


    return cleaned

# Step 3: Chunk each cleaned text using LangChain
def chunk_documents(cleaned: Dict[str, str], chunk_size=1200, chunk_overlap=220) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter( chunk_size=chunk_size,chunk_overlap=chunk_overlap,separators=["\n\n", "\n", ".", " ", ""])
    all_docs = []

    for filename, text in cleaned.items():
        chunks = splitter.split_text(text)

        for chunk in chunks:
            stripped_chunk = chunk.strip()

            # 🚫 Skip tiny garbage
            if len(stripped_chunk) < 30:
                continue

            # 🚫 Skip TOC-style blobs
            if looks_like_toc_chunk(stripped_chunk):
                continue

            doc = Document(
                page_content=stripped_chunk,
                metadata={"source": filename}
            )
            all_docs.append(doc)

    return all_docs


# Main pipeline
def process_airline_pdfs():

    pdf_folder = Path("C:/Users/Eren Arkangil/PyCharmMiscProject/traindata")

    # Step 1: Load cleaned data if available
    if CLEANED_PATH.exists():
        print(f"🔁 Loading cleaned data from: {CLEANED_PATH}")
        cleaned = load_pickle(CLEANED_PATH)
    else:
        print("🧼 Cleaning required — running partitioning and cleaning.")
        partitioned = partition_all_pdfs(pdf_folder)
        cleaned = clean_elements(partitioned)
        CLEANED_PATH.parent.mkdir(parents=True, exist_ok=True)
        save_pickle(cleaned, CLEANED_PATH)
        save_cleaned_texts(cleaned, folder="cleaned_texts")  # (Optional) for inspection

    # Step 2: Load chunks if available
    if CHUNKED_PATH.exists():
        print(f"🔁 Loading chunked documents from: {CHUNKED_PATH}")
        docs = load_pickle(CHUNKED_PATH)
    else:
        print("🔪 Chunking cleaned text...")
        docs = chunk_documents(cleaned)
        save_pickle(docs, CHUNKED_PATH)
        save_chunked_documents(docs, out_file="chunked_output.txt")  # (Optional) for inspection

    print(f"✅ Total documents created: {len(docs)}")

    return docs

def faiss_process(docs: List[Document]):
    # Generate or load embeddings
    embedded = load_or_generate_embeddings(docs)

    # Sanity check
    print(f"📦 Total chunks: {len(docs)}")
    print(f"🧠 Total embeddings: {len(embedded)}")

    # Build FAISS (if not already)
    index, metadata = load_or_build_faiss_index()
    print(f"📊 FAISS index contains {index.ntotal} vectors.")

    return index, metadata

# Entry point
if __name__ == "__main__":
    print('h')
    docs = process_airline_pdfs()
    index, metadata = faiss_process(docs)
    launch_gradio()