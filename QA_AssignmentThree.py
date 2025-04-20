import wikipedia
from transformers import pipeline
import streamlit as st
import re

# Load Wikipedia content
def load_wikipedia_content(page_title):
    page = wikipedia.page(page_title)
    text = page.content
    return text

# Clean text
def clean_text(text):
    text = re.sub(r'==.*?==+', '', text)  # Remove section titles
    text = text.replace('\n', '. ')       # Replace newlines with periods
    return text.strip()

# Chunk text for QA
def chunk_text(text, max_len=400):
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_len // 2):
        chunk = " ".join(words[i:i + max_len])
        chunks.append(chunk)
    return chunks

# Search best answer in chunks
def get_best_answer(question, chunks):
    best_answer = ""
    best_score = 0

    for chunk in chunks:
        result = qa_pipeline(question=question, context=chunk)
        if result["score"] > best_score:
            best_score = result["score"]
            best_answer = result["answer"]

    return best_answer, best_score

# Load model
model_name = "deepset/roberta-base-squad2"
qa_pipeline = pipeline("question-answering", model=model_name, tokenizer=model_name)

# UI starts here
st.title("Do you have a question in Mars Exploration?")

# Load and preprocess content
raw_text = load_wikipedia_content("Exploration of Mars")
cleaned_context = clean_text(raw_text)
chunks = chunk_text(cleaned_context)

# Ask a question
question = st.text_input("Ask a question about the exploration of Mars:")

# Answer it
if question:
    with st.spinner("Searching for answer..."):
        answer, score = get_best_answer(question, chunks)
        st.success(f"Answer: {answer} (Confidence: {score:.2f})")
