import os
import json
import torch
import pandas as pd
from openpyxl import load_workbook
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_NAME = "openai-community/roberta-base-openai-detector"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.eval()
    model.to(device)
except Exception as e:
    print(f"Error while loading the model: {e}")
    exit()

BATCH_SIZE = 8
DOCUMENTS_DIR = "documents"
RESULTS_FILE = "ai_detection_results.xlsx"
RESULTS_JSON = "ai_detection_results.json"


def preprocess_text(text):
    return text.replace("\n", " ").strip()


def chunk_text_by_tokens(text, tokenizer, max_length=512):
    """
    Chunk text by tokens rather than by characters. This ensures that each
    chunk aligns with token boundaries and doesn't get truncated unintentionally.
    """
    tokens = tokenizer.encode(text, add_special_tokens=False)
    token_chunks = [tokens[i:i+max_length] for i in range(0, len(tokens), max_length)]
    return token_chunks


def detect_ai_text_full(text, tokenizer, model, batch_size=BATCH_SIZE):
    if not text.strip():
        return 0.0

    token_chunks = chunk_text_by_tokens(text, tokenizer, max_length=512)
    if not token_chunks:
        return 0.0

    probabilities = []
    chunk_texts = [tokenizer.decode(chunk, skip_special_tokens=True) for chunk in token_chunks]
    for i in range(0, len(chunk_texts), batch_size):
        batch = chunk_texts[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors="pt", max_length=512, padding=True, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits
        batch_probs = torch.softmax(logits, dim=-1)[:, 1].tolist()
        probabilities.extend(batch_probs)
    return float(sum(probabilities) / len(probabilities)) if probabilities else 0.0


def process_documents(directory):
    documents = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            with open(os.path.join(directory, filename), "r", encoding="utf-8") as file:
                content = preprocess_text(file.read())
                if content:
                    documents.append({"filename": filename, "content": content})
    return documents


def save_results_to_files(results, results_json, results_excel):
    with open(results_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    df = pd.DataFrame(results)
    df.to_excel(results_excel, index=False, engine="xlsxwriter")


def main():
    if not os.path.exists(DOCUMENTS_DIR):
        print(f"The folder {DOCUMENTS_DIR} doesn't exist.")
        exit()
    documents = process_documents(DOCUMENTS_DIR)
    if not documents:
        print("0 documents found.")
        exit()
    results = []
    for doc in documents:
        prob_ai = detect_ai_text_full(doc["content"], tokenizer, model)
        results.append({'Filename': doc["filename"], 'Probability AI': prob_ai})
    save_results_to_files(results, RESULTS_JSON, RESULTS_FILE)
    print(f"Analysis finished. Results in {RESULTS_FILE} and {RESULTS_JSON}.")


if __name__ == "__main__":
    main()
