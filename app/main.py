import json

import numpy as np
import pytesseract
import requests

from sklearn.metrics.pairwise import cosine_similarity
from fastapi import FastAPI
from pdf2image import convert_from_path
from transformers import AutoTokenizer, AutoModel

app = FastAPI()


def convert_pdf():
    pages = convert_from_path("pdf.pdf")
    pdf_text = {}
    for i, page in enumerate(pages):
        if i not in [0, 1]:
            text = pytesseract.image_to_string(page, lang="por")
            pdf_text[i + 1] = text

    with open("output.json", "w", encoding="utf-8") as f:
        json.dump(pdf_text, f, ensure_ascii=False, indent=4)


def generate_embedding(text):
    print("carregando bert", flush=True)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModel.from_pretrained("bert-base-uncased")
    print("carregando bert end", flush=True)
    entradas = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
    saida = model(**entradas)

    embeddings = saida.last_hidden_state.mean(dim=1)

    embedding = embeddings.detach().numpy().tolist()
    return embedding

@app.get("/pdf_to_text", tags=["pdf_to_text"])
async def pdf_to_text():
    """ Converte o PDF em texto """
    convert_pdf()
    return {"message": "Sucesso! Texto extraído do PDF."}

@app.get("/gerar-embedding")
async def gerar_embedding():
    with open("output.json", "r", encoding="utf-8") as f:
        pdf_data = json.load(f)
    embeddings = {}

    for page, text in pdf_data.items():
        embedding = generate_embedding(text)
        embeddings[page] = embedding

    with open("embeddings.json", "w", encoding="utf-8") as f:
        json.dump(embeddings, f, ensure_ascii=False, indent=4)
    return 0


def comparar_embedding(embedding1, embedding2):
    embedding1 = np.array(embedding1).reshape(1, -1)
    embedding2 = np.array(embedding2).reshape(1, -1)

    similarity = cosine_similarity(embedding1, embedding2)[0][0]

    distancia = np.linalg.norm(embedding1 - embedding2)

    return {"similarity": similarity, "distance": distancia}

def find_best_match(embedding_consulta, embedding_pagina):
    best = []

    for page, embedding in embedding_pagina.items():
        resultado = comparar_embedding(embedding_consulta, embedding)
        best.append((page, resultado["similarity"]))

    best.sort(key=lambda x: x[1], reverse=True)
    return best
@app.post("/send-prompt", tags=["send_prompt"])
async def send_prompt(prompt: str):
    """ Recebe um prompt e gera uma resposta baseada no texto do PDF """

    with open("output.json", "r", encoding="utf-8") as f:
        pdf_data = json.load(f)
    with open("embeddings.json", "r", encoding="utf-8") as f:
        pdf_embeddings = json.load(f)

    user_embedding = generate_embedding(prompt)

    result = find_best_match(user_embedding, pdf_embeddings)
    best_3 = result[:3]
    texto = pdf_data[best_3[0][0]] + pdf_data[best_3[1][0]] + pdf_data[best_3[2][0]]
    # prompt_padrao = f"""
    # Você deve responder minha pergunta baseado nos dados que irei enviar. Responda utilizando somente os dados que estão
    # no texto enviado.
    #
    # TEXTO:
    #
    # {texto}
    #
    # PERGUNTA:
    #
    # {prompt}
    # """
    # response = requests.post(
    #     "http://ollama:11434/api/generate",
    #     json={
    #         "model": "llama3.2:latest",
    #         "prompt": prompt_padrao,
    #         "stream": False
    #     },
    # )
    # response_json = response.json()
    # print(response_json)
    # return response_json["response"]

