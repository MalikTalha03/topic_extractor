import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
import requests
import spacy
from pdfminer.high_level import extract_text as pdfminer_extract_text
from docx import Document
from pdf2image import convert_from_path
import pytesseract
from transformers import T5Tokenizer, T5ForConditionalGeneration
import os
import openai
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

load_dotenv()
API_KEY_SEARCH = os.getenv("API_KEY_SEARCH")
BING_API_KEY = os.getenv("BING_API_KEY")

SEACRH_ID = os.getenv("SEACRH_ID")

app = FastAPI()

def download_pdf(url, save_path):
    try:
        # Send a GET request to the URL
        response = requests.get(url)
        response.raise_for_status()  # Raise an HTTPError for bad responses

        # Save the PDF file locally
        with open(save_path, 'wb') as file:
            file.write(response.content)
        print(f"PDF downloaded successfully: {save_path}")
    except Exception as e:
        print(f"Error downloading PDF: {e}")
        return None
    return save_path


def extract_text_from_pdf(pdf_path):
    try:
        text = pdfminer_extract_text(pdf_path)
        if not text.strip():
            text = extract_text_from_scanned_pdf(pdf_path)
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        text = ""
    return text

def extract_text_from_scanned_pdf(pdf_path, dpi=300):
    text = ""
    try:
        pages = convert_from_path(pdf_path, dpi=dpi)
        for page_num, page in enumerate(pages, start=1):
            image_path = f"page_{page_num}.png"
            page.save(image_path, 'PNG')
            page_text = pytesseract.image_to_string(page)
            text += page_text + " "
            os.remove(image_path)
    except Exception as e:
        print(f"Error performing OCR on PDF: {e}")
    return text

def extract_text_from_word(docx_path):
    text = ""
    try:
        doc = Document(docx_path)
        for para in doc.paragraphs:
            text += para.text + " "
    except Exception as e:
        print(f"Error extracting text from DOCX: {e}")
    return text

def extract_text(file_path):
    if file_path.lower().endswith('.pdf'):
        text = extract_text_from_pdf(file_path)
    elif file_path.lower().endswith('.docx'):
        text = extract_text_from_word(file_path)
    else:
        raise ValueError("Unsupported file format. Please upload PDF or DOCX files.")
    return text



# BERT Embedding Functions

tokenizer_BERT = BertTokenizer.from_pretrained('bert-base-uncased')
model_bert = BertModel.from_pretrained('bert-base-uncased')

def get_bert_embedding(text):
    inputs = tokenizer_BERT(text, return_tensors='pt', truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model_bert(**inputs)
    embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    return embedding

# Google Search API Function

def fetch_google_search_results(query, num_results=10):
    params = {"key":API_KEY_SEARCH,"cx" :SEACRH_ID,"q": query, "num": num_results}
    url = "https://www.googleapis.com/customsearch/v1"
    response = requests.get(url,params=params)
    response.raise_for_status()
    print(response.json())
    data = response.json()
    return data.get('items',[])


# Bing Search API Function

def fetch_bing_search_results(query, num_results=10):
    headers = {
        "Ocp-Apim-Subscription-Key": BING_API_KEY
    }
    params = {
        "q": query,
        "count": num_results
    }
    url = "https://api.bing.microsoft.com/v7.0/search"
    
    response = requests.get(url, headers=headers, params=params)
    response.raise_for_status()
    
    data = response.json()
    return data.get('webPages', {}).get('value',[])

# Cosine Similarity Ranking Function

def cosine_similarity_ranking(assignment_embedding, search_results):
    ranked_results = []
    for result in search_results:
        snippet = result.get('snippet', '')
        if not snippet:
            continue
        snippet_embedding = get_bert_embedding(snippet)
        similarity = float(cosine_similarity([assignment_embedding], [snippet_embedding])[0][0])  # Convert to float
        ranked_results.append({
            'title': result.get('name', ''),
            'url': result.get('url', ''),
            'snippet': snippet,
            'similarity': similarity  # Ensure it's a float
        })
    ranked_results = sorted(ranked_results, key=lambda x: x['similarity'], reverse=True)
    return ranked_results


# Processing Function

def process_assignment(file_path):
    # Step 1: Extract Text
    text = extract_text(file_path)
    if not text.strip():
        print("No text extracted from the document.")
        return []
    results = get_results(text)
    return results


    # Step 2: Generate Query
# # Set your API key
def get_results(text):
    openai.api_key = os.environ.get("OPENAI_API_KEY")


    # Define your prompt
    prompt = f"""
    This text is from a document given as an assignment to a student. I want to you to read the text and understand the main task asked from the student and then suggest a web search query to fetch some web pages link that can help the students to learn about the concepts asked in the assignment.:
    {text}
    """

    # Call the GPT-4 API
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that generates search queries."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,  # Controls creativity
        max_tokens=50,    # Adjust the length of the output
    )

    # Extract the generated query
    query = response['choices'][0]['message']['content'].strip()
    # remove "" and also word Search Query if present
    query = query.replace('"','').replace('Search Query','').strip()
    print("Generated Query:", query)
#     print("Generated Query:", query)
    # query = "How to determine project scope and cost estimation for a software project"

    # query = "Comparison of software re-engineering tools, their development history, main features, practical applications, benefits and drawbacks, integration with IDEs and version control systems"

    
    # # Step 2: Extract Keywords
    # exclude_words = ['question','assignment','CLO','the assignment','marks','Bloom Taxonomy Level']

    # keywords = extract_keywords_spacy(text,top_n=10,exclude_words=exclude_words)
    # query = " ".join(keywords)
    # print("Extracted Keywords:", keywords)
    
    # print("Search Query:", query)
    
    # Step 3: Get BERT Embedding for Assignment
    assignment_embedding = get_bert_embedding(text)
    
    #  Step 4: Fetch Search Results
    # search_results = fetch_google_search_results(query)

    search_results = fetch_bing_search_results(query)
    headers = ["Title", "Snippet", "URL"]

    # # Step 5: Rank Results
    ranked_results = cosine_similarity_ranking(assignment_embedding, search_results)
    
    # Return Top 5 Links
    return ranked_results[:5]
    
# if __name__ == "__main__":
#     file_path = "temp_SPM 3 Temp(1).docx"
#     ranked_results = process_assignment(file_path)
#     print("Top 5 Links:")
#     for idx, result in enumerate(ranked_results, start=1):
#         print(f"{idx}. Title: {result['title']}")
#         print(f"   Snippet: {result['snippet']}")
#         print(f"   URL: {result['url']}")
#         print()
