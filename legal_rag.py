import streamlit as st
from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from mistralai import Mistral
import requests
from bs4 import BeautifulSoup

# ---------- Configuration ----------
st.set_page_config(page_title="Legal AI Assistant", layout="wide", page_icon="⚖️")

api_key = "lww1VhdApvjYGlQFlRlTrzNjrUZf3eno"  # Use environment variable in production
mistral_model = "open-mistral-7b"  # or use "pixtral-12b-2409"

# ---------- Utility Functions ----------
@st.cache_data(show_spinner=False)
def scrape_website(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    texts = soup.get_text(separator=" ", strip=True)
    return texts

@st.cache_resource(show_spinner=False)
def process_text(text):
    docs = [Document(page_content=text)]
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = splitter.split_documents(docs)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(split_docs, embeddings)
    return vectorstore

def ask_mistral(text, user_prompt):
    client = Mistral(api_key=api_key)
    response = client.chat.complete(
        model=mistral_model,
        messages=[
            {
                "role": "system",
                "content": "You are an AI assistant that processes legal text and answers legal questions."
            },
            {"role": "user", "content": f"{user_prompt}\n\n{text}"},
        ],
    )
    return response.choices[0].message.content.strip()

# ---------- Sidebar ----------
st.sidebar.title("⚙️ Configuration")
website_url = st.sidebar.text_input("🔗 Enter legal website URL", value="https://www.legalmatch.com/law-library/")
top_k = st.sidebar.slider("🔍 Top k documents", min_value=1, max_value=10, value=3)

# ---------- Main UI ----------
st.title("⚖️ Legal AI Assistant")
st.markdown("Ask legal questions based on scraped website data.")

with st.spinner("🔎 Scraping and processing website..."):
    raw_text = scrape_website(website_url)
    vectorstore = process_text(raw_text)

with st.expander("📝 Preview Scraped Document"):
    st.write(raw_text[:3000] + "...")

query = st.text_input("❓ Ask your legal question here")

if st.button("Get Answer", type="primary") and query:
    with st.spinner("🤖 Thinking..."):
        retrieved_docs = vectorstore.similarity_search(query, k=top_k)
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])
        answer = ask_mistral(context, query)

    st.markdown("### ✅ Answer:")
    st.success(answer)

    with st.expander("📄 Retrieved Context Used"):
        st.write(context)
