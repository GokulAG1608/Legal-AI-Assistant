# ⚖️ Legal AI Assistant

This project is an intelligent legal question-answering system built with **Streamlit**, **LangChain**, **FAISS**, **HuggingFace Embeddings**, and the **Mistral API**. It scrapes content from law-related websites, embeds it for retrieval, and answers user questions using a large language model.

---

## 🚀 Features

- 🧠 Scrapes legal content from URLs (e.g., Indian Kanoon)
- 🔍 Splits and embeds text using HuggingFace Sentence Transformers
- 📚 Stores embeddings using FAISS for fast vector search
- 🤖 Answers questions using Mistral API (LLM)
- 🖥️ Interactive and modern Streamlit UI
- 📂 Expander sections for retrieved text and previews

---

## 📦 Installation

Make sure Python 3.8+ is installed.

```bash
# 1. Clone the repository
git clone https://github.com/your-username/legal-ai-assistant.git
cd legal-ai-assistant

# 2. Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
