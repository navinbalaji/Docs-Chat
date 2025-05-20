# 📚 Multi-Docs QA with Ollama + Chroma

An interactive Streamlit app for scraping multiple documentation websites, embedding the content with `Ollama`'s `nomic-embed-text`, and querying it with powerful local LLMs via `LangChain`.

Supports conversational memory, multi-doc querying, and Chroma vector storage.

---

## 🚀 Features

- Scrape and embed multiple documentation sites
- Local vector search using `Chroma`
- Local embedding with `Ollama`'s `nomic-embed-text`
- Conversational Q&A with history using `Ollama` LLMs (like `llama3`, `vicuna`, etc.)
- Proxy support for restricted environments
- Fully local and private — no cloud services needed

---

## 🧠 Requirements

- Python 3.9 or above
- Ollama (https://ollama.com)
- Docker (optional for Chroma)
- Streamlit

---

## 📦 Installation


#### Clone this repo

```bash
git clone https://github.com/your-username/multi-docs-qa
cd multi-docs-qa
```

#### Create virtual environment
```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
```

#### Install Python dependencies

```bash
pip install -r requirements.txt
```

#### 📝 `requirements.txt`

```txt
streamlit
requests
beautifulsoup4
langchain
langchain-community
langchain-core
chromadb
ollama
```
## 🛠️ Ollama Setup

1. Install Ollama: [https://ollama.com/download](https://ollama.com/download)
2. Pull required models:

```bash
ollama pull nomic-embed-text
ollama pull llama3-chatqa:8b
```

You can also pull others like:

```bash
ollama pull llama2-chat
ollama pull vicuna
```

---

## 🏁 Running the App

```bash
streamlit run app.py
```

---

## 🌐 Usage

1. **Enter documentation URLs** (one per line).
2. (Optional) Enter proxy if behind a firewall.
3. Click **“Scrape and Embed Docs”**.
4. Once embedding is complete, select an LLM model.
5. Ask any question — and get instant answers from your documentation!

---

## 💡 Example Inputs

**Docs URLs**

```
https://shopify.dev/docs/api
https://shopify.dev/docs/apps
```

**Proxy (optional)**

```
http://username:password@proxyhost:port
```

---

## 🧱 Directory Structure

```
.
├── app.py              # Main Streamlit app
├── chroma_db/          # Persistent vector database
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

---

## 🧹 To Reset Vectorstore

```bash
rm -rf chroma_db
```

---

## 🛠️ Customization Ideas

- 💾 Add local file upload for PDFs or Markdown
- 🔐 Authenticate user access with Streamlit login
- 📁 Add source references with answers
- ☁️ Deploy on a local server, Docker, or Streamlit Cloud

---

## 🧠 How It Works

1. **Scraping**: Pages are fetched and parsed using `requests` + `BeautifulSoup`.
2. **Splitting**: Texts are chunked into manageable chunks with overlap using `LangChain`.
3. **Embedding**: Uses `Ollama`'s `nomic-embed-text` model to generate embeddings.
4. **Storage**: Embeddings are stored in `Chroma` for local vector similarity search.
5. **Q\&A**: User questions are answered via `LangChain`'s `ConversationalRetrievalChain` with memory.

---

## 🙋‍♂️ Questions or Help?

Open an issue or contact the maintainer!

---

## 📜 License

MIT License
