import streamlit as st
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser
import time
import random
import os

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain_community.llms import Ollama
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.retrievers import BaseRetriever
from langchain.schema import Document
from typing import List
from langchain.callbacks.manager import CallbackManagerForRetrieverRun

DB_DIR = "./chroma_db"
DOMAIN_NAME = "shopify.dev"  # Replace with your real domain
USER_AGENT = f"Mozilla/5.0 (compatible; DocQA-Bot/1.0; +https://{DOMAIN_NAME}/bot)"
HEADERS = {"User-Agent": USER_AGENT}

# --- Helper functions ---

def can_fetch_url(url, user_agent=USER_AGENT):
    parsed = urlparse(url)
    rp = RobotFileParser()
    rp.set_url(f"{parsed.scheme}://{parsed.netloc}/robots.txt")
    try:
        rp.read()
        return rp.can_fetch(user_agent, url)
    except:
        return True  # If robots.txt unreachable, allow cautiously

def polite_scrape_site(base_url, proxy=None):
    if not can_fetch_url(base_url):
        st.warning(f"Robots.txt disallows crawling {base_url}. Skipping.")
        return []

    visited, to_visit, all_texts = set(), [base_url], []
    proxies = {"http": proxy, "https": proxy} if proxy else None

    while to_visit:
        url = to_visit.pop()
        if url in visited:
            continue
        visited.add(url)

        if not can_fetch_url(url):
            st.warning(f"Robots.txt disallows crawling {url}. Skipping.")
            continue

        for retry in range(3):
            try:
                r = requests.get(url, headers=HEADERS, proxies=proxies, timeout=10)
                r.raise_for_status()
                soup = BeautifulSoup(r.text, "html.parser")
                text = soup.get_text(separator="\n", strip=True)
                all_texts.append(text)

                for a in soup.find_all("a", href=True):
                    full_url = urljoin(url, a["href"])
                    if full_url.startswith(base_url) and full_url not in visited:
                        to_visit.append(full_url)

                time.sleep(random.uniform(1, 3))
                break
            except Exception as e:
                st.warning(f"Error fetching {url}: {e}. Retry {retry+1}/3")
                time.sleep(2 ** retry)
        else:
            st.error(f"Failed to crawl {url}. Skipping.")
    return all_texts

def embed_docs(site_url, proxy=None):
    st.info(f"Scraping site: {site_url} ... please wait.")
    raw_texts = polite_scrape_site(site_url, proxy=proxy)
    st.info(f"Scraped {len(raw_texts)} pages. Embedding...")

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    docs = splitter.create_documents(raw_texts)

    embedding_model = OllamaEmbeddings(model="nomic-embed-text")

    domain = urlparse(site_url).netloc.replace(".", "_")
    db_path = os.path.join(DB_DIR, domain)
    os.makedirs(DB_DIR, exist_ok=True)

    vectorstore = Chroma.from_documents(docs, embedding_model, persist_directory=db_path)
    vectorstore.persist()

    st.session_state.vectorstores[domain] = vectorstore
    st.success(f"Finished embedding docs for {domain}")

# --- Fixed CombinedRetriever class ---

class CombinedRetriever(BaseRetriever):
    def __init__(self, retrievers: List[BaseRetriever]):
        super().__init__()
        self._retrievers = retrievers

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun = None
    ) -> List[Document]:
        docs = []
        for retriever in self._retrievers:
            docs.extend(retriever.get_relevant_documents(query))
        return docs

# --- Streamlit app ---

st.title("📚 Multi-Docs QA with Ollama + Chroma")

# Initialize session state
if "vectorstores" not in st.session_state:
    st.session_state.vectorstores = {}
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

docs_input = st.text_area("Enter docs URLs (one per line):", height=150)
proxy = st.text_input("Optional proxy (http://user:pass@host:port):", value="")

model_options = [
    "llama3-chatqa:8b",
    "llama3-chat",
    "llama2-chat",
    "vicuna",
    "alpaca",
    "wizard-vicuna",
]
llm_model = st.selectbox("Choose Ollama LLM model for Q&A:", model_options, index=0)

if st.button("Scrape and Embed Docs"):
    urls = [u.strip() for u in docs_input.splitlines() if u.strip()]
    if not urls:
        st.error("Please enter at least one URL.")
    else:
        for url in urls:
            embed_docs(url, proxy=proxy if proxy else None)

if st.session_state.vectorstores:
    retrievers = [vs.as_retriever(search_kwargs={"k": 3}) for vs in st.session_state.vectorstores.values()]
    combined_retriever = CombinedRetriever(retrievers)

    llm = Ollama(model=llm_model)

    qa = ConversationalRetrievalChain.from_llm(llm, combined_retriever, return_source_documents=False)

    query = st.text_input("Ask a question about your docs:")

    if st.button("Ask") and query:
        result = qa.invoke({"question": query, "chat_history": st.session_state.chat_history})

        st.session_state.chat_history.append(HumanMessage(content=query))
        st.session_state.chat_history.append(AIMessage(content=result["answer"]))

        st.markdown(f"**Answer:** {result['answer']}")

        st.markdown("---")
        st.markdown("### Chat history")
        for i in range(0, len(st.session_state.chat_history), 2):
            user_msg = st.session_state.chat_history[i].content
            bot_msg = st.session_state.chat_history[i + 1].content if i + 1 < len(st.session_state.chat_history) else ""
            st.markdown(f"**You:** {user_msg}")
            st.markdown(f"**Bot:** {bot_msg}")
else:
    st.info("Enter documentation URLs above, scrape & embed to start querying.")
