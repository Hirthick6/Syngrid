"""RAG Service - Handles web scraping, embeddings, and Q&A"""
import os
import time
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
load_dotenv()

class RAGService:
    """Complete RAG service with web scraping and Q&A"""
    
    def __init__(self):
        self.retriever = None
        self.cache = {}
        self.status = {
            "ready": False,
            "message": "Not initialized",
            "pages_scraped": 0
        }
        self.embeddings = None
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        self.api_base = "https://openrouter.ai/api/v1/chat/completions"
        self.model = "meta-llama/llama-3.2-3b-instruct:free"
        
        if not self.api_key:
            print("⚠️  WARNING: OPENROUTER_API_KEY not found in environment variables")

    def scrape_single_page(self, url, visited, base_domain):
        """Scrape a single page - optimized for parallel execution"""
        if url in visited:
            return None, []
        
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            resp = requests.get(url, headers=headers, timeout=15, verify=True)
            
            if resp.status_code != 200:
                print(f"  ⚠️  Status {resp.status_code}: {url[:50]}")
                return None, []

            visited.add(url)
            soup = BeautifulSoup(resp.text, "html.parser")
            
            # Remove unwanted elements
            for tag in soup(["script", "style", "nav", "footer", "header", "aside", "iframe", "noscript"]):
                tag.decompose()

            # Extract text
            text = soup.get_text(separator="\n", strip=True)
            lines = [line.strip() for line in text.split("\n") if line.strip() and len(line.strip()) > 3]
            clean_text = "\n".join(lines)

            # Extract links
            links = []
            for link in soup.find_all("a", href=True):
                try:
                    next_url = urljoin(url, link["href"])
                    parsed = urlparse(next_url)
                    
                    # Only include valid HTTP(S) links from same domain
                    if (parsed.scheme in ['http', 'https'] and 
                        parsed.netloc == base_domain and 
                        next_url not in visited and
                        not next_url.endswith(('.pdf', '.jpg', '.png', '.gif', '.zip'))):
                        links.append(next_url)
                except Exception:
                    continue

            return clean_text if len(clean_text) > 100 else None, links

        except requests.Timeout:
            print(f"  ⏱️  Timeout: {url[:50]}")
            return None, []
        except requests.RequestException as e:
            print(f"  ❌ Request error for {url[:50]}: {str(e)[:50]}")
            return None, []
        except Exception as e:
            print(f"  ❌ Error scraping {url[:50]}: {str(e)[:50]}")
            return None, []

    def scrape_website(self, base_url, max_pages=25):
        """Parallel web scraping for faster performance"""
        visited = set()
        all_content = []
        to_visit = [base_url]
        base_domain = urlparse(base_url).netloc

        print(f"🔍 Scraping {base_url} (max {max_pages} pages)...")
        start_time = time.time()

        with ThreadPoolExecutor(max_workers=5) as executor:
            while to_visit and len(visited) < max_pages:
                # Get next batch of URLs to scrape
                batch_size = min(5, max_pages - len(visited))
                batch = to_visit[:batch_size]
                to_visit = to_visit[batch_size:]
                
                # Submit scraping tasks
                futures = {
                    executor.submit(self.scrape_single_page, url, visited, base_domain): url 
                    for url in batch if url not in visited
                }

                # Process completed tasks
                for future in as_completed(futures):
                    url = futures[future]
                    try:
                        content, links = future.result()
                        if content:
                            all_content.append(content)
                            print(f"  ✓ [{len(visited)}/{max_pages}] {url[:70]}")
                        
                        # Add new links to queue
                        for link in links:
                            if link not in visited and link not in to_visit and len(to_visit) < 100:
                                to_visit.append(link)
                                
                    except Exception as e:
                        print(f"  ✗ Failed processing {url[:50]}: {str(e)[:50]}")

        elapsed = time.time() - start_time
        self.status["pages_scraped"] = len(visited)
        
        print(f"\n✅ Scraped {len(visited)} pages in {elapsed:.1f}s")
        print(f"📊 Content: {len(''.join(all_content)):,} chars\n")

        return "\n\n".join(all_content)

    def initialize_rag(self, url="https://syngrid.com/", max_pages=25):
        """Initialize RAG system with website content"""
        try:
            print("\n" + "="*60)
            print("🚀 INITIALIZING RAG CHATBOT")
            print("="*60)

            # Step 1: Scrape website
            self.status["message"] = "Scraping website..."
            content = self.scrape_website(url, max_pages)

            if len(content) < 500:
                self.status["message"] = "Insufficient content scraped"
                print("❌ Not enough content scraped")
                return False

            # Step 2: Split into chunks
            self.status["message"] = "Processing content..."
            print("\n📄 Splitting into chunks...")
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1500,
                chunk_overlap=150,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
            chunks = splitter.split_text(content)
            print(f"  ✓ Created {len(chunks)} chunks")

            if len(chunks) == 0:
                self.status["message"] = "No chunks created"
                print("❌ Failed to create text chunks")
                return False

            # Step 3: Load embeddings
            self.status["message"] = "Loading AI model..."
            print("\n🧠 Loading embeddings model...")
            
            if self.embeddings is None:
                try:
                    self.embeddings = HuggingFaceEmbeddings(
                        model_name="sentence-transformers/all-MiniLM-L6-v2",
                        model_kwargs={'device': 'cpu'},
                        encode_kwargs={'normalize_embeddings': True}
                    )
                    print("  ✓ Model loaded successfully")
                except Exception as e:
                    print(f"  ❌ Failed to load embeddings model: {e}")
                    self.status["message"] = f"Model loading error: {str(e)}"
                    return False

            # Step 4: Build vector database
            self.status["message"] = "Building knowledge base..."
            print("\n💾 Building vector database...")
            try:
                vectorstore = FAISS.from_texts(
                    chunks,
                    embedding=self.embeddings
                )
                print("  ✓ Vector database ready")
            except Exception as e:
                print(f"  ❌ Failed to build vector database: {e}")
                self.status["message"] = f"Vector DB error: {str(e)}"
                return False

            self.retriever = vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 4}
            )

            self.status["ready"] = True
            self.status["message"] = "Ready"
            print("\n✅ RAG Chatbot initialized successfully!\n")
            print("="*60 + "\n")
            return True

        except Exception as e:
            self.status["message"] = f"Error: {str(e)}"
            print(f"\n❌ Initialization failed: {e}\n")
            import traceback
            traceback.print_exc()
            return False

    def call_llm(self, question, context):
        """Call OpenRouter API for answer generation"""
        if not self.api_key:
            return "⚠️ API key not configured. Please set OPENROUTER_API_KEY environment variable."

        prompt = f"""Answer the question based ONLY on the context below. Be concise and helpful.

Context:
{context[:3000]}

Question: {question}

Answer (2-4 sentences):"""

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/yourusername/rag-chatbot",
            "X-Title": "RAG Chatbot"
        }

        try:
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant. Answer questions concisely based on the provided context."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.3,
                "max_tokens": 200
            }

            response = requests.post(
                self.api_base,
                headers=headers,
                json=payload,
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                if "choices" in result and len(result["choices"]) > 0:
                    return result["choices"][0]["message"]["content"].strip()
                else:
                    print(f"❌ Unexpected API response format: {result}")
                    return "I apologize, but I received an unexpected response. Please try again."
            else:
                print(f"❌ LLM Error: {response.status_code} - {response.text}")
                return "I apologize, but I'm having trouble generating a response. Please try again."

        except requests.Timeout:
            print("❌ LLM request timeout")
            return "Sorry, the request timed out. Please try again."
        except Exception as e:
            print(f"❌ LLM Exception: {e}")
            import traceback
            traceback.print_exc()
            return "Sorry, I couldn't process your question. Please try again."

    def ask_question(self, question):
        """Answer question using RAG"""
        if not self.status["ready"]:
            return "⏳ Chatbot is still initializing. Please wait a moment..."

        # Validate question
        if not question or len(question.strip()) < 3:
            return "Please ask a complete question."

        # Check cache
        q_key = question.lower().strip()
        if q_key in self.cache:
            return self.cache[q_key]

        # Retrieve relevant documents
        try:
            docs = self.retriever.invoke(question)
            if not docs:
                return "I don't have enough information to answer that question."

            context = "\n\n".join([doc.page_content for doc in docs])
            answer = self.call_llm(question, context)

            # Cache result
            if len(self.cache) < 1000:  # Limit cache size
                self.cache[q_key] = answer
            
            return answer
            
        except Exception as e:
            print(f"❌ Error answering question: {e}")
            import traceback
            traceback.print_exc()
            return "I encountered an error processing your question. Please try again."

    def get_status(self):
        """Get current status"""
        return self.status.copy()

    def clear_cache(self):
        """Clear answer cache"""
        cache_size = len(self.cache)
        self.cache.clear()
        return cache_size


# Initialize global RAG service
rag_service = RAGService()
