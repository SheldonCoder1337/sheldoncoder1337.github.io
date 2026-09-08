# Retrieval-Augmented Generation System Architecture

## Overview

2024 University of Macau, Institute of Collaborative Innovation, Data Science, AI Application and Development Fall Course: Introduction to Retrieval-Augmented Generation System Architecture

### Architecture

RAG overall architecture

<iframe src="/statics/cs08-software-engineering/02-rag-arc/rag-src-architecture.html" width="100%" height="500px" style="border: none;"></iframe>

### Runtime Dataflow

RAG runtime dataflow · modules × chapters

<iframe src="/statics/cs08-software-engineering/02-rag-arc/rag-runtime-dataflow.html" width="100%" height="500px" style="border: none;"></iframe>

## Step-by-Step

### Config

Centralized configuration management: all adjustable parameters, paths, and keys are here; change them in one place.

Conventions:
- Sensitive information (API keys) is only read from environment variables / .env — never hard-coded in the source.
- Paths are always derived from the location of this file, not from the runtime working directory, so the script works no matter where it is invoked from.

```python title="config.py"
import os
from pathlib import Path

# Explicitly point to rag_project/.env so the script can find it regardless of the working directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# python-dotenv is just a convenience for reading keys from the .env file;
# if it is not installed, reading environment variables directly still works.
# (This allows smoke tests / CI to run with zero dependencies.)
try:
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")
except ModuleNotFoundError:
    pass

# --- Paths ---
DATA_DIR = PROJECT_ROOT / "data" / "sample_docs"
CHROMA_DIR = PROJECT_ROOT / os.getenv("CHROMA_PERSIST_DIR", "chroma_data")

# --- LLM ---
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-chat")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.3"))

# --- Embedding / Reranker models ---
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")

# --- Chunking / Retrieval / Reranking defaults (scripts can override) ---
CHUNK_SIZE = 512  # target number of characters per chunk (Chapter 3)
CHUNK_OVERLAP = 50  # overlap characters between adjacent chunks, to avoid cutting a sentence in half
RETRIEVE_TOP_N = 20  # candidate pool size for recall: each of the two paths takes top N before fusion (Chapter 5)
RERANK_TOP_K = 5  # number of items kept after reranking, finally fed to the LLM (Chapter 6)
TOP_K = 5  # when reranking is disabled, the number of chunks returned directly by retrieval

# Switches: if the heavy models are not installed, or you want to get the pipeline running first,
# you can disable these in .env (set to 0 / false)
USE_RERANK = os.getenv("USE_RERANK", "1").lower() not in ("0", "false", "no")
USE_QUERY_REWRITE = os.getenv("USE_QUERY_REWRITE", "1").lower() not in ("0", "false", "no")


def validate_config() -> None:
    """Validate critical configuration at startup; give clear Chinese prompts for missing items
    instead of a stack trace when the API is called."""
    if not DEEPSEEK_API_KEY or DEEPSEEK_API_KEY.startswith("sk-your-key"):
        raise ValueError(
            "\n❌ Missing DEEPSEEK_API_KEY\n"
            "   1) Copy the config template: cp .env.example .env\n"
            "   2) Edit .env and fill in your DeepSeek API key\n"
            "   3) Apply at: https://platform.deepseek.com/api_keys\n"
        )
```

### Loader

PDF loading module — corresponds to the "loading" step in Chapter 3 (Document Preprocessing and Chunking).  
Reads PDFs from disk into plain text + metadata and hands them downstream for chunking.

```python title="loader.py"
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class LoadedDoc:
    """A single document after loading."""

    source: str  # file name, used for provenance (to tell the user where the answer came from)
    text: str  # extracted full text
    metadata: dict = field(default_factory=dict)  # additional info like page count, character count


def load_pdf(path: Path) -> LoadedDoc:
    """Load a single PDF file into a LoadedDoc.

    Args:
        path: PDF file path

    Returns:
        LoadedDoc containing full text and source metadata
    """
    from pypdf import PdfReader  # lazy import: only depend on pypdf when actually reading PDFs

    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    text = "\n".join(pages)
    return LoadedDoc(
        source=path.name,
        text=text,
        metadata={"num_pages": len(pages), "num_chars": len(text)},
    )


def load_dir(dir_path: Path) -> list[LoadedDoc]:
    """Load all PDFs in a directory.

    Args:
        dir_path: directory containing PDFs (default data/sample_docs/)

    Returns:
        list of LoadedDoc, one per PDF (sorted by file name for reproducibility)
    """
    docs: list[LoadedDoc] = []
    for pdf_path in sorted(Path(dir_path).glob("*.pdf")):
        doc = load_pdf(pdf_path)
        if doc.text.strip():  # skip empty PDFs with no extractable text
            docs.append(doc)
    return docs
```

### Chunker

Document chunking module — corresponds to Chapter 3 (Document Preprocessing and Chunking).

Splits long texts into small chunks with metadata. Chunking quality directly determines retrieval quality —  
the synthetic teaching scenario in the article where "the nuclear radiation clause is cut in the middle"  
illustrates the risk of improper chunking.

This implementation performs **structure-aware + semantic-boundary** chunking (simplified version):  
it prefers to cut at the boundaries of "Article N / headings / paragraphs", falls back to sentence boundaries  
when a unit is too long, and aligns overlaps to complete sentences so that a sentence is rarely split mid-way.

```python title="chunker.py"
import re
from dataclasses import dataclass, field

from .loader import LoadedDoc

# Structural boundaries: common Chinese clause patterns like "第N条 / 一、 / （一） / 1." etc.;
# matches are preferred positions to start a new block.
_HEADING_RE = re.compile(
    r"(?=第[一二三四五六七八九十百零\d]+条)"  # 第一条 / 第10条
    r"|(?=^[一二三四五六七八九十]+、)"  # 一、二、
    r"|(?=^（[一二三四五六七八九十\d]+）)",  # (一)(二)
    re.MULTILINE,
)
# Sentence boundaries: used for secondary splitting of very long paragraphs and for aligning overlaps.
_SENT_RE = re.compile(r"(?<=[。!?；!?;\n])")


@dataclass
class Chunk:
    """A text block after chunking."""

    text: str  # the text of this chunk
    source: str  # source file name (for provenance)
    chunk_id: int  # sequence number within the source document
    metadata: dict = field(default_factory=dict)


def _split_units(text: str) -> list[str]:
    """First split by structural boundaries into 'semantic units';
    units that are still too long are then split by sentence boundaries."""
    units: list[str] = []
    for block in _HEADING_RE.split(text):
        block = block.strip()
        if not block:
            continue
        # if a single structural block is not too long, keep it whole;
        # otherwise split it into smaller units by sentence boundaries.
        if len(block) <= 1024:
            units.append(block)
        else:
            units.extend(s for s in _SENT_RE.split(block) if s.strip())
    return units


def _tail_overlap(text: str, overlap: int) -> str:
    """Take the last `overlap` characters of a text and trim them to the nearest sentence boundary
    (so the overlap does not start with a half sentence)."""
    if overlap <= 0 or len(text) <= overlap:
        return text if overlap > 0 else ""
    tail = text[-overlap:]
    # find the first sentence start inside the tail to avoid an overlap beginning mid-sentence
    m = re.search(r"[。!?；!?;\n]", tail)
    return tail[m.end():] if m else tail


def chunk_text(
    text: str,
    chunk_size: int = 512,
    chunk_overlap: int = 50,
    source: str = "",
) -> list[Chunk]:
    """Split a piece of text into a list of Chunks.

    Args:
        text: text to be chunked
        chunk_size: target chunk size (number of characters)
        chunk_overlap: overlap characters between adjacent chunks, to avoid splitting sentences
        source: document source (written into each chunk's metadata)

    Returns:
        list of Chunks (in order, chunk_id starts from 0)
    """
    chunks: list[Chunk] = []
    buf = ""  # the chunk currently being accumulated

    def flush() -> None:
        nonlocal buf
        if buf.strip():
            cid = len(chunks)
            chunks.append(Chunk(text=buf.strip(), source=source, chunk_id=cid, metadata={"source": source}))
        buf = ""

    for unit in _split_units(text):
        # if it fits, keep accumulating; otherwise close the current chunk and start a new one with overlap
        if buf and len(buf) + len(unit) > chunk_size:
            prev = buf
            flush()
            buf = _tail_overlap(prev, chunk_overlap)
        buf += unit
        # after accumulation, if still too long (a single unit is huge), close it directly to avoid unbounded growth
        while len(buf) > chunk_size:
            cut = buf[:chunk_size]
            m = list(_SENT_RE.finditer(cut))
            split_at = m[-1].end() if m else chunk_size  # try to cut at a sentence boundary
            cid = len(chunks)
            chunks.append(Chunk(text=buf[:split_at].strip(), source=source, chunk_id=cid, metadata={"source": source}))
            buf = _tail_overlap(buf[:split_at], chunk_overlap) + buf[split_at:]
    flush()
    return chunks


def chunk_doc(doc: LoadedDoc, chunk_size: int = 512, chunk_overlap: int = 50) -> list[Chunk]:
    """Chunk a loaded document (a convenience wrapper around chunk_text)."""
    return chunk_text(doc.text, chunk_size=chunk_size, chunk_overlap=chunk_overlap, source=doc.source)
```

### Embedding

Embedding module — corresponds to Chapter 4 (Embedding Model Selection).

Converts text into vectors so that "similar meaning" can be measured by "close vector distance".  
By default it uses bge-m3 for local inference (Chinese-friendly, free, data never leaves the machine).

```python title="embedder.py"
class Embedder:
    """Text vectorizer. Encapsulates model loading and batch encoding; downstream code only calls encode.

    The model is loaded lazily on first use, so merely importing this module does not trigger a ~2GB download.
    """

    def __init__(self, model_name: str = "BAAI/bge-m3") -> None:
        """Record the model name; actual model loading is deferred until the first encode (lazy).

        Args:
            model_name: HuggingFace model name, default bge-m3
        """
        self.model_name = model_name
        self._model = None  # lazy: loaded only when first needed

    def _ensure_model(self):
        """Load the model on first encode (avoid pulling 2GB of dependencies just by importing)."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer  # lazy import

            self._model = SentenceTransformer(self.model_name)
        return self._model

    def encode(self, texts: list[str]) -> list[list[float]]:
        """Encode a batch of texts into (normalized) vectors.

        Args:
            texts: list of texts

        Returns:
            list of vectors, one per input
        """
        model = self._ensure_model()
        # after normalization, inner product and cosine are equivalent;
        # tolist() converts to pure Python for easy storage in ChromaDB
        vecs = model.encode(texts, normalize_embeddings=True)
        return [v.tolist() for v in vecs]

    def encode_query(self, query: str) -> list[float]:
        """Encode a single query (some models use a special instruction prefix for queries, so a separate interface)."""
        return self.encode([query])[0]
```

### Vectorstore

Vector store module — a local ChromaDB wrapper, corresponding to the storage side of Chapter 5 (Retrieval Recall).

Stores chunk vectors and retrieves them by similarity. Uses ChromaDB local persistence — zero ops, clone and run.

```python title="vectorstore.py"
from pathlib import Path
from .chunker import Chunk


class VectorStore:
    """ChromaDB vector store wrapper. chromadb is imported lazily, so importing this module alone
    does not require it to be installed."""

    def __init__(self, persist_dir: Path, collection: str = "rag_docs") -> None:
        """Open or create a persistent vector collection.

        Args:
            persist_dir: local persistence directory
            collection: collection name
        """
        import chromadb  # lazy import

        self._client = chromadb.PersistentClient(path=str(persist_dir))
        # disable Chroma's built-in embedding (we compute vectors ourselves with bge-m3 and pass them in)
        self._col = self._client.get_or_create_collection(
            name=collection, metadata={"hnsw:space": "cosine"}
        )

    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        """Write chunks and their vectors into the store.

        Args:
            chunks: list of text chunks
            embeddings: vectors corresponding one-to-one to chunks
        """
        if not chunks:
            return
        ids = [f"{c.source}#{c.chunk_id}" for c in chunks]  # source + sequence, stable and deduplicable
        self._col.add(
            ids=ids,
            embeddings=embeddings,
            documents=[c.text for c in chunks],
            metadatas=[{"source": c.source, "chunk_id": c.chunk_id} for c in chunks],
        )

    def query(self, query_embedding: list[float], top_k: int = 5) -> list[Chunk]:
        """Retrieve the most similar chunks by vector similarity.

        Args:
            query_embedding: query vector
            top_k: number of chunks to return

        Returns:
            list of Chunks ordered by decreasing similarity
        """
        res = self._col.query(query_embeddings=[query_embedding], n_results=top_k)
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        return [
            Chunk(text=d, source=m.get("source", ""), chunk_id=int(m.get("chunk_id", 0)), metadata=dict(m))
            for d, m in zip(docs, metas)
        ]

    def all_chunks(self) -> list[Chunk]:
        """Return all chunks in the store (the BM25 path needs full text to rebuild the inverted index in memory)."""
        res = self._col.get(include=["documents", "metadatas"])
        docs = res.get("documents", []) or []
        metas = res.get("metadatas", []) or []
        return [
            Chunk(text=d, source=m.get("source", ""), chunk_id=int(m.get("chunk_id", 0)), metadata=dict(m))
            for d, m in zip(docs, metas)
        ]

    def count(self) -> int:
        """Return the number of chunks currently in the store (for health checks after indexing)."""
        return self._col.count()
```

### Retriever

Retrieval module — corresponds to Chapter 5 (Retrieval Recall).

Vector retrieval excels at "meaning similarity", while BM25 excels at "exact keyword matching"; the two are complementary.  
This module implements dual-path retrieval + fusion (RRF), which catches synonymous rewrites and does not miss proper nouns.  
RRF (Reciprocal Rank Fusion) considers only ranks, not scores, naturally sidestepping the problem of different score scales.

```python title="retriever.py"
import math
import re
from dataclasses import dataclass

from .chunker import Chunk
from .embedder import Embedder
from .vectorstore import VectorStore


@dataclass
class RetrievedChunk:
    """A retrieval result: a chunk plus its relevance score (here the RRF fused score)."""

    chunk: Chunk
    score: float


def _tokenize(text: str) -> list[str]:
    """Tokenization: prefer jieba (word-level, more accurate); fall back to character-level if not installed.
    Both enable BM25 to work."""
    try:
        import jieba  # optional dependency

        return [t for t in jieba.lcut(text) if t.strip()]
    except Exception:
        return [ch for ch in re.sub(r"\s", "", text)]  # character-level fallback


class _BM25:
    """Minimal usable BM25 implementation (corresponds to the hand-written one in Chapter 5, slightly engineered)."""

    def __init__(self, docs_tokens: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs = docs_tokens
        self.N = len(docs_tokens)
        self.avgdl = sum(len(d) for d in docs_tokens) / self.N if self.N else 0.0
        self.df: dict[str, int] = {}
        for d in docs_tokens:
            for t in set(d):
                self.df[t] = self.df.get(t, 0) + 1

    def _idf(self, t: str) -> float:
        n = self.df.get(t, 0)
        return math.log(1 + (self.N - n + 0.5) / (n + 0.5))

    def scores(self, query: str) -> list[float]:
        q = _tokenize(query)
        out = []
        for doc in self.docs:
            score, dl = 0.0, len(doc)
            for t in q:
                f = doc.count(t)
                if not f:
                    continue
                tf = f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1)))
                score += self._idf(t) * tf
            out.append(score)
        return out


class Retriever:
    """Hybrid vector + BM25 retriever."""

    def __init__(self, store: VectorStore, embedder: Embedder, chunks: list[Chunk]) -> None:
        """Initialize the retriever.

        Args:
            store: vector store with an already built index (vector path)
            embedder: query vectorizer (vector path)
            chunks: all chunks (BM25 path builds its inverted index in memory)
        """
        self.store = store
        self.embedder = embedder
        self.chunks = chunks
        self._bm25 = _BM25([_tokenize(c.text) for c in chunks]) if chunks else None

    @staticmethod
    def _key(c: Chunk) -> str:
        return f"{c.source}#{c.chunk_id}"

    def retrieve(self, query: str, top_k: int = 5, top_n: int = 20, rrf_k: int = 60) -> list[RetrievedChunk]:
        """Hybrid retrieval: each path takes top_n candidates, then RRF fusion, return top_k.

        Args:
            query: user query
            top_k: final number of results to return
            top_n: number of candidates recalled per path (larger pool raises recall ceiling)
            rrf_k: RRF constant, empirical value 60

        Returns:
            list of RetrievedChunk sorted by fused score
        """
        ranked_lists: list[list[Chunk]] = []

        # Vector path (auto-skip if the store is empty or embedding model missing, so it doesn't crash)
        try:
            qv = self.embedder.encode_query(query)
            ranked_lists.append(self.store.query(qv, top_k=top_n))
        except Exception:
            pass

        # BM25 path
        if self._bm25 is not None:
            scores = self._bm25.scores(query)
            order = sorted(range(len(self.chunks)), key=lambda i: scores[i], reverse=True)
            ranked_lists.append([self.chunks[i] for i in order[:top_n] if scores[i] > 0])

        # RRF fusion: each chunk contributes 1/(rrf_k + rank) in each ranked list
        fused: dict[str, float] = {}
        by_key: dict[str, Chunk] = {}
        for lst in ranked_lists:
            for rank, c in enumerate(lst, start=1):
                k = self._key(c)
                fused[k] = fused.get(k, 0.0) + 1.0 / (rrf_k + rank)
                by_key.setdefault(k, c)

        ordered = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        return [RetrievedChunk(chunk=by_key[k], score=s) for k, s in ordered]
```

### Query

Query understanding and rewriting module — corresponds to Chapter 7.

Users' raw questions are often colloquial, vague, and missing keywords. Rewriting / expanding the query before retrieval  
may improve recall on samples where the query and document wording differ; whether it helps, and the latency / cost trade-offs,  
must be measured on the same evaluation set and runtime configuration.

Two paths:
- With a DeepSeek key and rewriting enabled → call the LLM for normalized rewriting + synonymous expansion (better results);
- Without a key or disabled → rule-based fallback: remove filler words (ensures the pipeline runs without a key and never crashes).

```python title="query_processor.py"
import json
import re
from dataclasses import dataclass, field

from . import config

# Colloquial / meaningless words removed by the rule-based fallback
_FILLER = ["请问", "麻烦问下", "我想问一下", "想了解一下", "一下", "请", "啊", "呢", "呀", "吧", "哈"]


@dataclass
class ProcessedQuery:
    """A processed query: one normalized main query plus several expanded queries."""

    original: str  # user's original question
    rewritten: str  # normalized query after rewriting
    expansions: list[str] = field(default_factory=list)  # synonymous / multi-angle expanded queries

    def all_queries(self) -> list[str]:
        """Main query + expansions deduplicated, for retrieval to try one by one."""
        seen, out = set(), []
        for q in [self.rewritten, *self.expansions, self.original]:
            q = q.strip()
            if q and q not in seen:
                seen.add(q)
                out.append(q)
        return out


def _rule_rewrite(query: str) -> str:
    """Rule-based fallback rewriting: remove filler words, trim leading/trailing punctuation and whitespace."""
    q = query
    for f in _FILLER:
        q = q.replace(f, "")
    return q.strip(" ?？!!。,,") or query.strip()


def _llm_rewrite(query: str) -> ProcessedQuery:
    """Call DeepSeek to rewrite and expand; require JSON output; fall back to rule-based on parse failure."""
    from openai import OpenAI  # lazy import

    client = OpenAI(api_key=config.DEEPSEEK_API_KEY, base_url=config.DEEPSEEK_BASE_URL)
    prompt = (
        "你是检索查询改写器。把用户问题改写成更书面、更贴近文档表达的一句话,"
        "并给出 1-3 个同义/不同角度的扩展查询(用于提高召回)。\n"
        '只输出 JSON,格式:{"rewritten": "...", "expansions": ["...", "..."]}\n\n'
        f"用户问题:{query}"
    )
    resp = client.chat.completions.create(
        model=config.LLM_MODEL,
        temperature=0.2,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = resp.choices[0].message.content or ""
    m = re.search(r"\{.*\}", raw, re.S)  # tolerate extra explanatory text from the model
    try:
        data = json.loads(m.group(0)) if m else {}
        rewritten = (data.get("rewritten") or "").strip() or _rule_rewrite(query)
        expansions = [e.strip() for e in data.get("expansions", []) if e and e.strip()]
        return ProcessedQuery(original=query, rewritten=rewritten, expansions=expansions[:3])
    except (ValueError, AttributeError):
        return ProcessedQuery(original=query, rewritten=_rule_rewrite(query))


def process_query(query: str) -> ProcessedQuery:
    """Understand, rewrite, and expand the raw query.

    Args:
        query: user's original question

    Returns:
        ProcessedQuery
    """
    has_key = config.DEEPSEEK_API_KEY and not config.DEEPSEEK_API_KEY.startswith("sk-your-key")
    if config.USE_QUERY_REWRITE and has_key:
        try:
            return _llm_rewrite(query)
        except Exception:
            pass  # network/quota issues should not crash the entire Q&A; degrade to rule-based
    return ProcessedQuery(original=query, rewritten=_rule_rewrite(query))
```

### Reranker

Reranking module — corresponds to Chapter 6 (Reranking and Retrieval Optimization).

Retrieval recall aims to be "fast and broad", so irrelevant chunks inevitably slip in. Reranking uses a heavier  
cross-encoder model to precisely score each candidate's relevance, pushing the truly relevant ones to the top  
before feeding them to the LLM.

```python title="reranker.py"
from .retriever import RetrievedChunk


class Reranker:
    """Reranker based on a cross-encoder. Model is lazily loaded; importing this module alone does not trigger a download."""

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3") -> None:
        """Record the model name; actual loading is deferred until the first rerank.

        Args:
            model_name: cross-encoder reranking model name
        """
        self.model_name = model_name
        self._model = None

    def _ensure_model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder  # lazy import

            self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(self, query: str, candidates: list[RetrievedChunk], top_k: int = 5) -> list[RetrievedChunk]:
        """Rescore and reorder candidate chunks.

        Args:
            query: user query
            candidates: candidates recalled during retrieval
            top_k: number of candidates to keep after reranking

        Returns:
            list of RetrievedChunk sorted by reranking score (descending)
        """
        if not candidates:
            return []
        model = self._ensure_model()
        # Batch inference: feed all (query, doc) pairs at once instead of looping one by one
        # (first speed-up trick in Chapter 6)
        pairs = [(query, rc.chunk.text) for rc in candidates]
        scores = model.predict(pairs)
        reranked = [RetrievedChunk(chunk=rc.chunk, score=float(s)) for rc, s in zip(candidates, scores)]
        reranked.sort(key=lambda rc: rc.score, reverse=True)
        return reranked[:top_k]
```

### Generator

Answer generation module — corresponds to Chapter 9 (Citation-aware Answer Generation).

Puts the retrieved context into a prompt, calls the LLM to generate an answer, and requires citation of sources,  
achieving "traceable and not making things up".

```python title="generator.py"
from dataclasses import dataclass, field

from . import config
from .retriever import RetrievedChunk

# System prompt: hard-code "only based on provided material + cite sources + refuse to answer if absent"
# as the first barrier against hallucination (Chapter 9)
_SYSTEM_PROMPT = (
    "你是严谨的保险客服助手。请严格遵守:\n"
    "1) 只能依据【参考资料】回答,资料里没有的信息,直接说「根据现有资料无法确定」,绝不编造;\n"
    "2) 在用到资料的地方,用 [编号] 标注来源(对应资料前的序号),做到有据可查;\n"
    "3) 回答简洁、用词通俗,面向没有保险背景的普通用户。"
)


@dataclass
class Answer:
    """Generated answer + cited sources."""

    text: str  # answer body
    sources: list[str] = field(default_factory=list)  # names of source files cited


def _format_contexts(contexts: list[RetrievedChunk]) -> str:
    """Number and format retrieved chunks into a 【参考资料】 block for prompt citation."""
    blocks = []
    for i, rc in enumerate(contexts, start=1):
        blocks.append(f"[{i}] (来源:{rc.chunk.source}) {rc.chunk.text}")
    return "\n\n".join(blocks)


def build_messages(query: str, contexts: list[RetrievedChunk]) -> list[dict]:
    """Construct messages to send to the LLM (pure function, no network, easy for unit testing / prompt inspection)."""
    user = f"【参考资料】\n{_format_contexts(contexts)}\n\n【问题】\n{query}"
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def generate(query: str, contexts: list[RetrievedChunk]) -> Answer:
    """Generate a cited answer based on the retrieved context.

    Args:
        query: user question
        contexts: contexts after retrieval / reranking

    Returns:
        Answer: answer text + list of sources
    """
    if not contexts:  # no retrieval results → refuse to answer rather than let the model hallucinate (Chapter 9)
        return Answer(text="根据现有资料无法确定。没有检索到与该问题相关的内容。", sources=[])

    from openai import OpenAI  # lazy import

    client = OpenAI(api_key=config.DEEPSEEK_API_KEY, base_url=config.DEEPSEEK_BASE_URL)
    resp = client.chat.completions.create(
        model=config.LLM_MODEL,
        temperature=config.LLM_TEMPERATURE,
        messages=build_messages(query, contexts),
    )
    text = resp.choices[0].message.content or ""
    sources = sorted({rc.chunk.source for rc in contexts})
    return Answer(text=text, sources=sources)
```

### PipelineRAG

Main pipeline — chains loader → chunker → embedder → vectorstore → retriever → reranker → query_processor → generator  
into a complete flow.

This is the "assembly workshop" of the whole project: each module above is a part; here they are assembled into  
a machine that turns "a question" into "an answer with provenance".

`scripts/build_index.py` calls `build_index()` for offline indexing; `scripts/ask.py` calls `ask()` for online Q&A.

```python title="pipeline.py"
from pathlib import Path

from . import config
from .chunker import chunk_doc
from .embedder import Embedder
from .generator import Answer, generate
from .loader import load_dir
from .query_processor import process_query
from .reranker import Reranker
from .retriever import RetrievedChunk, Retriever
from .vectorstore import VectorStore


class RAGPipeline:
    """End-to-end RAG flow. Build the index once, then ask repeatedly."""

    def __init__(self, persist_dir: Path | None = None) -> None:
        """Assemble the modules (connect the vector store; models are lazily loaded on demand).

        Args:
            persist_dir: vector store persistence directory (defaults to config.CHROMA_DIR)
        """
        self.persist_dir = Path(persist_dir or config.CHROMA_DIR)
        self.embedder = Embedder(config.EMBEDDING_MODEL)
        self.store = VectorStore(self.persist_dir)
        self.reranker = Reranker(config.RERANKER_MODEL) if config.USE_RERANK else None
        self._refresh_retriever()

    def _refresh_retriever(self) -> None:
        """(Re)load all chunks from the vector store and rebuild the retriever (BM25 path needs full text)."""
        chunks = self.store.all_chunks()
        self.retriever = Retriever(self.store, self.embedder, chunks)

    def build_index(self, docs_dir: Path | None = None) -> int:
        """Offline indexing: load → chunk → vectorize → store.

        Args:
            docs_dir: document directory (defaults to config.DATA_DIR)

        Returns:
            total number of chunks added to the index
        """
        docs = load_dir(Path(docs_dir or config.DATA_DIR))
        all_chunks = []
        for doc in docs:
            all_chunks.extend(chunk_doc(doc, config.CHUNK_SIZE, config.CHUNK_OVERLAP))
        if all_chunks:
            embeddings = self.embedder.encode([c.text for c in all_chunks])
            self.store.add(all_chunks, embeddings)
        self._refresh_retriever()  # make newly added chunks immediately visible to the BM25 path
        return len(all_chunks)

    def ask(self, query: str) -> Answer:
        """Online Q&A: query rewrite → dual-path retrieval (including expansions) → rerank → cited generation.

        Args:
            query: user question

        Returns:
            Answer with citations
        """
        pq = process_query(query)

        # Use multiple queries (rewritten + expansions) to recall separately, deduplicate by chunk, merge into candidate pool
        pool: dict[str, RetrievedChunk] = {}
        for q in pq.all_queries()[:3]:
            for rc in self.retriever.retrieve(q, top_k=config.RETRIEVE_TOP_N, top_n=config.RETRIEVE_TOP_N):
                key = f"{rc.chunk.source}#{rc.chunk.chunk_id}"
                if key not in pool or rc.score > pool[key].score:
                    pool[key] = rc
        candidates = sorted(pool.values(), key=lambda rc: rc.score, reverse=True)

        # Rerank using the original question for precise scoring; if reranking is disabled, take top K directly
        if self.reranker is not None and candidates:
            contexts = self.reranker.rerank(query, candidates, top_k=config.RERANK_TOP_K)
        else:
            contexts = candidates[: config.RERANK_TOP_K]

        return generate(query, contexts)
```

## Take-away Msgs

1. **Modular separation** – Each stage (loading, chunking, embedding, retrieval, reranking, generation) is a self-contained module with clear interfaces, making the pipeline easy to understand, test, and extend.
2. **Lazy loading** – Heavy dependencies (models, libraries) are imported and instantiated only when actually needed, so the system can boot quickly and run even when optional components are missing.
3. **Centralized configuration** – All tunable parameters, paths, and API keys live in one place (`config.py`), loaded from environment variables / `.env` for security and portability.
4. **Structural chunking matters** – Respecting document structure (headings, clauses, sentences) and using overlap avoids cutting critical information in half, directly improving retrieval quality.
5. **Hybrid retrieval wins** – Combining dense vector search (semantic similarity) with BM25 (exact keyword match) via RRF gives robust recall across different query styles, especially for Chinese documents.
6. **Query rewriting boosts recall** – Normalizing and expanding colloquial or vague user questions (via LLM or rule-based fallback) helps bridge the gap between query wording and document wording.
7. **Reranking refines results** – A cross-encoder reranker applied to the candidate pool can significantly improve precision, ensuring only the most relevant chunks reach the generator.
8. **Citation and refusal guard against hallucination** – The generator is explicitly instructed to answer only from retrieved context, cite sources, and refuse when insufficient information is available.
9. **Graceful degradation** – Each optional component (reranker, query rewrite, LLM key) has a safe fallback, so the pipeline remains functional even under partial configuration or missing dependencies.
10. **Pipeline assembly is the final step** – The `RAGPipeline` class wires all modules together, providing a clean `build_index()` for offline setup and `ask()` for online Q&A, making the whole system reproducible and production-ready.