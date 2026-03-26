"""Semantic NLP analysis for understanding documents and extracting relevant content.

Uses TF-IDF vectorization and cosine similarity to semantically understand
document content and find sections relevant to a user's query.
"""
import re
import math
from typing import Dict, List, Tuple, Optional
from collections import Counter


# ── Lightweight TF-IDF implementation ─────────────────────────────────────────

def _tokenize(text: str) -> List[str]:
    """Lowercase tokenisation with stop-word removal."""
    STOP = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "shall",
        "should", "may", "might", "can", "could", "of", "in", "to", "for",
        "with", "on", "at", "by", "from", "as", "into", "about", "between",
        "through", "during", "before", "after", "above", "below", "up", "down",
        "out", "off", "over", "under", "again", "further", "then", "once",
        "and", "but", "or", "nor", "not", "so", "yet", "both", "either",
        "neither", "each", "every", "all", "any", "few", "more", "most",
        "other", "some", "such", "no", "only", "own", "same", "than", "too",
        "very", "just", "because", "if", "when", "while", "where", "how",
        "what", "which", "who", "whom", "this", "that", "these", "those",
        "it", "its", "i", "me", "my", "we", "our", "you", "your", "he",
        "him", "his", "she", "her", "they", "them", "their",
        "generate", "generated", "generating", "test", "tests", "testing",
        "testcase", "testcases", "case", "cases", "prompt",
    }
    words = re.findall(r"[a-z][a-z0-9\-]{1,}", text.lower())
    return [w for w in words if w not in STOP]


class _TfIdf:
    """Minimal TF-IDF engine operating on a corpus of text chunks."""

    def __init__(self):
        self.idf: Dict[str, float] = {}
        self.doc_vectors: List[Dict[str, float]] = []
        self.n_docs = 0

    def fit(self, documents: List[str]):
        """Build vocabulary and IDF from a list of text chunks."""
        tokenised = [_tokenize(d) for d in documents]
        self.n_docs = len(tokenised)
        df: Counter = Counter()
        for tokens in tokenised:
            for t in set(tokens):
                df[t] += 1
        self.idf = {
            term: math.log((self.n_docs + 1) / (count + 1)) + 1
            for term, count in df.items()
        }
        self.doc_vectors = [self._tfidf_vec(tokens) for tokens in tokenised]

    def query(self, text: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """Return (index, score) pairs ranked by cosine similarity to *text*."""
        q_vec = self._tfidf_vec(_tokenize(text))
        scores: List[Tuple[int, float]] = []
        for idx, d_vec in enumerate(self.doc_vectors):
            s = self._cosine(q_vec, d_vec)
            if s > 0:
                scores.append((idx, s))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def _tfidf_vec(self, tokens: List[str]) -> Dict[str, float]:
        tf = Counter(tokens)
        total = len(tokens) or 1
        return {t: (c / total) * self.idf.get(t, 1.0) for t, c in tf.items()}

    @staticmethod
    def _cosine(a: Dict[str, float], b: Dict[str, float]) -> float:
        common = set(a) & set(b)
        if not common:
            return 0.0
        dot = sum(a[k] * b[k] for k in common)
        mag_a = math.sqrt(sum(v * v for v in a.values()))
        mag_b = math.sqrt(sum(v * v for v in b.values()))
        if mag_a == 0 or mag_b == 0:
            return 0.0
        return dot / (mag_a * mag_b)


# ── Main analyser ─────────────────────────────────────────────────────────────

class NLPAnalyzer:
    """Semantic document analyser.

    Builds a TF-IDF index over document chunks so that a user prompt can
    semantically locate the most relevant parts of a large document.
    """

    # ── public API ────────────────────────────────────────────────────────────

    def analyze(self, document_content: str, user_prompt: Optional[str] = None) -> Dict:
        """Analyse a document and optionally filter by user prompt.

        Returns dict with:
            raw_content, chunks, relevant_chunks, product_name, domain
        """
        chunks = self._smart_chunk(document_content)

        if user_prompt and user_prompt.strip():
            relevant = self._semantic_search(chunks, user_prompt)
        else:
            relevant = [(c, 1.0) for c in chunks]

        return {
            "raw_content": document_content,
            "chunks": chunks,
            "relevant_chunks": relevant,
            "product_name": self._extract_product_name(document_content),
            "domain": self._detect_domain(document_content),
        }

    # ── smart chunking ────────────────────────────────────────────────────────

    def _smart_chunk(self, content: str) -> List[Dict]:
        """Split content into semantic chunks, preserving context.

        Priority: heading-based → paragraph → sliding-window sentences.
        """
        chunks: List[Dict] = []

        # 1. Markdown headings (#, ##, etc.)
        md_headings = list(re.finditer(r"(?:^|\n)(#{1,4})\s+(.+)", content))
        if len(md_headings) >= 2:
            for i, m in enumerate(md_headings):
                start = m.end()
                end = md_headings[i + 1].start() if i + 1 < len(md_headings) else len(content)
                title = m.group(2).strip()
                body = content[start:end].strip()
                if title and (len(body) > 15 or len(title) > 10):
                    chunks.append({"title": title, "body": body, "full": f"{title}. {body}"})
            if chunks:
                return self._merge_small_chunks(chunks)

        # 1b. Numbered top-level headings (only matches lines that look like
        #     section headings, not numbered list items inside a section body)
        num_headings = list(re.finditer(
            r"(?:^|\n)(\d+(?:\.\d+)*[\.\)])\s+([A-Z][^\n]{5,})", content
        ))
        if len(num_headings) >= 2:
            for i, m in enumerate(num_headings):
                start = m.end()
                end = num_headings[i + 1].start() if i + 1 < len(num_headings) else len(content)
                title = m.group(2).strip()
                body = content[start:end].strip()
                if title and (len(body) > 15 or len(title) > 10):
                    chunks.append({"title": title, "body": body, "full": f"{title}. {body}"})
            if chunks:
                return self._merge_small_chunks(chunks)

        # 2. Paragraph separation
        paragraphs = re.split(r"\n\s*\n", content.strip())
        if len(paragraphs) >= 3:
            for para in paragraphs:
                para = para.strip()
                if len(para) < 15:
                    continue
                title = self._first_sentence(para)
                chunks.append({"title": title, "body": para, "full": para})
            if len(chunks) >= 2:
                return self._merge_small_chunks(chunks)

        # 3. Bullet/dash items
        bullet_items = re.split(r"\n\s*[-•*]\s+", content.strip())
        if len(bullet_items) >= 3:
            for item in bullet_items:
                item = item.strip()
                if len(item) < 10:
                    continue
                title = self._first_sentence(item)
                chunks.append({"title": title, "body": item, "full": item})
            if len(chunks) >= 2:
                return self._merge_small_chunks(chunks)

        # 4. Sliding-window sentences
        sentences = self._split_sentences(content)
        window, overlap = 4, 1
        i = 0
        while i < len(sentences):
            group = sentences[i : i + window]
            combined = " ".join(group)
            title = self._first_sentence(combined)
            chunks.append({"title": title, "body": combined, "full": combined})
            i += window - overlap

        if not chunks:
            chunks.append({
                "title": self._first_sentence(content),
                "body": content.strip(),
                "full": content.strip(),
            })

        return chunks

    def _merge_small_chunks(self, chunks: List[Dict], min_len: int = 60) -> List[Dict]:
        """Merge short consecutive chunks to ensure adequate context."""
        merged: List[Dict] = []
        buf: Optional[Dict] = None
        for c in chunks:
            if buf is None:
                buf = dict(c)
                continue
            if len(buf["full"]) < min_len:
                buf["body"] = buf["body"] + "\n" + c["body"]
                buf["full"] = buf["full"] + " " + c["full"]
            else:
                merged.append(buf)
                buf = dict(c)
        if buf:
            merged.append(buf)
        return merged if merged else chunks

    # ── semantic search ───────────────────────────────────────────────────────

    def _semantic_search(
        self, chunks: List[Dict], prompt: str
    ) -> List[Tuple[Dict, float]]:
        """Rank chunks by semantic similarity to *prompt* using TF-IDF."""
        if not chunks:
            return []

        corpus = [c["full"] for c in chunks]
        engine = _TfIdf()
        engine.fit(corpus)
        results = engine.query(prompt, top_k=len(chunks))

        # Exact-keyword boosting for technical terms
        prompt_keywords = set(_tokenize(prompt))
        expanded_prompt_keywords = set(prompt_keywords)
        for kw in list(prompt_keywords):
            # Lightweight stemming helps match prompt words like
            # "commission" <-> "commissioning" and plural variants.
            if kw.endswith("ing") and len(kw) > 5:
                expanded_prompt_keywords.add(kw[:-3])
            if kw.endswith("ed") and len(kw) > 4:
                expanded_prompt_keywords.add(kw[:-2])
            if kw.endswith("s") and len(kw) > 4:
                expanded_prompt_keywords.add(kw[:-1])
        technical = {
            w for w in expanded_prompt_keywords
            if len(w) > 4 or "-" in w or any(ch.isdigit() for ch in w)
        }

        ranked: List[Tuple[Dict, float]] = []
        seen = set()
        for idx, score in results:
            chunk_lower = chunks[idx]["full"].lower()
            bonus = sum(0.1 for kw in technical if kw in chunk_lower)
            ranked.append((chunks[idx], min(score + bonus, 1.0)))
            seen.add(idx)

        # Include missed chunks that have keyword overlap
        for idx, chunk in enumerate(chunks):
            if idx in seen:
                continue
            chunk_lower = chunk["full"].lower()
            kw_hits = sum(1 for kw in expanded_prompt_keywords if kw in chunk_lower)
            if kw_hits >= 2:
                ranked.append((chunk, kw_hits * 0.05))

        ranked.sort(key=lambda x: x[1], reverse=True)

        # Keep only strongly relevant chunks when prompt exists.
        # This prevents broad documents from returning all sections.
        if ranked:
            max_score = ranked[0][1]
            threshold = max(max_score * 0.55, 0.08)
            filtered = [(c, s) for c, s in ranked if s >= threshold]
            if not filtered:
                filtered = ranked[:1]
            return filtered

        return ranked

    # ── utility helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        parts = re.split(r"(?<=[.!?])\s+(?=[A-Z])|\n+", text)
        return [s.strip() for s in parts if s.strip() and len(s.strip()) > 8]

    @staticmethod
    def _first_sentence(text: str) -> str:
        text = text.strip()
        m = re.match(r"(.+?[.!?])(?:\s|$)", text)
        if m and len(m.group(1)) <= 100:
            return m.group(1).strip()
        first_line = text.split("\n")[0].strip()
        if len(first_line) <= 80:
            return first_line
        return first_line[:80].rsplit(" ", 1)[0]

    @staticmethod
    def _extract_product_name(content: str) -> str:
        patterns = [
            r"(?:product|system|device|server|module|unit)\s*(?:name)?\s*[:\-]\s*(.+?)(?:\n|$)",
            r"(Panel\s*Server\s*(?:EPC|EGX)?)",
            r"(EcoStruxure\s+[\w\s]{3,30})",
            r"(PowerTag\s*[\w\d]*)",
        ]
        for pattern in patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                return (match.group(1) if match.lastindex else match.group(0)).strip()
        return ""

    @staticmethod
    def _detect_domain(content: str) -> str:
        content_lower = content.lower()
        domain_keywords = {
            "commissioning": [
                "commissioning", "commission", "setup", "configure", "install",
                "firmware", "panel server", "modbus", "rs-485", "powertag",
            ],
            "api_testing": ["api", "endpoint", "rest", "http", "request", "response", "json"],
            "security": [
                "security", "authentication", "authorization", "login",
                "password", "token", "encryption",
            ],
            "performance": ["performance", "load test", "stress test", "throughput", "latency"],
            "integration": ["integration", "third-party", "external service", "webhook"],
            "ui_testing": ["user interface", "button", "form", "display", "navigation"],
            "network": ["network", "tcp", "ethernet", "wireless", "wifi", "bluetooth"],
            "firmware": ["firmware", "update", "upgrade", "version", "flash"],
        }
        scores = {
            d: sum(content_lower.count(kw) for kw in kws)
            for d, kws in domain_keywords.items()
        }
        scores = {d: s for d, s in scores.items() if s > 0}
        return max(scores, key=scores.get) if scores else "general"
