"""Semantic NLP analysis for understanding documents and extracting relevant content.

Uses structure-aware TF-IDF with heading/content type weightaging for accurate
semantic search and ranking.
"""
import re
import math
from typing import Dict, List, Tuple, Optional, Set
from collections import Counter


# ── Enhanced tokenization with synonym expansion ───────────────────────────────

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


def _expand_keywords(tokens: List[str]) -> Set[str]:
    """Expand tokens with stemming and common variants for better matching."""
    expanded = set(tokens)
    
    # Domain-specific synonyms for better matching
    synonyms = {
        "login": {"authorize", "authenticate", "signin", "sign-in"},
        "database": {"db", "data"},
        "configuration": {"config", "setup"},
        "execute": {"run", "perform"},
        "verify": {"check", "validate", "confirm"},
        "delete": {"remove", "purge"},
        "create": {"add", "new", "generate"},
        "update": {"modify", "change", "edit"},
        "retrieve": {"fetch", "get", "obtain"},
        "transmit": {"send", "transfer"},
        "commissioning": {"commission", "setup", "configure", "initialize"},
        "error": {"exception", "failure", "issue"},
        "success": {"pass", "complete", "ok"},
    }
    
    for token in list(tokens):
        # Add token variants
        if token in synonyms:
            expanded.update(synonyms[token])
        
        # Stemming variants
        if token.endswith("ing") and len(token) > 5:
            expanded.add(token[:-3])  # "commissioning" -> "commission"
        if token.endswith("ed") and len(token) > 4:
            expanded.add(token[:-2])  # "configured" -> "configure"
        if token.endswith("s") and len(token) > 4 and not token.endswith("ss"):
            expanded.add(token[:-1])  # "tests" -> "test"
        if token.endswith("tion"):
            expanded.add(token[:-4])  # "configuration" -> "configure"
    
    return expanded


class _TfIdf:
    """Enhanced TF-IDF engine with structure-aware weighting.
    
    Applies different weightages based on content type:
    - H1/Title headings: 3.0x
    - H2/Section headings: 2.5x
    - H3/Subsection headings: 2.0x
    - Bold/Emphasized: 1.5x
    - Body text: 1.0x
    """

    def __init__(self):
        self.idf: Dict[str, float] = {}
        self.doc_vectors: List[Dict[str, float]] = []
        self.doc_weights: List[float] = []  # Content-type weightages
        self.n_docs = 0

    def fit(self, documents: List[str], weights: List[float] = None):
        """Build vocabulary and IDF from a list of text chunks with optional weights.
        
        Args:
            documents: List of text chunks
            weights: Optional list of weights per document (default 1.0 for all)
        """
        if weights is None:
            weights = [1.0] * len(documents)
        
        self.doc_weights = weights
        tokenised = [_tokenize(d) for d in documents]
        self.n_docs = len(tokenised)
        
        # Build IDF across all documents
        df: Counter = Counter()
        for tokens in tokenised:
            expanded = _expand_keywords(tokens)
            for t in expanded:
                df[t] += 1
        
        self.idf = {
            term: math.log((self.n_docs + 1) / (count + 1)) + 1
            for term, count in df.items()
        }
        
        # Create weighted TF-IDF vectors
        self.doc_vectors = [
            self._tfidf_vec(tokenised[i], weights[i])
            for i in range(len(documents))
        ]

    def query(self, text: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """Return (index, score) pairs ranked by weighted cosine similarity to *text*."""
        q_vec = self._tfidf_vec(_tokenize(text), weight=1.0)
        scores: List[Tuple[int, float]] = []
        
        for idx, d_vec in enumerate(self.doc_vectors):
            s = self._cosine(q_vec, d_vec)
            if s > 0:
                scores.append((idx, s))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def _tfidf_vec(self, tokens: List[str], weight: float = 1.0) -> Dict[str, float]:
        """Create TF-IDF vector with content-type weighting applied."""
        expanded = _expand_keywords(tokens)
        tf = Counter(t for tokens_list in [tokens] for t in tokens_list)  # Original TF
        
        total = len(tokens) or 1
        vector = {}
        
        for t in expanded:
            # Use expanded keywords for matching but weight by expanded token count
            tf_score = (tf.get(t, 0) + 1) / total  # Smoothing
            idf_score = self.idf.get(t, 1.0)
            vector[t] = (tf_score * idf_score) * weight
        
        return vector

    @staticmethod
    def _cosine(a: Dict[str, float], b: Dict[str, float]) -> float:
        """Compute cosine similarity between two TF-IDF vectors."""
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

    # ── smart chunking with structure awareness ───────────────────────────────

    def _smart_chunk(self, content: str) -> List[Dict]:
        """Split content into semantic chunks, preserving structure and content type.

        Priority: heading-based → paragraph → sliding-window sentences.
        Each chunk includes metadata about its type and hierarchy level.
        """
        chunks: List[Dict] = []

        # 1. Markdown headings (#, ##, etc.) - with level detection
        md_headings = list(re.finditer(r"(?:^|\n)(#{1,4})\s+(.+)", content))
        if len(md_headings) >= 2:
            for i, m in enumerate(md_headings):
                start = m.end()
                end = md_headings[i + 1].start() if i + 1 < len(md_headings) else len(content)
                title = m.group(2).strip()
                body = content[start:end].strip()
                heading_level = len(m.group(1))  # Number of # symbols
                
                if title and (len(body) > 15 or len(title) > 10):
                    # Calculate weight based on heading level
                    # H1 (#): 3.0x, H2 (##): 2.5x, H3 (###): 2.0x, H4 (####): 1.5x
                    weights = {1: 3.0, 2: 2.5, 3: 2.0, 4: 1.5}
                    weight = weights.get(heading_level, 1.5)
                    
                    chunks.append({
                        "title": title,
                        "body": body,
                        "full": f"{title}. {body}",
                        "type": "markdown_heading",
                        "level": heading_level,
                        "weight": weight
                    })
            if chunks:
                return self._merge_small_chunks(chunks)

        # 1b. Numbered top-level headings with section detection
        num_headings = list(re.finditer(
            r"(?:^|\n)(\d+(?:\.\d+)*[\.\)])\s+([A-Z][^\n]{5,})", content
        ))
        if len(num_headings) >= 2:
            for i, m in enumerate(num_headings):
                start = m.end()
                end = num_headings[i + 1].start() if i + 1 < len(num_headings) else len(content)
                title = m.group(2).strip()
                body = content[start:end].strip()
                
                # Detect section hierarchy depth from numbering
                num_parts = len(m.group(1).split("."))
                weight = {1: 3.0, 2: 2.5, 3: 2.0}.get(num_parts, 1.5)
                
                if title and (len(body) > 15 or len(title) > 10):
                    chunks.append({
                        "title": title,
                        "body": body,
                        "full": f"{title}. {body}",
                        "type": "numbered_heading",
                        "level": num_parts,
                        "weight": weight
                    })
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
                chunks.append({
                    "title": title,
                    "body": para,
                    "full": para,
                    "type": "paragraph",
                    "level": 0,
                    "weight": 1.0
                })
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
                chunks.append({
                    "title": title,
                    "body": item,
                    "full": item,
                    "type": "bullet",
                    "level": 0,
                    "weight": 1.2  # Slightly higher than paragraph
                })
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
            chunks.append({
                "title": title,
                "body": combined,
                "full": combined,
                "type": "sentence_group",
                "level": 0,
                "weight": 1.0
            })
            i += window - overlap

        if not chunks:
            chunks.append({
                "title": self._first_sentence(content),
                "body": content.strip(),
                "full": content.strip(),
                "type": "full_document",
                "level": 0,
                "weight": 1.0
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
                # Preserve the higher weight
                if "weight" in c and "weight" in buf:
                    buf["weight"] = max(buf.get("weight", 1.0), c.get("weight", 1.0))
            else:
                merged.append(buf)
                buf = dict(c)
        if buf:
            merged.append(buf)
        # Ensure all chunks have default weights
        for chunk in merged if merged else chunks:
            if "weight" not in chunk:
                chunk["weight"] = 1.0
            if "type" not in chunk:
                chunk["type"] = "unknown"
            if "level" not in chunk:
                chunk["level"] = 0
        return merged if merged else chunks

    # ── semantic search with structure awareness ──────────────────────────────

    def _semantic_search(
        self, chunks: List[Dict], prompt: str
    ) -> List[Tuple[Dict, float]]:
        """Rank chunks by structure-aware semantic similarity to *prompt*.
        
        Uses weighted TF-IDF that prioritizes:
        - Headings/titles (higher weights)
        - Content type (heading vs body)
        - Heading depth level
        """
        if not chunks:
            return []

        # Ensure all chunks have weight metadata
        for chunk in chunks:
            if "weight" not in chunk:
                chunk["weight"] = 1.0
            if "type" not in chunk:
                chunk["type"] = "unknown"

        corpus = [c["full"] for c in chunks]
        weights = [c.get("weight", 1.0) for c in chunks]
        
        engine = _TfIdf()
        engine.fit(corpus, weights=weights)
        results = engine.query(prompt, top_k=len(chunks))

        # Enhanced keyword extraction and expansion
        prompt_keywords = set(_tokenize(prompt))
        expanded_prompt_keywords = _expand_keywords(list(prompt_keywords))
        
        # Technical terms (longer words, hyphenated, containing digits)
        technical = {
            w for w in expanded_prompt_keywords
            if len(w) > 4 or "-" in w or any(ch.isdigit() for ch in w)
        }

        ranked: List[Tuple[Dict, float]] = []
        seen = set()
        
        # Process TF-IDF results with additional bonuses
        for idx, score in results:
            chunk_lower = chunks[idx]["full"].lower()
            
            # Technical keyword bonus - more generous than before (0.1 -> 0.15)
            technical_bonus = sum(0.15 for kw in technical if kw in chunk_lower)
            
            # Structure bonus based on chunk type
            struct_bonus = 0.0
            chunk_type = chunks[idx].get("type", "unknown")
            if chunk_type in ["markdown_heading", "numbered_heading"]:
                # Additional bonus for matching keywords in headings
                heading_kw_matches = sum(1 for kw in prompt_keywords if kw in chunk_lower)
                struct_bonus = 0.2 * heading_kw_matches  # Higher bonus for heading matches
            
            final_score = min(score + technical_bonus + struct_bonus, 1.0)
            ranked.append((chunks[idx], final_score))
            seen.add(idx)

        # Include missed chunks that have strong keyword overlap
        for idx, chunk in enumerate(chunks):
            if idx in seen:
                continue
            chunk_lower = chunk["full"].lower()
            kw_hits = sum(1 for kw in expanded_prompt_keywords if kw in chunk_lower)
            
            # Lower threshold with bonus for keyword matches in headings
            if kw_hits >= 2:
                base_score = kw_hits * 0.08
                if chunk.get("type") in ["markdown_heading", "numbered_heading"]:
                    base_score *= 1.5  # Boost for heading matches
                ranked.append((chunk, min(base_score, 0.5)))

        ranked.sort(key=lambda x: x[1], reverse=True)

        # Improved threshold logic - be more inclusive
        if ranked:
            max_score = ranked[0][1]
            # Lower threshold from 0.55 to 0.40 to include more relevant results
            threshold = max(max_score * 0.40, 0.06)
            filtered = [(c, s) for c, s in ranked if s >= threshold]
            if not filtered:
                # If filtering is too strict, include at least top 3
                filtered = ranked[:max(3, len(ranked) // 3)]
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
