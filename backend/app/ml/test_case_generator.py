"""AI Test Case Generator — Semantic version.

Generates test cases that genuinely reflect the document content.
Instead of slot-filling templates, it reads the actual text of each
relevant chunk, extracts real actions / conditions / expectations
described in the document, and composes test cases from them.
"""
import re
from typing import List, Optional, Dict, Tuple
from app.models.schemas import TestCase, DetailLevelEnum
from app.ml.nlp_analyzer import NLPAnalyzer
from app.ml.custom_trainer import CustomTrainer


class TestCaseGenerator:
    """Generate document-aware test cases using semantic analysis."""

    def __init__(self):
        self.nlp_analyzer = NLPAnalyzer()
        self.custom_trainer = CustomTrainer()

    # ── public entry point ────────────────────────────────────────────────────

    def generate(
        self,
        document_content: str,
        user_prompt: Optional[str] = None,
        test_types: Optional[List[str]] = None,
        detail_level: DetailLevelEnum = DetailLevelEnum.DETAILED,
        num_test_cases: Optional[int] = None,
        id_prefix: str = "ATC",
    ) -> List[TestCase]:
        """Generate test cases from *document_content*.

        1. Semantically chunk the document.
        2. If a user_prompt is given, rank chunks by relevance and keep top ones.
        3. For each relevant chunk, build a test case whose steps / results /
           preconditions are derived from the actual text (not templates).
        """
        analysis = self.nlp_analyzer.analyze(document_content, user_prompt)
        relevant_chunks: List[Tuple[Dict, float]] = analysis.get("relevant_chunks", [])
        domain = analysis.get("domain", "general")
        product_name = analysis.get("product_name", "")
        learned_context_words = set()
        trainer_boost_terms = set()

        if user_prompt and user_prompt.strip():
            learned_context_words = self._learned_terms_for_prompt(user_prompt)
            trainer_boost_terms = set(self.custom_trainer.get_prompt_boost_terms(user_prompt))
            relevant_chunks = self._rerank_relevant_chunks(
                relevant_chunks,
                user_prompt,
                learned_context_words,
                trainer_boost_terms,
            )
            relevant_chunks = [
                (chunk, score)
                for chunk, score in relevant_chunks
                if self._is_prompt_relevant(chunk, user_prompt, score, learned_context_words)
            ]

        extracted_keywords = self.custom_trainer.extract_keywords(document_content)
        recommendations = self.custom_trainer.get_recommendations(extracted_keywords)
        default_test_type = self._determine_test_type(domain, test_types, recommendations)

        test_cases: List[TestCase] = []
        base_id = 1

        for idx, (chunk, relevance) in enumerate(relevant_chunks):
            if num_test_cases and len(test_cases) >= num_test_cases:
                break

            tc = self._build_test_case(
                chunk=chunk,
                relevance=relevance,
                index=idx,
                base_id=base_id,
                id_prefix=id_prefix,
                test_type=default_test_type,
                detail_level=detail_level,
                product_name=product_name,
                user_prompt=user_prompt,
            )
            if tc:
                test_cases.append(tc)

        if num_test_cases:
            test_cases = test_cases[:num_test_cases]

        return test_cases

    # ── core builder ──────────────────────────────────────────────────────────

    def _build_test_case(
        self,
        chunk: Dict,
        relevance: float,
        index: int,
        base_id: int,
        id_prefix: str,
        test_type: str,
        detail_level: DetailLevelEnum,
        product_name: str,
        user_prompt: Optional[str],
    ) -> Optional[TestCase]:
        """Build a single test case from a document chunk."""
        title_raw = self._normalize_text(chunk.get("title", "").strip())
        body = self._normalize_text(chunk.get("body", "").strip())
        full = self._normalize_text(chunk.get("full", body))

        if not title_raw or len(title_raw) < 4:
            return None

        title = self._clean_title(title_raw)
        title = self._pick_prompt_focused_title(title, body, user_prompt)
        tc_id = (
            f"{product_name}-{id_prefix}-{base_id + index:03d}"
            if product_name
            else f"{id_prefix}-{base_id + index:03d}"
        )

        # ── Extract information directly from the chunk text ──
        description = self._build_description(title, body, user_prompt)
        steps = self._extract_steps_from_text(body, full)
        expected_results = self._extract_expected_results(body, full, steps)
        preconditions = self._extract_preconditions(body, full)
        objective = self._build_objective(title, description, user_prompt)
        priority = self._determine_priority(full, relevance)
        final_result = self._build_final_result(title, expected_results)
        acceptance_criteria = self._build_acceptance_criteria(expected_results, final_result)

        # Apply detail level
        steps, expected_results, preconditions = self._apply_detail_level(
            steps, expected_results, preconditions, detail_level
        )

        return TestCase(
            id=tc_id,
            test_type=test_type,
            title=title,
            method="Manual Testing",
            description=description,
            objective=objective,
            preconditions=preconditions,
            steps=steps,
            expected_results=expected_results,
            priority=priority,
            acceptance_criteria=acceptance_criteria,
            final_result=final_result,
        )

    # ── description ───────────────────────────────────────────────────────────

    def _build_description(
        self, title: str, body: str, user_prompt: Optional[str]
    ) -> str:
        """Build description from actual document text."""
        # Take first 2-3 meaningful sentences from body
        sentences = self._get_sentences(body)
        desc_sentences = []
        for s in sentences:
            # Skip very short or heading-like lines
            if len(s) < 15:
                continue
            s_lower = s.lower()
            if any(marker in s_lower for marker in [
                "for more information",
                "for more detailed information",
                "refer to",
                "page ",
                "documentation",
                "download link",
            ]):
                continue
            # Don't repeat the title
            if s.lower().strip(". ") == title.lower().strip(". "):
                continue
            desc_sentences.append(s)
            if len(desc_sentences) >= 3:
                break

        if desc_sentences:
            desc = " ".join(desc_sentences)
        else:
            desc = body[:300] if len(body) > 20 else title

        desc = self._normalize_text(desc)
        return desc[:500]

    # ── steps extraction ──────────────────────────────────────────────────────

    def _extract_steps_from_text(self, body: str, full: str) -> List[str]:
        """Extract test steps from the actual document content.

        Strategy:
        1. Look for explicit numbered / bulleted lists (these are the real steps)
        2. Look for imperative sentences (verbs at start)
        3. Look for conditional / procedural sentences ("when", "if", "after")
        4. Synthesise verification steps from descriptive statements
        """
        steps: List[str] = []
        seen_lower: set = set()

        def _add(step_text: str):
            s = self._normalize_text(step_text.strip().rstrip("."))
            if len(s) < 8:
                return
            key = s.lower()
            if key not in seen_lower:
                seen_lower.add(key)
                steps.append(s[0].upper() + s[1:] if s else s)

        # 1. Explicit list items (numbered or bulleted) — line by line
        for line in full.split("\n"):
            m = re.match(r"\s*(?:[-•*►]|\d+[\.\)])\s+(.+)", line)
            if m:
                cleaned = self._normalize_text(m.group(1).strip())
                if len(cleaned) > 8:
                    _add(cleaned)

        # 2. Imperative sentences (start with action verb)
        action_verbs = {
            "open", "navigate", "click", "select", "enter", "type", "verify",
            "check", "enable", "disable", "configure", "set", "save", "login",
            "log", "start", "stop", "create", "delete", "update", "assign",
            "connect", "disconnect", "test", "validate", "confirm", "apply",
            "install", "upload", "download", "send", "receive", "initiate",
            "wait", "reboot", "restart", "detect", "discover", "pair", "unpair",
            "add", "remove", "press", "tap", "swipe", "scroll", "turn",
            "power", "plug", "unplug", "insert", "scan", "charge", "reset",
            "replace", "attach", "detach", "close", "access", "ensure",
            "observe", "record", "note", "measure", "adjust", "switch",
        }
        for line in re.split(r"[.\n]+", full):
            line = self._normalize_text(line.strip().lstrip("•-*►0123456789.) "))
            if not line:
                continue
            first_word = line.split()[0].lower() if line.split() else ""
            if first_word in action_verbs and len(line) > 10:
                _add(line)

        # 3. Conditional / procedural sentences (only if not already captured)
        if len(steps) < 5:
            conditionals = re.findall(
                r"(?:^|\n)\s*((?:when|if|after|before|once)\s+.{15,}?)(?:\.|$)",
                full, re.IGNORECASE | re.MULTILINE,
            )
            for c in conditionals:
                _add(f"Verify condition: {self._normalize_text(c.strip())}")

        # 4. If still very few, synthesise from descriptive content
        if len(steps) < 3:
            sentences = self._get_sentences(body)
            for sent in sentences:
                if len(sent) < 15:
                    continue
                clean = self._normalize_text(sent.strip().rstrip("."))
                # Skip sentences that duplicate existing steps
                if clean.lower() in seen_lower:
                    continue
                # Convert specification-style to verification step
                if re.search(
                    r"\b(shall|should|must|supports?|allows?|provides?|displays?|shows?)\b",
                    sent, re.IGNORECASE,
                ):
                    _add(f"Verify that {clean[0].lower() + clean[1:]}")
                elif len(steps) < 5 and len(clean) > 20:
                    _add(f"Validate: {clean}")
                if len(steps) >= 8:
                    break

        return steps[:12]

    # ── expected results ──────────────────────────────────────────────────────

    def _extract_expected_results(
        self, body: str, full: str, steps: List[str]
    ) -> List[str]:
        """Derive expected results from document text and steps."""
        results: List[str] = []
        seen: set = set()

        def _add(r: str):
            r = self._normalize_text(r.strip())
            key = r.lower()
            if key not in seen and len(r) > 8:
                seen.add(key)
                results.append(r)

        # 1. Look for explicit expected results / acceptance criteria in text
        for marker in [
            r"expected\s*(?:result|outcome|behavior)",
            r"acceptance\s*criteria",
            r"(?:should|shall|must)\s+",
            r"result\s*:",
        ]:
            for m in re.finditer(marker, full, re.IGNORECASE):
                after = full[m.end() : m.end() + 200]
                first_sent = re.match(r"\s*:?\s*(.+?)(?:\.|$)", after)
                if first_sent:
                    _add(first_sent.group(1).strip())

        # 2. Extract "should/shall/must" assertions
        assertions = re.findall(
            r"(?:the\s+)?(?:\w+\s+){0,3}(?:should|shall|must|will)\s+(.{10,80}?)(?:\.|$)",
            full, re.IGNORECASE,
        )
        for a in assertions:
            _add(a.strip())

        # 3. Map results to steps — derive what success looks like per step
        if len(results) < len(steps):
            for step in steps:
                if len(results) >= 12:
                    break
                sl = step.lower()
                if "verify" in sl or "check" in sl:
                    core = re.sub(
                        r"^(?:verify:?\s*(?:that)?|check:?\s*)", "", step, flags=re.IGNORECASE
                    ).strip()
                    if core:
                        _add(f"{core[0].upper() + core[1:]} confirmed as expected")
                elif "power off" in sl or "disconnect" in sl:
                    _add("Device is powered off and disconnected")
                elif "remove" in sl:
                    _add("Component removed successfully")
                elif "insert" in sl or "replace" in sl:
                    _add("Component inserted/replaced correctly")
                elif "reconnect" in sl or "connect" in sl:
                    _add("Connection re-established successfully")
                elif "power on" in sl:
                    _add("Device powers on normally")
                elif "enable" in sl:
                    subject = re.sub(r"^enable\s+", "", step, flags=re.IGNORECASE).strip()
                    _add(f"{subject} is enabled and active")
                elif "disable" in sl:
                    subject = re.sub(r"^disable\s+", "", step, flags=re.IGNORECASE).strip()
                    _add(f"{subject} is disabled")
                elif "configure" in sl or "set " in sl or "adjust" in sl:
                    _add("Configuration applied correctly")
                elif "login" in sl or "log in" in sl:
                    _add("User logged in successfully")
                elif "navigate" in sl or "open" in sl or "access" in sl:
                    _add("Target page/screen is displayed")
                elif "save" in sl or "apply" in sl:
                    _add("Changes saved without errors")
                elif "ensure" in sl:
                    core = re.sub(r"^ensure\s+(?:that\s+)?", "", step, flags=re.IGNORECASE).strip()
                    _add(f"{core[0].upper() + core[1:]}" if core else "Condition confirmed")
                else:
                    # Build a concise expected result
                    _add(f"Step completed: {step[:55].rstrip()}")

                if len(results) >= 12:
                    break

        return [self._normalize_text(r) for r in results[:12]]

    # ── preconditions ─────────────────────────────────────────────────────────

    def _extract_preconditions(self, body: str, full: str) -> List[str]:
        """Extract preconditions from document text. Returns empty if none found."""
        preconds: List[str] = []
        seen: set = set()

        def _add(p: str):
            p = self._normalize_text(p.strip())
            key = p.lower()
            if key not in seen and len(p) > 8:
                seen.add(key)
                preconds.append(p)

        # Explicit precondition / prerequisite sections
        for marker in [
            r"pre-?conditions?\s*[:\-]",
            r"prerequisites?\s*[:\-]",
            r"requirements?\s*[:\-]",
            r"before\s+(?:you\s+)?(?:begin|start|proceed)",
        ]:
            m = re.search(marker, full, re.IGNORECASE)
            if m:
                after = full[m.end() : m.end() + 500]
                items = re.findall(
                    r"(?:^|\n)\s*(?:[-•*]|\d+[\.\)])\s+(.+?)(?=\n|$)", after
                )
                for item in items:
                    _add(item.strip())
                if not items:
                    # Take first few lines
                    for line in after.split("\n")[:3]:
                        line = line.strip()
                        if len(line) > 10:
                            _add(line)

        # Implicit preconditions from "ensure", "make sure", "required"
        implicit = re.findall(
            r"(?:ensure|make sure|required|necessary|needed)\s+(?:that\s+)?(.{10,80}?)(?:\.|$)",
            full, re.IGNORECASE,
        )
        for imp in implicit:
            _add(imp.strip())

        return preconds[:5]

    # ── objective ─────────────────────────────────────────────────────────────

    def _build_objective(
        self, title: str, description: str, user_prompt: Optional[str]
    ) -> str:
        if user_prompt:
            return f"Validate {title} according to the requested prompt focus"
        return f"Validate {title}"

    # ── priority ───────────────────────────────────────────────────────────────

    def _determine_priority(self, text: str, relevance: float) -> str:
        text_lower = text.lower()
        critical_kw = [
            "security", "authentication", "login", "password", "firmware",
            "critical", "safety", "emergency", "failure", "data loss",
        ]
        high_kw = [
            "configure", "network", "communication", "alarm", "update",
            "important", "required", "mandatory",
        ]
        if any(kw in text_lower for kw in critical_kw):
            return "Critical"
        if any(kw in text_lower for kw in high_kw):
            return "High"
        if relevance >= 0.6:
            return "High"
        if relevance >= 0.3:
            return "Medium"
        return "Low"

    # ── final result / acceptance criteria ─────────────────────────────────────

    def _build_final_result(self, title: str, expected_results: List[str]) -> str:
        if expected_results:
            return f"All verifications passed for: {title}"
        return f"Test completed for: {title}"

    def _build_acceptance_criteria(
        self, expected_results: List[str], final_result: str
    ) -> List[str]:
        criteria = list(expected_results[:4])
        if final_result:
            criteria.append(final_result)
        return criteria if criteria else ["Test case requirements met"]

    # ── test type ──────────────────────────────────────────────────────────────

    def _determine_test_type(
        self,
        domain: str,
        test_types: Optional[List[str]],
        recommendations: Optional[Dict] = None,
    ) -> str:
        if test_types and len(test_types) > 0:
            return test_types[0]
        if recommendations and recommendations.get("confidence", 0) >= 0.35:
            predicted = recommendations.get("test_types", {})
            if predicted:
                return max(predicted, key=predicted.get)
        domain_map = {
            "commissioning": "Commissioning",
            "api_testing": "API Testing",
            "security": "Security Testing",
            "performance": "Performance Testing",
            "integration": "Integration Testing",
            "ui_testing": "Usability Testing",
            "network": "Network Testing",
            "firmware": "Firmware Testing",
        }
        return domain_map.get(domain, "Functional Testing")

    def _rerank_relevant_chunks(
        self,
        relevant_chunks: List[Tuple[Dict, float]],
        prompt: str,
        learned_context_words: set,
        trainer_boost_terms: set,
    ) -> List[Tuple[Dict, float]]:
        """Apply trainer-informed reranking on top of semantic analyzer scores."""
        if not relevant_chunks:
            return []

        prompt_tokens = {
            t for t in re.findall(r"[a-z0-9\-]{3,}", prompt.lower())
            if t not in {"test", "tests", "testcase", "testcases", "generate", "case", "cases"}
        }

        rescored: List[Tuple[Dict, float]] = []
        for chunk, score in relevant_chunks:
            title = self._normalize_text(chunk.get("title", "")).lower()
            body = self._normalize_text(chunk.get("body", "")).lower()
            full = f"{title} {body}".strip()

            title_hits = sum(1 for t in prompt_tokens if t in title)
            body_hits = sum(1 for t in prompt_tokens if t in body)
            learned_hits = sum(1 for t in learned_context_words if t in full)
            trainer_hits = sum(1 for t in trainer_boost_terms if t in full)

            # Title/heading hits are more important than body hits.
            lexical_boost = (title_hits * 0.16) + (body_hits * 0.06)
            learned_boost = min(0.18, learned_hits * 0.01)
            trainer_boost = min(0.22, trainer_hits * 0.01)

            # Use chunk structural weight from NLP analyzer metadata.
            structural_weight = float(chunk.get("weight", 1.0))
            structural_boost = min((structural_weight - 1.0) * 0.08, 0.22)

            final_score = min(score + lexical_boost + learned_boost + trainer_boost + structural_boost, 1.0)
            rescored.append((chunk, final_score))

        rescored.sort(key=lambda x: x[1], reverse=True)
        return rescored

    # ── detail level ──────────────────────────────────────────────────────────

    @staticmethod
    def _apply_detail_level(steps, expected_results, preconditions, detail_level):
        if detail_level == DetailLevelEnum.BASIC:
            return steps[:3], expected_results[:2], preconditions[:2]
        elif detail_level == DetailLevelEnum.COMPREHENSIVE:
            return steps, expected_results, preconditions
        return steps[:8], expected_results[:6], preconditions[:5]

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _clean_title(title: str) -> str:
        title = re.sub(r"^[\s#*•\-\d\.\)]+", "", title).strip()
        title = re.sub(
            r"^(?:The\s+)?(?:system\s+)?(?:shall|should|must|will)\s+",
            "", title, flags=re.IGNORECASE,
        ).strip()
        if title:
            title = title[0].upper() + title[1:]
        if len(title) > 80:
            title = title[:80].rsplit(" ", 1)[0]
        return title

    @staticmethod
    def _get_sentences(text: str) -> List[str]:
        """Split text into sentences."""
        parts = re.split(r"(?<=[.!?])\s+(?=[A-Z])|\n+", text)
        return [s.strip() for s in parts if s.strip() and len(s.strip()) > 8]

    @staticmethod
    def _normalize_text(text: str) -> str:
        """Remove document artefacts that reduce output quality."""
        if not text:
            return ""
        cleaned = text
        cleaned = cleaned.replace("\u25e6", " ")
        cleaned = cleaned.replace("\u2013", "-")
        cleaned = cleaned.replace("\u2014", "-")
        cleaned = cleaned.replace("\u00ad", "")
        # Remove document reference codes and page artefacts.
        cleaned = re.sub(r"\bDOCA\d+[A-Z]*-\d+\b", " ", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\bpage\s+\d+\b", " ", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\b\d{1,3}\b(?=\s*$)", " ", cleaned)
        # Fix line-break hyphenation and whitespace noise.
        cleaned = re.sub(r"(\w)-\s+(\w)", r"\1\2", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    def _pick_prompt_focused_title(
        self,
        title: str,
        body: str,
        user_prompt: Optional[str],
    ) -> str:
        """Prefer a heading/subheading from chunk body that best matches prompt."""
        if not user_prompt or not body:
            return title

        prompt_tokens = {
            t for t in re.findall(r"[a-z0-9\-]{3,}", user_prompt.lower())
            if t not in {"test", "tests", "testcase", "testcases", "generate", "case", "cases"}
        }
        if not prompt_tokens:
            return title

        candidates = []
        for line in body.split("\n"):
            clean = self._clean_title(self._normalize_text(line))
            if len(clean) < 6 or len(clean) > 120:
                continue
            hits = sum(1 for t in prompt_tokens if t in clean.lower())
            if hits > 0:
                # Prefer concise heading-like lines.
                heading_bonus = 1 if len(clean.split()) <= 10 else 0
                candidates.append((hits + heading_bonus, clean))

        if not candidates:
            return title

        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]

    def _is_prompt_relevant(
        self,
        chunk: Dict,
        prompt: str,
        score: float,
        learned_context_words: Optional[set] = None,
    ) -> bool:
        """Hard gate so prompt-based requests return only relevant test cases."""
        chunk_text = self._normalize_text(
            f"{chunk.get('title', '')} {chunk.get('body', '')}"
        ).lower()
        prompt_tokens = {
            t for t in re.findall(r"[a-z0-9\-]{3,}", prompt.lower())
            if t not in {"test", "tests", "testcase", "testcases", "generate", "case", "cases"}
        }
        if not prompt_tokens:
            return score >= 0.08

        overlap = sum(1 for t in prompt_tokens if t in chunk_text)
        overlap_ratio = overlap / len(prompt_tokens)

        learned_overlap = 0
        if learned_context_words:
            learned_overlap = sum(1 for t in learned_context_words if t in chunk_text)

        # Require either meaningful lexical overlap or strong semantic score.
        if overlap >= 2:
            return True
        if overlap_ratio >= 0.35:
            return True
        if learned_overlap >= 2:
            return True
        return score >= 0.18

    def _learned_terms_for_prompt(self, prompt: str) -> set:
        """Gather helpful terms from previously trained examples.

        This uses locally stored training data (no external API) to improve
        prompt-to-chunk relevance filtering.
        """
        prompt_tokens = set(re.findall(r"[a-z0-9\-]{4,}", prompt.lower()))
        if not prompt_tokens:
            return set()

        examples = self.custom_trainer.get_all_examples()
        learned_terms = set()

        for example in examples[-250:]:
            source_text = self._normalize_text(example.get("document_content", "")).lower()
            tc_titles = " ".join(
                self._normalize_text(tc.get("title", ""))
                for tc in example.get("test_cases", [])
            ).lower()
            combined = f"{source_text} {tc_titles}"

            if any(t in combined for t in prompt_tokens):
                words = set(re.findall(r"[a-z0-9\-]{4,}", combined))
                learned_terms.update(words)

        # Keep bounded size for performance and noise control.
        return set(sorted(learned_terms)[:250])
