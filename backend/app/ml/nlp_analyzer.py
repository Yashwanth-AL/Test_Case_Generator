"""NLP analysis for extracting structured features and sections from documents"""
import re
from typing import Dict, List, Tuple


class NLPAnalyzer:
    """Extract structured information from documents - parses actual content, not generic keywords"""

    def analyze(self, document_content: str) -> Dict:
        """
        Parse document into discrete features/sections that each become a test case.
        Returns structured data extracted directly from the document text.
        """
        sections = self._split_into_sections(document_content)
        features = self._extract_features(document_content, sections)

        return {
            "raw_content": document_content,
            "sections": sections,
            "features": features,
            "product_name": self._extract_product_name(document_content),
            "domain": self._detect_domain(document_content),
        }

    def _split_into_sections(self, content: str) -> List[Dict]:
        """
        Split document into logical sections based on headings, numbering, or blank-line separation.
        Each section has a title and body text.
        """
        sections = []

        # Try numbered sections: "1. Title", "2.1 Title", "1) Title"
        numbered_pattern = r'(?:^|\n)\s*(\d+(?:\.\d+)*[\.\)]\s*)(.+?)(?=\n\s*\d+(?:\.\d+)*[\.\)]\s|\Z)'
        numbered_matches = re.findall(numbered_pattern, content, re.DOTALL)

        if len(numbered_matches) >= 2:
            for num, body in numbered_matches:
                lines = body.strip().split('\n')
                title = lines[0].strip()
                body_text = '\n'.join(lines[1:]).strip() if len(lines) > 1 else title
                if len(title) > 3:
                    sections.append({"title": title, "body": body_text or title})
            return sections

        # Try markdown headings: "# Title", "## Title"
        heading_pattern = r'(?:^|\n)\s*(#{1,4})\s+(.+?)(?=\n\s*#{1,4}\s|\Z)'
        heading_matches = re.findall(heading_pattern, content, re.DOTALL)

        if len(heading_matches) >= 2:
            for hashes, body in heading_matches:
                lines = body.strip().split('\n')
                title = lines[0].strip()
                body_text = '\n'.join(lines[1:]).strip()
                if len(title) > 3:
                    sections.append({"title": title, "body": body_text or title})
            return sections

        # Try bullet/dash separated items
        bullet_pattern = r'(?:^|\n)\s*[-•*]\s+(.+?)(?=\n\s*[-•*]\s|\Z)'
        bullet_matches = re.findall(bullet_pattern, content, re.DOTALL)

        if len(bullet_matches) >= 2:
            for item in bullet_matches:
                lines = item.strip().split('\n')
                title = lines[0].strip()
                body_text = '\n'.join(lines[1:]).strip()
                if len(title) > 3:
                    sections.append({"title": title, "body": body_text or title})
            return sections

        # Try paragraph separation (double newline)
        paragraphs = re.split(r'\n\s*\n', content.strip())
        if len(paragraphs) >= 2:
            for para in paragraphs:
                para = para.strip()
                if len(para) > 10:
                    lines = para.split('\n')
                    title = lines[0].strip()
                    body_text = '\n'.join(lines[1:]).strip()
                    sections.append({"title": title, "body": body_text or title})
            if len(sections) >= 2:
                return sections

        # Try single newline separation — each line is a separate feature
        lines = [line.strip() for line in content.strip().split('\n') if line.strip()]
        if len(lines) >= 2:
            sections = []
            for line in lines:
                if len(line) > 10:
                    # Split on first period to get title vs body
                    period_idx = line.find('. ')
                    if period_idx > 5:
                        title = line[:period_idx].strip()
                        body = line[period_idx + 2:].strip()
                        sections.append({"title": title, "body": body or title})
                    else:
                        sections.append({"title": line, "body": line})
            if len(sections) >= 2:
                return sections

        # Try sentence-based splitting for dense text blocks
        # Split on ". " followed by a capital letter (new topic)
        sentence_parts = re.split(r'\.\s+(?=[A-Z])', content.strip())
        if len(sentence_parts) >= 3:
            sections = []
            i = 0
            while i < len(sentence_parts):
                title_sent = sentence_parts[i].strip()
                body_sent = sentence_parts[i + 1].strip() if i + 1 < len(sentence_parts) else ""
                if len(title_sent) > 10:
                    sections.append({
                        "title": title_sent,
                        "body": body_sent or title_sent
                    })
                    i += 2  # skip the body sentence
                else:
                    i += 1
            if len(sections) >= 2:
                return sections

        # Fallback: treat entire content as one section
        if not sections:
            first_line = content.strip().split('\n')[0].strip()
            sections.append({"title": first_line, "body": content.strip()})

        return sections

    def _extract_features(self, content: str, sections: List[Dict]) -> List[Dict]:
        """
        Extract testable features from sections. Each feature becomes one test case.
        Extracts: title, description, preconditions, steps, expected_results, final_result
        """
        features = []

        for section in sections:
            feature = self._parse_feature_from_section(section)
            if feature and feature.get("title"):
                features.append(feature)

        # If no features could be parsed from sections, extract from full content
        if not features:
            features = self._extract_features_from_flat_text(content)

        return features

    def _parse_feature_from_section(self, section: Dict) -> Dict:
        """Parse a single section into a structured feature with steps, preconditions, etc."""
        title = section["title"]
        body = section["body"]
        full_text = f"{title}\n{body}"

        feature = {
            "title": self._clean_title(title),
            "description": "",
            "preconditions": [],
            "steps": [],
            "expected_results": [],
            "final_result": "",
        }

        # Extract description - first sentence or line after title
        desc_lines = []
        body_lines = body.split('\n')
        for line in body_lines:
            line = line.strip()
            if not line:
                continue
            # Stop at known section headers
            if re.match(r'(?:pre-?conditions?|steps|expected|final|result|acceptance)', line, re.IGNORECASE):
                break
            if not re.match(r'^\s*[-•*\d]', line):
                desc_lines.append(line)
                if len(desc_lines) >= 2:
                    break
        feature["description"] = ' '.join(desc_lines).strip() if desc_lines else title

        # Extract preconditions
        feature["preconditions"] = self._extract_list_after_header(
            full_text,
            r'(?:pre-?conditions?|prerequisites?|requirements?)\s*[:\-]?\s*'
        )

        # Extract steps
        feature["steps"] = self._extract_list_after_header(
            full_text,
            r'(?:steps?|procedure|actions?|test steps?)\s*(?:&\s*expected\s*results?)?\s*[:\-]?\s*'
        )

        # If steps contain "→" or "->", split into step and expected result pairs
        if feature["steps"]:
            parsed_steps = []
            parsed_results = []
            for step in feature["steps"]:
                if '→' in step or '->' in step:
                    parts = re.split(r'\s*[→\->]+\s*', step, maxsplit=1)
                    parsed_steps.append(parts[0].strip())
                    if len(parts) > 1:
                        parsed_results.append(parts[1].strip())
                else:
                    parsed_steps.append(step.strip())
            feature["steps"] = parsed_steps
            if parsed_results:
                feature["expected_results"] = parsed_results

        # Extract expected results (if not already from step→result pairs)
        if not feature["expected_results"]:
            feature["expected_results"] = self._extract_list_after_header(
                full_text,
                r'(?:expected\s*results?|expected\s*outcome|verification)\s*[:\-]?\s*'
            )

        # Extract final result
        final_match = re.search(
            r'(?:final\s*result|overall\s*result|conclusion)\s*[:\-]?\s*(.+?)(?:\n|$)',
            full_text, re.IGNORECASE
        )
        if final_match:
            feature["final_result"] = final_match.group(1).strip()

        # If no steps were found, try to extract action verbs from body
        if not feature["steps"]:
            feature["steps"] = self._extract_action_steps(body)

        # If no expected results, generate from steps
        if not feature["expected_results"] and feature["steps"]:
            feature["expected_results"] = [
                f"{step} completed successfully" for step in feature["steps"][:5]
            ]

        return feature

    def _extract_list_after_header(self, text: str, header_pattern: str) -> List[str]:
        """Extract a list of items after a header pattern"""
        match = re.search(header_pattern, text, re.IGNORECASE)
        if not match:
            return []

        after_header = text[match.end():]
        items = []

        # Try bullet/numbered list items
        list_items = re.findall(
            r'(?:^|\n)\s*(?:[-•*]|\d+[\.\)])\s+(.+?)(?=\n\s*(?:[-•*]|\d+[\.\)])\s|\n\s*\n|\Z)',
            after_header, re.DOTALL
        )

        if list_items:
            for item in list_items:
                cleaned = item.strip().replace('\n', ' ')
                if cleaned and len(cleaned) > 2:
                    items.append(cleaned)
            return items[:15]

        # Try comma-separated items
        first_line = after_header.split('\n')[0].strip()
        if ',' in first_line:
            items = [item.strip() for item in first_line.split(',') if item.strip()]
            return items[:15]

        # Try line-by-line items until empty line or next header
        for line in after_header.split('\n'):
            line = line.strip()
            if not line:
                if items:
                    break
                continue
            if re.match(r'(?:pre-?conditions?|steps|expected|final|result|acceptance)', line, re.IGNORECASE):
                break
            if len(line) > 2:
                items.append(line)

        return items[:15]

    def _extract_action_steps(self, text: str) -> List[str]:
        """Extract action steps by looking for imperative verbs"""
        action_verbs = [
            'open', 'navigate', 'click', 'select', 'enter', 'verify', 'check',
            'enable', 'disable', 'configure', 'set', 'save', 'login', 'log in',
            'start', 'stop', 'create', 'delete', 'update', 'assign', 'connect',
            'test', 'validate', 'confirm', 'apply', 'install', 'upload', 'download',
            'send', 'receive', 'initiate', 'wait', 'reboot', 'restart', 'detect',
            'discover', 'pair', 'unpair', 'add', 'remove'
        ]

        steps = []
        for line in text.split('\n'):
            line = line.strip().lstrip('•-*0123456789.) ')
            if not line:
                continue
            first_word = line.split()[0].lower() if line.split() else ""
            if first_word in action_verbs:
                steps.append(line)

        return steps[:15]

    def _extract_features_from_flat_text(self, content: str) -> List[Dict]:
        """Extract features from unstructured text by finding action sentences"""
        features = []
        sentences = re.split(r'[.!?\n]+', content)

        current_feature = None
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 10:
                continue

            # Start new feature on capitalized or action-oriented sentences
            if re.match(r'^[A-Z]', sentence) and len(sentence) > 15:
                if current_feature and current_feature.get("steps"):
                    features.append(current_feature)
                current_feature = {
                    "title": sentence[:80],
                    "description": sentence,
                    "preconditions": [],
                    "steps": [],
                    "expected_results": [],
                    "final_result": "",
                }
            elif current_feature:
                current_feature["steps"].append(sentence)

        if current_feature and current_feature.get("title"):
            features.append(current_feature)

        return features

    def _clean_title(self, title: str) -> str:
        """Convert raw section text into a concise, meaningful test case title"""
        # Remove leading numbers, bullets, hashes
        title = re.sub(r'^[\s#*•\-\d\.\)]+', '', title).strip()
        # Remove spec-style prefix: "The system shall ..." → keep the rest
        title = re.sub(
            r'^The\s+(?:system\s+)?(?:shall|should|must|will)\s+', '',
            title, flags=re.IGNORECASE
        ).strip()
        # Capitalise first letter
        if title:
            title = title[0].upper() + title[1:]

        # If short enough, return as-is
        if len(title) <= 65:
            return title

        # Remove trailing device context "on/for/in [device name]"
        trimmed = re.sub(
            r'\s+(?:on|in|for|using|via|from|at|through)\s+(?:the\s+)?[\w\s\-]{3,50}$',
            '', title, flags=re.IGNORECASE
        ).strip()
        if 10 < len(trimmed) <= 65:
            return trimmed

        # Cut at first " and " or ","
        and_match = re.search(r'(?:,|\s+and\s+)', title[:70])
        if and_match and and_match.start() > 15:
            return title[:and_match.start()].strip()

        # Keep first 65 chars at a word boundary
        truncated = title[:65]
        last_space = truncated.rfind(' ')
        if last_space > 20:
            return truncated[:last_space]
        return truncated

    def _extract_product_name(self, content: str) -> str:
        """Try to extract product/system name from the document"""
        # Look for known patterns
        patterns = [
            r'(?:product|system|device|server|module|unit)\s*(?:name)?\s*[:\-]\s*(.+?)(?:\n|$)',
            r'(?:Panel\s*Server|PowerTag|EcoStruxure|Modbus|EM\d+)',
        ]
        for pattern in patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                return match.group(0).strip() if not match.lastindex else match.group(1).strip()
        return ""

    def _detect_domain(self, content: str) -> str:
        """Detect the domain/category of the document"""
        content_lower = content.lower()

        domain_keywords = {
            "commissioning": ["commissioning", "commission", "setup", "configure", "install", "firmware",
                              "panel server", "modbus", "rs-485", "ethernet", "pairing", "discovery",
                              "powertag", "ecostruxure", "baud rate", "network settings"],
            "api_testing": ["api", "endpoint", "rest", "http", "request", "response", "json", "payload"],
            "security": ["security", "authentication", "authorization", "login", "password", "token",
                         "permission", "lockout", "encryption", "ssl", "tls", "vulnerability"],
            "performance": ["performance", "load test", "stress test", "throughput", "latency",
                           "scalability", "response time", "benchmark"],
            "integration": ["integration", "third-party", "external service", "webhook",
                           "communication protocol", "data exchange"],
            "ui_testing": ["ui", "user interface", "button", "form", "display", "navigation",
                          "responsive", "accessibility", "layout"],
            "network": ["network", "tcp", "ip", "ethernet", "wireless", "wifi", "bluetooth",
                       "communication", "protocol", "filtering"],
            "firmware": ["firmware", "update", "upgrade", "version", "flash", "reboot",
                        "software update"],
            "alarm_monitoring": ["alarm", "alert", "threshold", "monitoring", "overload",
                                "notification", "trigger"],
            "cloud": ["cloud", "publication", "dashboard", "remote", "iot", "data publication"],
        }

        scores = {}
        for domain, keywords in domain_keywords.items():
            score = sum(content_lower.count(kw) for kw in keywords)
            if score > 0:
                scores[domain] = score

        if scores:
            return max(scores, key=scores.get)
        return "general"
