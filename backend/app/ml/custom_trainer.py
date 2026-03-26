"""Custom training system for learning from user-provided examples"""
import json
import os
from typing import Dict, List, Any, Optional
from datetime import datetime
import re


class CustomTrainer:
    """Manages custom training examples and learned patterns"""
    
    DEFAULT_TRAINING_DATA_PATH = os.path.join(
        os.path.dirname(__file__), 
        "..", 
        "data", 
        "custom_training_examples.json"
    )
    
    def __init__(self, data_path: Optional[str] = None):
        """Initialize custom trainer"""
        self.data_path = data_path or self.DEFAULT_TRAINING_DATA_PATH
        self._ensure_data_file_exists()
    
    def _ensure_data_file_exists(self):
        """Create data directory and file if they don't exist"""
        os.makedirs(os.path.dirname(self.data_path), exist_ok=True)
        
        if not os.path.exists(self.data_path):
            initial_data = {
                "examples": [],
                "learned_patterns": [],
                "last_updated": datetime.now().isoformat()
            }
            self._save_data(initial_data)
    
    def _load_data(self) -> Dict:
        """Load training data from file"""
        try:
            with open(self.data_path, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {"examples": [], "learned_patterns": [], "last_updated": datetime.now().isoformat()}
    
    def _save_data(self, data: Dict):
        """Save training data to file"""
        data["last_updated"] = datetime.now().isoformat()
        with open(self.data_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def add_example(
        self,
        document_content: str,
        test_cases: List[Dict[str, Any]],
        tags: Optional[List[str]] = None
    ) -> Dict:
        """
        Add a training example
        
        Args:
            document_content: The input document
            test_cases: List of test cases generated for this document
            tags: Optional tags to categorize the example
        
        Returns:
            Added example with metadata
        """
        data = self._load_data()
        
        # Extract keywords from document for pattern learning
        extracted_keywords = self._extract_keywords(document_content)
        
        example = {
            "id": f"EXP-{len(data['examples']) + 1:04d}",
            "document_content": document_content,
            "test_cases": test_cases,
            "tags": tags or [],
            "extracted_keywords": extracted_keywords,
            "test_case_count": len(test_cases),
            "created_at": datetime.now().isoformat()
        }
        
        data["examples"].append(example)
        
        # Learn patterns from this example
        self._learn_patterns(data, example)
        
        self._save_data(data)
        return example
    
    def _extract_keywords(self, content: str) -> Dict[str, int]:
        """Extract and count keywords from document"""
        keywords_to_track = {
            "api": ["api", "endpoint", "rest", "request", "response", "http"],
            "security": ["security", "authentication", "authorization", "login", "password", "token", "permission"],
            "database": ["database", "sql", "query", "table", "entity", "persist", "data"],
            "performance": ["performance", "load", "stress", "throughput", "latency", "optimization"],
            "integration": ["integration", "third-party", "external", "service", "communication"],
            "ui": ["ui", "interface", "button", "form", "display", "navigation", "responsive"],
            "workflow": ["workflow", "scenario", "process", "step", "flow", "journey"],
        }
        
        content_lower = content.lower()
        extracted = {}
        
        for category, keywords in keywords_to_track.items():
            count = sum(content_lower.count(kw) for kw in keywords)
            if count > 0:
                extracted[category] = count
        
        return extracted
    
    def _learn_patterns(self, data: Dict, example: Dict):
        """Learn patterns from the example"""
        test_case_types = {}
        
        # Count test case types in this example
        for tc in example["test_cases"]:
            tc_type = tc.get("test_type", "Unit Testing")
            test_case_types[tc_type] = test_case_types.get(tc_type, 0) + 1
        
        # Create or update patterns
        for category, keyword_count in example["extracted_keywords"].items():
            # Find existing pattern or create new one
            pattern = None
            for p in data["learned_patterns"]:
                if p["category"] == category:
                    pattern = p
                    break
            
            if not pattern:
                pattern = {
                    "category": category,
                    "test_case_types": {},
                    "frequency": 0,
                    "examples": []
                }
                data["learned_patterns"].append(pattern)
            
            # Update pattern
            pattern["frequency"] += 1
            pattern["examples"].append(example["id"])
            
            # Track which test case types are associated with this category
            for tc_type, count in test_case_types.items():
                if tc_type not in pattern["test_case_types"]:
                    pattern["test_case_types"][tc_type] = 0
                pattern["test_case_types"][tc_type] += count
    
    def get_all_examples(self) -> List[Dict]:
        """Get all training examples"""
        data = self._load_data()
        return data.get("examples", [])
    
    def get_examples_by_tag(self, tag: str) -> List[Dict]:
        """Get examples filtered by tag"""
        data = self._load_data()
        examples = data.get("examples", [])
        return [e for e in examples if tag in e.get("tags", [])]
    
    def get_learned_patterns(self) -> List[Dict]:
        """Get all learned patterns"""
        data = self._load_data()
        return data.get("learned_patterns", [])
    
    def get_pattern_by_category(self, category: str) -> Optional[Dict]:
        """Get learned pattern for a category"""
        patterns = self.get_learned_patterns()
        for pattern in patterns:
            if pattern["category"] == category:
                return pattern
        return None
    
    def find_relevant_examples(
        self,
        extracted_keywords: Dict[str, int],
        limit: int = 5
    ) -> List[Dict]:
        """
        Find training examples similar to the current document
        
        Args:
            extracted_keywords: Keywords extracted from current document
            limit: Maximum number of examples to return
        
        Returns:
            List of relevant examples, sorted by relevance
        """
        examples = self.get_all_examples()
        
        # Score each example based on keyword match
        scored_examples = []
        for example in examples:
            score = 0
            example_keywords = example.get("extracted_keywords", {})
            
            # Calculate similarity score
            for category, count in extracted_keywords.items():
                if category in example_keywords:
                    score += min(count, example_keywords[category])
            
            if score > 0:
                scored_examples.append((example, score))
        
        # Sort by score descending
        scored_examples.sort(key=lambda x: x[1], reverse=True)
        
        return [ex for ex, score in scored_examples[:limit]]
    
    def get_recommendations(self, extracted_keywords: Dict[str, int]) -> Dict[str, Any]:
        """
        Get test case type recommendations based on learned patterns
        
        Args:
            extracted_keywords: Keywords from current document
        
        Returns:
            Recommendations for which test case types to generate
        """
        recommendations = {
            "test_types": {},
            "confidence": 0,
            "learned_from_examples": 0
        }
        
        relevant_examples = self.find_relevant_examples(extracted_keywords)
        recommendations["learned_from_examples"] = len(relevant_examples)
        
        if not relevant_examples:
            return recommendations
        
        # Aggregate test case types from relevant examples
        type_scores = {}
        for example in relevant_examples:
            for tc in example["test_cases"]:
                tc_type = tc.get("test_type", "Unit Testing")
                type_scores[tc_type] = type_scores.get(tc_type, 0) + 1
        
        # Calculate confidence based on agreement among examples
        if type_scores:
            max_score = max(type_scores.values())
            recommendations["confidence"] = min(max_score / len(relevant_examples), 1.0)
        
        recommendations["test_types"] = type_scores
        
        return recommendations
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get training statistics"""
        data = self._load_data()
        examples = data.get("examples", [])
        patterns = data.get("learned_patterns", [])
        
        total_test_cases = sum(e.get("test_case_count", 0) for e in examples)
        
        test_type_distribution = {}
        for example in examples:
            for tc in example.get("test_cases", []):
                tc_type = tc.get("test_type", "Unknown")
                test_type_distribution[tc_type] = test_type_distribution.get(tc_type, 0) + 1
        
        return {
            "total_examples": len(examples),
            "total_patterns": len(patterns),
            "total_test_cases_trained": total_test_cases,
            "test_type_distribution": test_type_distribution,
            "categories_learned": [p["category"] for p in patterns],
            "last_updated": data.get("last_updated")
        }
