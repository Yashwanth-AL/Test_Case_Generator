"""AI Test Case Generator - Data-Specific Version
Generates test cases that are specific to the actual document content,
not generic templates. Uses training examples as reference patterns.
"""
import re
from typing import List, Optional, Dict
from app.models.schemas import TestCase, DetailLevelEnum
from app.ml.nlp_analyzer import NLPAnalyzer
from app.ml.custom_trainer import CustomTrainer


class TestCaseGenerator:
    """Generate data-specific test cases by parsing document features
    and matching against training examples."""

    def __init__(self):
        self.nlp_analyzer = NLPAnalyzer()
        self.custom_trainer = CustomTrainer()

    def generate(
        self,
        document_content: str,
        test_types: Optional[List[str]] = None,
        detail_level: DetailLevelEnum = DetailLevelEnum.DETAILED,
        num_test_cases: Optional[int] = None,
        id_prefix: str = "ATC",
    ) -> List[TestCase]:
        """
        Generate test cases from document content.

        Strategy:
        1. Parse document into discrete features/sections
        2. For each feature, find the most similar training example
        3. Extract entities (action, subject, device) from the feature
        4. Generate content-specific steps/results via slot-filling
        5. Use training example only for test type / priority metadata
        """
        analysis = self.nlp_analyzer.analyze(document_content)
        features = analysis.get("features", [])
        domain = analysis.get("domain", "general")
        product_name = analysis.get("product_name", "")

        default_test_type = self._determine_test_type(domain, document_content)
        if test_types and len(test_types) > 0:
            default_test_type = test_types[0]

        all_examples = self.custom_trainer.get_all_examples()

        test_cases = []
        base_id = 1647

        for idx, feature in enumerate(features):
            if num_test_cases and len(test_cases) >= num_test_cases:
                break

            best_example = self._find_best_matching_example(feature, all_examples)

            tc = self._generate_test_case_for_feature(
                feature=feature,
                example_template=best_example,
                index=idx,
                base_id=base_id,
                id_prefix=id_prefix,
                test_type=default_test_type,
                detail_level=detail_level,
                product_name=product_name,
            )

            if tc:
                test_cases.append(tc)

        if num_test_cases:
            test_cases = test_cases[:num_test_cases]

        return test_cases

    # ── Type / domain helpers ──────────────────────────────────────────────────

    def _determine_test_type(self, domain: str, content: str) -> str:
        """Determine the test type from domain and training patterns"""
        patterns = self.custom_trainer.get_learned_patterns()
        if patterns:
            best_pattern = max(patterns, key=lambda p: p.get("frequency", 0))
            test_types = best_pattern.get("test_case_types", {})
            if test_types:
                return max(test_types, key=test_types.get)

        domain_to_type = {
            "commissioning": "Commissioning",
            "api_testing": "API Testing",
            "security": "Security Testing",
            "performance": "Performance Testing",
            "integration": "Integration Testing",
            "ui_testing": "Usability Testing",
            "network": "Commissioning",
            "firmware": "Commissioning",
            "alarm_monitoring": "Commissioning",
            "cloud": "Integration Testing",
        }
        return domain_to_type.get(domain, "Functional Testing")

    # ── Training example matching ──────────────────────────────────────────────

    def _find_best_matching_example(
        self, feature: Dict, examples: List[Dict]
    ) -> Optional[Dict]:
        """Find the training example most similar to this feature"""
        if not examples:
            return None

        feature_text = (
            f"{feature.get('title', '')} "
            f"{feature.get('description', '')} "
            f"{' '.join(feature.get('steps', []))}"
        )
        feature_words = set(re.findall(r'\b\w{3,}\b', feature_text.lower()))

        best_score = 0
        best_example = None

        specific_domain_words = {
            'modbus', 'rs-485', 'rs485', 'serial', 'baud', 'parity',
            'powertag', 'ecostruxure', 'epc', 'overload', 'lockout',
            'tcp', 'filtering', 'switched', 'firmware', 'pairing',
            'wireless', 'wired',
        }

        for example in examples:
            example_text = example.get("document_content", "")
            for tc in example.get("test_cases", []):
                example_text += f" {tc.get('title', '')} {tc.get('description', '')}"

            example_words = set(re.findall(r'\b\w{3,}\b', example_text.lower()))

            if not feature_words or not example_words:
                continue

            intersection = feature_words & example_words
            union = feature_words | example_words
            score = len(intersection) / len(union) if union else 0

            domain_overlap = feature_words & example_words & specific_domain_words
            score += len(domain_overlap) * 0.15

            if score > best_score:
                best_score = score
                best_example = example

        return best_example if best_score >= 0.25 else None

    # ── Top-level test case builder ───────────────────────────────────────────

    def _generate_test_case_for_feature(
        self,
        feature: Dict,
        example_template: Optional[Dict],
        index: int,
        base_id: int,
        id_prefix: str,
        test_type: str,
        detail_level: DetailLevelEnum,
        product_name: str,
    ) -> Optional[TestCase]:
        """Route to the appropriate builder depending on whether a template matched"""
        title = feature.get("title", "").strip()
        if not title or len(title) < 3:
            return None

        tc_id = (
            f"{product_name}-{id_prefix}-{base_id + index}"
            if product_name
            else f"{id_prefix}-{base_id + index}"
        )

        if example_template and example_template.get("test_cases"):
            return self._generate_from_example_template(
                feature=feature,
                example=example_template,
                tc_id=tc_id,
                test_type=test_type,
                detail_level=detail_level,
            )

        return self._generate_from_feature_data(
            feature=feature,
            tc_id=tc_id,
            test_type=test_type,
            detail_level=detail_level,
        )

    # ── Core builders ─────────────────────────────────────────────────────────

    def _generate_from_example_template(
        self,
        feature: Dict,
        example: Dict,
        tc_id: str,
        test_type: str,
        detail_level: DetailLevelEnum,
    ) -> TestCase:
        """Use training example for test type / priority metadata, but generate
        content-specific steps, results, and preconditions from the feature."""
        template_tc = example["test_cases"][0]

        title = feature.get("title", "")
        description = feature.get("description", title)

        entities = self._extract_feature_entities(title, description)
        action = entities["action"]
        subject = entities["subject"]
        device = entities["device"]
        params = entities["params"]

        # Only reuse the feature's own parsed steps when they are well-formed
        feature_steps = feature.get("steps", [])
        if len(feature_steps) >= 3 and all(len(s) > 10 for s in feature_steps):
            steps = feature_steps
        else:
            steps = self._generate_contextual_steps(action, subject, device, params, description)

        feature_results = feature.get("expected_results", [])
        if len(feature_results) >= 3 and all(len(r) > 10 for r in feature_results):
            expected_results = feature_results
        else:
            expected_results = self._generate_contextual_results(action, subject, device, steps)

        feature_preconds = feature.get("preconditions", [])
        if len(feature_preconds) >= 2:
            preconditions = feature_preconds
        else:
            preconditions = self._generate_contextual_preconditions(
                action, subject, device, description
            )

        feature_final = feature.get("final_result", "")
        final_result = (
            feature_final
            if feature_final
            else self._generate_contextual_final_result(action, subject, device)
        )

        if not description or description == title or len(description) < 15:
            description = f"Validate '{title}' based on the provided document requirements."

        objective = self._build_objective(action, subject, device)
        example_type = template_tc.get("test_type", test_type)
        priority = template_tc.get(
            "priority", self._determine_priority(title, description, steps)
        )
        acceptance_criteria = self._build_acceptance_criteria(
            title, expected_results, final_result
        )

        steps, expected_results, preconditions = self._apply_detail_level(
            steps, expected_results, preconditions, detail_level
        )

        return TestCase(
            id=tc_id,
            test_type=example_type,
            title=title,
            method=template_tc.get("method", "Manual Testing"),
            description=description,
            objective=objective,
            preconditions=preconditions,
            steps=steps,
            expected_results=expected_results,
            priority=priority,
            acceptance_criteria=acceptance_criteria,
            final_result=final_result,
        )

    def _generate_from_feature_data(
        self,
        feature: Dict,
        tc_id: str,
        test_type: str,
        detail_level: DetailLevelEnum,
    ) -> TestCase:
        """Generate test case entirely from parsed feature data (no matching example)"""
        title = feature.get("title", "")
        description = feature.get("description", title)

        entities = self._extract_feature_entities(title, description)
        action = entities["action"]
        subject = entities["subject"]
        device = entities["device"]
        params = entities["params"]

        feature_steps = feature.get("steps", [])
        if len(feature_steps) >= 3:
            steps = feature_steps
        else:
            steps = self._generate_contextual_steps(action, subject, device, params, description)

        feature_results = feature.get("expected_results", [])
        if len(feature_results) >= 3:
            expected_results = feature_results
        else:
            expected_results = self._generate_contextual_results(action, subject, device, steps)

        preconditions = feature.get("preconditions", [])
        if len(preconditions) < 2:
            preconditions = self._generate_contextual_preconditions(
                action, subject, device, description
            )

        final_result = feature.get("final_result", "") or self._generate_contextual_final_result(
            action, subject, device
        )

        priority = self._determine_priority(title, description, steps)
        objective = self._build_objective(action, subject, device)
        acceptance_criteria = self._build_acceptance_criteria(title, expected_results, final_result)

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

    # ── Entity extraction ──────────────────────────────────────────────────────

    def _extract_feature_entities(self, title: str, description: str) -> Dict:
        """Extract action type, subject, and device from a feature's title + description"""
        text_lower = f"{title} {description}".lower()

        action_map = [
            ("enable",    r'\b(?:enable|activate|turn\s+on)\b'),
            ("disable",   r'\b(?:disable|deactivate|turn\s+off)\b'),
            ("configure", r'\b(?:configure|set\s+up|setup|define|specify|assign)\b'),
            ("verify",    r'\b(?:verify|check|confirm|validate|ensure)\b'),
            ("update",    r'\b(?:update|upgrade|install|flash)\b'),
            ("monitor",   r'\b(?:monitor|track|observe|detect|alert)\b'),
            ("test",      r'\b(?:test|testing|evaluate)\b'),
        ]
        action = "test"
        for action_name, pattern in action_map:
            if re.search(pattern, text_lower):
                action = action_name
                break

        subject = self._extract_subject_from_title(title, action)
        device = self._extract_device_name(f"{title} {description}")
        params = self._extract_params_from_text(f"{title} {description}")

        return {"action": action, "subject": subject, "device": device, "params": params}

    def _extract_subject_from_title(self, title: str, action: str) -> str:
        """Extract the testable subject by stripping the action verb and device context"""
        subject = re.sub(
            r'^(?:configure|enable|disable|verify|test|check|validate|confirm|update|'
            r'ensure|install|set\s+up|establish|create|delete|modify|assign|monitor)\s+'
            r'(?:the\s+|a\s+|an\s+)?',
            '', title.strip(), flags=re.IGNORECASE
        ).strip()

        # Remove trailing "on/for/in [device]"
        subject = re.sub(
            r'\s+(?:on|for|in|of|at|via|using|through|from)\s+(?:the\s+)?[\w\s\-]{2,}$',
            '', subject, flags=re.IGNORECASE
        ).strip()

        # Remove "after/when/while" conditions
        subject = re.sub(
            r'\s+(?:after|when|while|during|by|with)\s+.+$',
            '', subject, flags=re.IGNORECASE
        ).strip()

        if len(subject) > 55:
            and_match = re.search(r'(?:\s+and\s+|,)', subject[:55])
            if and_match and and_match.start() > 10:
                subject = subject[:and_match.start()].strip()
            else:
                subject = subject[:55].rsplit(' ', 1)[0]

        return subject if len(subject) > 3 else title[:55]

    def _extract_device_name(self, text: str) -> str:
        """Extract device / system name from text"""
        device_patterns = [
            r'\b(Panel\s*Server\s*(?:EPC|EGX)?)\b',
            r'\b(EcoStruxure\s+[\w\s]{3,30})',
            r'\b(PowerTag\s*[\w\d]*)',
            r'\b(EM\d+[\w\d]*)',
            r'\b(BMX\s*[\w\d]+)',
            r'\b(Modicon\s+[\w\d]+)',
        ]
        for pattern in device_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        generic = re.search(
            r'\b(panel\s+server|web\s+server|gateway|controller|device)\b',
            text, re.IGNORECASE
        )
        if generic:
            return generic.group(1).strip().title()
        return ""

    def _build_objective(self, action: str, subject: str, device: str) -> str:
        """Build a device-aware objective without forcing unrelated product names."""
        subj = subject if subject else "the specified feature"
        if device:
            return f"To {action} and validate {subj} on {device}"
        return f"To {action} and validate {subj} according to the provided requirements"

    def _extract_params_from_text(self, text: str) -> List[str]:
        """Extract specific parameter values mentioned in text"""
        params = []
        param_patterns = [
            (r'baud\s*rate[:\s]+(\d+)', 'Baud Rate'),
            (r'\bport[:\s]+(\d+)', 'Port'),
            (r'\bparity[:\s]+(\w+)', 'Parity'),
            (r'timeout[:\s]+(\d+\s*(?:ms|s|sec)?)', 'Timeout'),
            (r'interval[:\s]+(\d+\s*(?:ms|s|min)?)', 'Interval'),
            (r'ip\s*address[:\s]+([\d\.]+)', 'IP'),
        ]
        for pattern, label in param_patterns:
            m = re.search(pattern, text, re.IGNORECASE)
            if m:
                params.append(f"{label}: {m.group(1)}")
        return params[:4]

    # ── Contextual step / result / precondition generation ────────────────────

    def _generate_contextual_steps(
        self,
        action: str,
        subject: str,
        device: str,
        params: List[str],
        description: str,
    ) -> List[str]:
        """Generate content-specific steps via action + subject + device slot-filling"""
        d = device if device else "the target system"
        s = subject if subject else "the feature"
        p_str = f" ({', '.join(params[:2])})" if params else ""

        templates: Dict[str, List[str]] = {
            "enable": [
                f"Log in to {d} web interface with admin credentials",
                f"Navigate to the {s} settings page",
                f"Enable {s}",
                f"Configure required parameters{p_str}",
                f"Save the configuration",
                f"Verify {s} status shows Active / Enabled",
                f"Send a test request to confirm {s} is operational",
            ],
            "disable": [
                f"Log in to {d} web interface with admin credentials",
                f"Navigate to the {s} settings page",
                f"Disable {s}",
                f"Save the configuration",
                f"Verify {s} is no longer active",
                f"Confirm no unintended side effects from disabling {s}",
            ],
            "configure": [
                f"Log in to {d} web interface with admin credentials",
                f"Navigate to the {s} configuration section",
                f"Review the current {s} settings",
                f"Set {s} parameters as required{p_str}",
                f"Apply and save the configuration",
                f"Verify {s} settings are correctly reflected",
                f"Validate {s} is functioning as expected",
            ],
            "verify": [
                f"Log in to {d}",
                f"Navigate to the {s} section",
                f"Check the current {s} status and values",
                f"Verify {s} meets the specified requirements{p_str}",
                f"Confirm {s} behaviour under normal operating conditions",
                f"Document the verification result",
            ],
            "update": [
                f"Obtain the latest {s} file / package",
                f"Log in to {d} management interface with admin credentials",
                f"Navigate to the {s} update section",
                f"Upload the {s} file",
                f"Initiate the update process",
                f"Wait for the update to complete",
                f"Verify {s} version / status after update",
                f"Confirm {d} functionality post-update",
            ],
            "monitor": [
                f"Log in to {d}",
                f"Navigate to the {s} monitoring section",
                f"Configure {s} monitoring parameters{p_str}",
                f"Set alert thresholds for {s}",
                f"Simulate a {s} condition change",
                f"Verify {s} alert / notification is triggered",
                f"Confirm monitoring data is recorded correctly",
            ],
            "test": [
                f"Prepare test environment for {s}",
                f"Log in to {d} with appropriate credentials",
                f"Navigate to the {s} section",
                f"Execute {s} test procedure{p_str}",
                f"Monitor system response during the test",
                f"Verify {s} behaves as expected",
                f"Document test outcome and results",
            ],
        }
        return templates.get(action, templates["test"])

    def _generate_contextual_results(
        self,
        action: str,
        subject: str,
        device: str,
        steps: List[str],
    ) -> List[str]:
        """Generate expected results mapped 1-to-1 with contextual steps"""
        d = device if device else "the system"
        s = subject if subject else "the feature"
        results = []
        for step in steps:
            sl = step.lower()
            if "log in" in sl or "login" in sl:
                results.append(f"Login successful; {d} interface is accessible")
            elif "navigate" in sl:
                results.append(f"Requested section / page displayed correctly")
            elif "enable" in sl:
                results.append(f"{s} is enabled successfully")
            elif "disable" in sl:
                results.append(f"{s} is disabled successfully")
            elif "review" in sl or "check" in sl:
                results.append(f"Current {s} settings are visible and readable")
            elif "set " in sl or "configure" in sl:
                results.append(f"{s} parameters are configured correctly")
            elif "apply" in sl or "save" in sl:
                results.append(f"Configuration saved without errors")
            elif "verify" in sl:
                results.append(f"{s} is verified and meets requirements")
            elif "validate" in sl or "test" in sl:
                results.append(f"{s} passes validation / test")
            elif "upload" in sl:
                results.append(f"File uploaded to {d} successfully")
            elif "obtain" in sl or "download" in sl:
                results.append(f"File obtained / downloaded successfully")
            elif "initiate" in sl or "start" in sl:
                results.append(f"Process initiated successfully")
            elif "wait" in sl:
                results.append(f"Process completes within expected time")
            elif "confirm" in sl:
                results.append(f"{s} confirmed functioning correctly")
            elif "simulate" in sl:
                results.append(f"Simulated condition triggered as expected")
            elif "document" in sl:
                results.append(f"Test results documented accurately")
            elif "monitor" in sl:
                results.append(f"Monitoring data captured and displayed correctly")
            elif "send" in sl:
                results.append(f"{s} responds correctly to the test request")
            else:
                results.append(f"{s} operation completed successfully")
        return results
    def _generate_contextual_results(
        self,
        action: str,
        subject: str,
        device: str,
        steps: List[str],
    ) -> List[str]:
        """Generate expected results mapped 1-to-1 with contextual steps"""
        d = device if device else "the system"
        s = subject if subject else "the feature"
        results = []
        for step in steps:
            sl = step.lower()
            if re.search(r'\blog\s+in\b|\blogin\b', sl):
                results.append(f"Login successful; {d} interface is accessible")
            elif re.search(r'\bnavigate\b', sl):
                results.append(f"Requested section / page displayed correctly")
            elif re.search(r'\benable\b', sl):
                results.append(f"{s} is enabled successfully")
            elif re.search(r'\bdisable\b', sl):
                results.append(f"{s} is disabled successfully")
            elif re.search(r'\breview\b|\bcheck\b', sl):
                results.append(f"Current {s} settings are visible and readable")
            elif re.search(r'\bset\b|\bconfigure\b', sl):
                results.append(f"{s} parameters are configured correctly")
            elif re.search(r'\bapply\b|\bsave\b', sl):
                results.append(f"Configuration saved without errors")
            elif re.search(r'\bverify\b', sl):
                results.append(f"{s} is verified and meets requirements")
            elif re.search(r'\bobtain\b|\bdownload\b', sl):
                results.append(f"File obtained / downloaded successfully")
            elif re.search(r'\bupload\b', sl):
                results.append(f"File uploaded to {d} successfully")
            elif re.search(r'\binitiate\b|\bstart\b', sl):
                results.append(f"Process initiated successfully")
            elif re.search(r'\bwait\b', sl):
                results.append(f"Process completes within expected time")
            elif re.search(r'\bconfirm\b', sl):
                results.append(f"{s} confirmed functioning correctly")
            elif re.search(r'\bsimulate\b', sl):
                results.append(f"Simulated condition triggered as expected")
            elif re.search(r'\bdocument\b', sl):
                results.append(f"Test results documented accurately")
            elif re.search(r'\bmonitor\b', sl):
                results.append(f"Monitoring data captured and displayed correctly")
            elif re.search(r'\bsend\b', sl):
                results.append(f"{s} responds correctly to the test request")
            elif re.search(r'\bvalidate\b|\btest\b|\bexecute\b', sl):
                results.append(f"{s} passes validation / test")
            else:
                results.append(f"{s} operation completed successfully")
        return results

    def _generate_contextual_preconditions(
        self,
        action: str,
        subject: str,
        device: str,
        description: str,
    ) -> List[str]:
        """Generate preconditions specific to the feature"""
        preconditions: List[str] = []
        text = f"{subject} {device} {description}".lower()

        preconditions.append(
            f"{device if device else 'System'} is powered ON and connected to the network"
        )
        if any(kw in text for kw in ['web', 'browser', 'interface', 'http', 'ui']):
            preconditions.append("Web browser is available and up to date")
        if action in ['configure', 'enable', 'disable', 'update', 'test']:
            preconditions.append("Admin credentials are available")
        if any(kw in text for kw in ['firmware', 'update', 'upgrade', 'flash']):
            preconditions.append("Firmware / update file is downloaded and available")
        if any(kw in text for kw in ['modbus', 'rs-485', 'serial', 'rs485']):
            preconditions.append("Serial communication cable is connected")
        if any(kw in text for kw in ['powertag', 'wireless', 'pairing', 'pair']):
            preconditions.append("Device is in pairing / discovery mode")
        if any(kw in text for kw in ['cloud', 'publication', 'remote', 'iot']):
            preconditions.append("Internet connection is available")
        if any(kw in text for kw in ['alarm', 'alert', 'threshold', 'overload']):
            preconditions.append("Monitoring / alarm thresholds are defined")

        return preconditions[:5]

    def _generate_contextual_final_result(
        self, action: str, subject: str, device: str
    ) -> str:
        """Generate a content-specific final result statement"""
        d = device if device else "the system"
        s = subject if subject else "the feature"
        templates = {
            "enable":    f"{s} is successfully enabled and operational on {d}",
            "disable":   f"{s} is successfully disabled on {d}",
            "configure": f"{s} is correctly configured and validated on {d}",
            "verify":    f"{s} is verified and meets all specified requirements",
            "update":    f"{s} update is successful; {d} is functioning correctly post-update",
            "monitor":
                f"{s} monitoring is active and all alerts / notifications are working",
            "test":      f"{s} test completed; results are within acceptable limits on {d}",
        }
        return templates.get(action, f"{s} test case completed successfully on {d}")

    # ── Remaining shared helpers ───────────────────────────────────────────────

    def _determine_priority(self, title: str, description: str, steps: List[str]) -> str:
        """Determine priority from content"""
        text = f"{title} {description}".lower()

        critical_keywords = [
            'security', 'authentication', 'login', 'password', 'firmware',
            'critical', 'communication', 'enable', 'disable', 'modbus',
        ]
        high_keywords = [
            'configure', 'network', 'ethernet', 'discovery', 'pairing',
            'alarm', 'cloud', 'update', 'filter', 'important',
        ]
        medium_keywords = ['display', 'label', 'assign', 'navigation', 'interface']

        if any(kw in text for kw in critical_keywords):
            return "Critical"
        elif any(kw in text for kw in high_keywords):
            return "High"
        elif any(kw in text for kw in medium_keywords):
            return "Medium"
        return "High"

    def _build_acceptance_criteria(
        self, title: str, expected_results: List[str], final_result: str
    ) -> List[str]:
        """Build acceptance criteria from expected results"""
        criteria = list(expected_results[:3])
        if final_result:
            criteria.append(final_result)
        if not criteria:
            criteria = [f"{title} meets requirements"]
        return criteria

    def _apply_detail_level(
        self,
        steps: List[str],
        expected_results: List[str],
        preconditions: List[str],
        detail_level: DetailLevelEnum,
    ):
        """Adjust output detail based on detail level"""
        if detail_level == DetailLevelEnum.BASIC:
            return steps[:3], expected_results[:2], preconditions[:2]
        elif detail_level == DetailLevelEnum.COMPREHENSIVE:
            return steps, expected_results, preconditions
        else:  # DETAILED
            return steps[:8], expected_results[:6], preconditions[:5]
