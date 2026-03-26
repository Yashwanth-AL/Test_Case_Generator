"""Test case generation routes"""
from fastapi import APIRouter, HTTPException
from app.models.schemas import (
    GenerateTestCasesRequest, GenerateTestCasesResponse,
    FeedbackRequest, FeedbackResponse,
)
from app.ml.test_case_generator import TestCaseGenerator
from app.ml.custom_trainer import CustomTrainer
from datetime import datetime
import json, os

router = APIRouter()
generator = TestCaseGenerator()
feedback_trainer = CustomTrainer()

FEEDBACK_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "feedback_store.json"
)


def _load_feedback():
    try:
        with open(FEEDBACK_PATH, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {"feedback": []}


def _save_feedback(data):
    os.makedirs(os.path.dirname(FEEDBACK_PATH), exist_ok=True)
    with open(FEEDBACK_PATH, "w") as f:
        json.dump(data, f, indent=2)


def _build_case_from_feedback(generated_case, comment):
    """Create a corrected-style case using user comment guidance."""
    if not generated_case:
        return None

    case = dict(generated_case)
    comment_text = (comment or "").strip()
    if not comment_text:
        return case

    current_desc = case.get("description", "")
    case["description"] = (
        f"{current_desc} User feedback preference: {comment_text}".strip()
    )

    acceptance = case.get("acceptance_criteria", [])
    if not isinstance(acceptance, list):
        acceptance = []
    acceptance.append(f"User feedback addressed: {comment_text}")
    case["acceptance_criteria"] = acceptance[:8]

    return case


@router.post("/generate", response_model=GenerateTestCasesResponse)
async def generate_test_cases(request: GenerateTestCasesRequest):
    """Generate test cases from document content"""
    try:
        test_cases = generator.generate(
            document_content=request.document_content,
            user_prompt=request.user_prompt,
            test_types=request.test_types,
            detail_level=request.detail_level,
            num_test_cases=request.num_test_cases,
            id_prefix=request.id_prefix,
        )

        # Passive backend training: learn from each successful generation.
        # This keeps adaptation local and improves future prompt relevance.
        try:
            if test_cases:
                tc_payload = [tc.model_dump() for tc in test_cases[:5]]
                tags = ["auto_training", "generation"]
                if request.user_prompt:
                    tags.append("prompted")
                feedback_trainer.add_example(
                    document_content=request.document_content[:2000],
                    test_cases=tc_payload,
                    tags=tags,
                )
        except Exception:
            # Training should never block generation response.
            pass

        return GenerateTestCasesResponse(
            test_cases=test_cases,
            summary=f"Generated {len(test_cases)} test cases from document",
            total_generated=len(test_cases)
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(request: FeedbackRequest):
    """Submit feedback on a generated test case to improve future generation"""
    try:
        store = _load_feedback()
        feedback_id = f"FB-{len(store['feedback']) + 1:04d}"

        entry = {
            "id": feedback_id,
            "test_case_id": request.test_case_id,
            "document_content": request.document_content[:500],
            "user_prompt": request.user_prompt,
            "rating": request.rating,
            "comment": request.comment,
            "generated_test_case": request.generated_test_case,
            "corrected_test_case": request.corrected_test_case,
            "created_at": datetime.now().isoformat(),
        }
        store["feedback"].append(entry)
        _save_feedback(store)

        # Highest-priority learning source: explicit corrected test case
        if request.corrected_test_case:
            feedback_trainer.add_example(
                document_content=request.document_content[:500],
                test_cases=[request.corrected_test_case],
                tags=["user_feedback", "corrected", f"rating_{request.rating}"],
            )

        # Positive reinforcement: learn from cases users rated highly
        elif request.generated_test_case and request.rating >= 4:
            feedback_trainer.add_example(
                document_content=request.document_content[:500],
                test_cases=[request.generated_test_case],
                tags=["user_feedback", "positive", f"rating_{request.rating}"],
            )

        # Low rating with comment: synthesize a correction-oriented example
        elif request.generated_test_case and request.rating <= 2 and request.comment:
            corrected_like_case = _build_case_from_feedback(
                request.generated_test_case,
                request.comment,
            )
            if corrected_like_case:
                feedback_trainer.add_example(
                    document_content=request.document_content[:500],
                    test_cases=[corrected_like_case],
                    tags=["user_feedback", "correction_hint", f"rating_{request.rating}"],
                )

        return FeedbackResponse(
            status="success",
            message="Feedback recorded. Thank you!",
            feedback_id=feedback_id,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/types")
async def get_test_types():
    """Get available test types"""
    return {
        "types": [
            "Commissioning",
            "API Testing",
            "Security Testing",
            "Usability Testing",
            "Performance Testing",
            "Integration Testing",
            "Unit Testing",
            "Functional Testing",
            "Configuration Testing",
        ]
    }
