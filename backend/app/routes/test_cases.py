"""Test case generation routes"""
from fastapi import APIRouter, HTTPException
from app.models.schemas import (
    GenerateTestCasesRequest, GenerateTestCasesResponse,
)
from app.ml.test_case_generator import TestCaseGenerator
from app.ml.custom_trainer import CustomTrainer

router = APIRouter()
generator = TestCaseGenerator()
feedback_trainer = CustomTrainer()


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
