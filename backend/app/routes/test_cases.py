"""Test case generation routes"""
from fastapi import APIRouter, HTTPException
from app.models.schemas import GenerateTestCasesRequest, GenerateTestCasesResponse
from app.ml.test_case_generator import TestCaseGenerator

router = APIRouter()
generator = TestCaseGenerator()


@router.post("/generate", response_model=GenerateTestCasesResponse)
async def generate_test_cases(request: GenerateTestCasesRequest):
    """Generate test cases from document content"""
    try:
        test_cases = generator.generate(
            document_content=request.document_content,
            test_types=request.test_types,
            detail_level=request.detail_level,
            num_test_cases=request.num_test_cases,
            id_prefix=request.id_prefix,
        )

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
