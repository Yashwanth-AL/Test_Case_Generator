"""API routes for model training"""
from fastapi import APIRouter, HTTPException
from typing import Optional, List
from pydantic import BaseModel
from app.ml.custom_trainer import CustomTrainer


# Initialize custom trainer
custom_trainer = CustomTrainer()

router = APIRouter(prefix="/api/training", tags=["Training"])


# Request/Response Models
class TestCaseInput(BaseModel):
    """Test case input model"""
    test_type: str
    title: str
    description: str
    objective: str
    preconditions: List[str]
    steps: List[str]
    expected_results: List[str]
    priority: str
    acceptance_criteria: List[str]


class TrainingExampleInput(BaseModel):
    """Request to add a training example"""
    document_content: str
    test_cases: List[TestCaseInput]
    tags: Optional[List[str]] = None


class TrainingExampleResponse(BaseModel):
    """Response with training example details"""
    id: str
    document_content: str
    test_case_count: int
    extracted_keywords: dict
    created_at: str


class LearnedPatternResponse(BaseModel):
    """Response with learned pattern details"""
    category: str
    test_case_types: dict
    frequency: int
    examples: List[str]


class TrainingStatsResponse(BaseModel):
    """Response with training statistics"""
    total_examples: int
    total_patterns: int
    total_test_cases_trained: int
    test_type_distribution: dict
    categories_learned: List[str]
    last_updated: str


class RecommendationResponse(BaseModel):
    """Response with test case recommendations"""
    test_types: dict
    confidence: float
    learned_from_examples: int


# Endpoints

@router.post("/add-example", response_model=TrainingExampleResponse)
async def add_training_example(example: TrainingExampleInput):
    """
    Add a training example for the model to learn from
    
    The model will learn patterns from your document and test cases,
    improving its ability to generate similar test cases for similar documents.
    """
    try:
        # Convert to dict format
        test_cases_dict = [tc.dict() for tc in example.test_cases]
        
        result = custom_trainer.add_example(
            document_content=example.document_content,
            test_cases=test_cases_dict,
            tags=example.tags or []
        )
        
        return TrainingExampleResponse(
            id=result["id"],
            document_content=result["document_content"][:200] + "...",  # Truncate for response
            test_case_count=result["test_case_count"],
            extracted_keywords=result["extracted_keywords"],
            created_at=result["created_at"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to add training example: {str(e)}")


@router.get("/examples", response_model=List[TrainingExampleResponse])
async def get_all_examples():
    """Get all training examples"""
    try:
        examples = custom_trainer.get_all_examples()
        return [
            TrainingExampleResponse(
                id=e["id"],
                document_content=e["document_content"][:200] + "...",
                test_case_count=e["test_case_count"],
                extracted_keywords=e["extracted_keywords"],
                created_at=e["created_at"]
            )
            for e in examples
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get examples: {str(e)}")


@router.get("/examples-by-tag/{tag}", response_model=List[TrainingExampleResponse])
async def get_examples_by_tag(tag: str):
    """Get training examples filtered by tag"""
    try:
        examples = custom_trainer.get_examples_by_tag(tag)
        return [
            TrainingExampleResponse(
                id=e["id"],
                document_content=e["document_content"][:200] + "...",
                test_case_count=e["test_case_count"],
                extracted_keywords=e["extracted_keywords"],
                created_at=e["created_at"]
            )
            for e in examples
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get examples: {str(e)}")


@router.get("/patterns", response_model=List[LearnedPatternResponse])
async def get_learned_patterns():
    """Get all learned patterns from training examples"""
    try:
        patterns = custom_trainer.get_learned_patterns()
        return patterns
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get patterns: {str(e)}")


@router.get("/patterns/{category}", response_model=LearnedPatternResponse)
async def get_pattern_by_category(category: str):
    """Get learned pattern for a specific category"""
    try:
        pattern = custom_trainer.get_pattern_by_category(category)
        if not pattern:
            raise HTTPException(status_code=404, detail=f"No pattern found for category: {category}")
        return pattern
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get pattern: {str(e)}")


@router.post("/get-recommendations", response_model=RecommendationResponse)
async def get_recommendations(keywords: dict):
    """
    Get test case type recommendations based on document keywords
    
    This uses the learned patterns to recommend which test case types
    are most appropriate for your document.
    """
    try:
        recommendations = custom_trainer.get_recommendations(keywords)
        return RecommendationResponse(
            test_types=recommendations["test_types"],
            confidence=recommendations["confidence"],
            learned_from_examples=recommendations["learned_from_examples"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get recommendations: {str(e)}")


@router.get("/statistics", response_model=TrainingStatsResponse)
async def get_training_statistics():
    """Get training statistics and model health"""
    try:
        stats = custom_trainer.get_statistics()
        return TrainingStatsResponse(**stats)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get statistics: {str(e)}")


@router.get("/status")
async def get_training_status():
    """Get current training status"""
    try:
        stats = custom_trainer.get_statistics()
        return {
            "status": "ready",
            "examples_trained": stats["total_examples"],
            "patterns_learned": stats["total_patterns"],
            "test_cases_analyzed": stats["total_test_cases_trained"],
            "categories": stats["categories_learned"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get status: {str(e)}")
