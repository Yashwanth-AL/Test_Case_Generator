<!-- Copilot Instructions for AI Test Case Generator Project -->

# AI Test Case Generator - Development Instructions

## Project Overview
This is an AI-powered Test Case Generator that analyzes documents and generates comprehensive test cases using a trained ML model.

## Tech Stack
- **Backend**: Python, FastAPI
- **Frontend**: React, Vite
- **ML**: Local trained model with NLP analysis
- **Database**: None (MVP - in-memory)

## Key Principles
1. **No External APIs for AI** - Uses local trained model with NLP analysis
2. **Document Intelligence** - Extracts actual business rules and workflows
3. **Trainable Model** - Can be extended with more examples
4. **Separation of Concerns** - Backend handles AI, Frontend handles UI

## Project Structure

### Backend (`backend/`)
```
app/
├── main.py              # FastAPI setup and routing
├── models/
│   └── schemas.py       # Pydantic models for validation
├── routes/
│   ├── documents.py     # File upload/paste endpoints
│   └── test_cases.py    # Test case generation endpoints
├── ml/
│   ├── test_case_generator.py    # Main generator logic
│   ├── nlp_analyzer.py           # Document analysis
│   ├── model_trainer.py          # Model training/loading
│   └── training_data.py          # Training dataset
└── utils/
    └── file_handler.py  # File processing utilities
```

### Frontend (`frontend/`)
```
src/
├── components/
│   ├── DocumentUpload.jsx     # File/text input
│   ├── TestCaseGenerator.jsx  # Configuration
│   └── TestCaseDisplay.jsx    # Results
├── App.jsx              # Main app logic
├── services/            # API services (future)
└── styles/              # CSS files
```

## Development Workflow

### Starting the Application
1. **Backend**:
   ```bash
   cd backend
   python -m venv venv
   venv\Scripts\activate
   pip install -r requirements.txt
   uvicorn app.main:app --reload
   ```

2. **Frontend**:
   ```bash
   cd frontend
   npm install
   npm run dev
   ```

3. **Access**: http://localhost:3000

### Adding Features

#### Model Improvements
1. Add more training examples to `backend/app/ml/training_data.py`
2. Enhance NLP analysis in `backend/app/ml/nlp_analyzer.py`
3. Improve pattern matching in `backend/app/ml/model_trainer.py`

#### New Test Types
1. Add to `training_data.py` with examples
2. Update enum in `backend/app/models/schemas.py`
3. Update frontend dropdown in `DocumentUpload.jsx`

#### File Format Support
1. Add handler in `backend/app/utils/file_handler.py`
2. Update requirements.txt if new dependencies needed

### Important Files to Modify

#### For Better Test Case Generation
- `backend/app/ml/training_data.py` - Main training dataset
- `backend/app/ml/nlp_analyzer.py` - Document analysis logic
- `backend/app/ml/test_case_generator.py` - Generation algorithm

#### For Frontend Improvements
- `frontend/src/App.jsx` - Main logic
- `frontend/src/components/*.jsx` - UI components
- `frontend/.css` files - Styling

## API Contracts

### Generate Test Cases
```json
POST /api/test-cases/generate
{
  "document_content": "string",
  "test_types": ["API Testing", "Security Testing"],
  "detail_level": "Detailed|Basic|Comprehensive",
  "num_test_cases": 10
}

Response {
  "test_cases": [TestCase],
  "summary": "string",
  "total_generated": number
}
```

### Test Case Structure
```json
{
  "id": "TC-001",
  "test_type": "API Testing",
  "title": "string",
  "description": "string",
  "objective": "string",
  "preconditions": ["string"],
  "steps": ["string"],
  "expected_results": ["string"],
  "priority": "Critical|High|Medium|Low",
  "acceptance_criteria": ["string"]
}
```

## Common Tasks

### Testing the Backend
```bash
cd backend
# API docs: http://localhost:8000/docs
# Swagger UI for testing endpoints
```

### Building Frontend
```bash
cd frontend
npm run build
npm run preview
```

### Debugging

#### Backend Issues
1. Check CORS configuration in `app/main.py`
2. Verify requirements installed: `pip list`
3. Check API logs in terminal

#### Frontend Issues
1. Browser DevTools (F12)
2. Check proxy in `vite.config.js`
3. Verify components are importing correctly

## Performance Considerations
- Model loads once on startup
- NLP analysis uses regex (lightweight)
- No database queries (MVP)
- Export generation is fast

## Security Considerations
- File upload validation
- Input sanitization
- File type restrictions
- No sensitive data storage

## Next Steps for Enhancement
1. Database integration
2. User authentication
3. Advanced NLP (transformers)
4. More specialized test types
5. Batch processing
6. Analytics dashboard

---

**When making changes, ensure both backend and frontend are kept in sync with schemas and types.**
