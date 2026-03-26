# AI Test Case Generator

An intelligent, AI-powered application that automatically generates comprehensive, data-driven test cases from requirements documents and specifications.

## 🎯 Project Overview

Instead of creating generic test templates, the AI Test Case Generator intelligently analyzes input documents to extract specific data patterns, business rules, workflows, and constraints, then generates test cases tailored to the actual content.

### Key Features

✅ **Intelligent Document Analysis** - Extract business rules, workflows, constraints, and data entities from documents
✅ **AI-Powered Test Case Generation** - Generate relevant, specific test cases based on extracted information
✅ **Multiple Input Formats** - Support for TXT, PDF, DOCX, MD, CSV
✅ **Test Type Variety** - API, Security, Performance, Usability, Integration, Unit Testing
✅ **Configurable Output** - Control detail levels and number of test cases
✅ **Easy Export** - Download as CSV or JSON
✅ **Modern UI** - Professional, intuitive interface

## 🏗️ Project Structure

```
ai-test-case-generator/
├── backend/                    # Python FastAPI backend
│   ├── app/
│   │   ├── main.py            # FastAPI application
│   │   ├── models/            # Data models
│   │   ├── routes/            # API endpoints
│   │   ├── ml/                # AI/ML modules
│   │   └── utils/             # Utilities
│   ├── requirements.txt
│   └── README.md
│
├── frontend/                   # React frontend
│   ├── src/
│   │   ├── components/        # React components
│   │   ├── App.jsx
│   │   └── index.css
│   ├── package.json
│   ├── vite.config.js
│   └── README.md
│
└── README.md
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Node.js 16+
- npm or yarn

### Backend Setup

1. Navigate to backend directory:
```bash
cd backend
```

2. Create virtual environment:
```bash
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Mac/Linux
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the backend:
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Backend API available at: http://localhost:8000
API Docs: http://localhost:8000/docs

### Frontend Setup

1. Navigate to frontend directory:
```bash
cd frontend
```

2. Install dependencies:
```bash
npm install
```

3. Start development server:
```bash
npm run dev
```

Frontend available at: http://localhost:3000

## 📖 How It Works

### 1. **Document Upload**
   - Upload a document (TXT, PDF, DOCX, MD, CSV) or paste text content
   - System processes and analyzes the content

### 2. **AI Analysis**
   - NLP analyzer extracts:
     - Business rules
     - Workflows
     - Constraints
     - Data entities
     - Keywords and patterns

### 3. **Test Case Generation**
   - Trained AI model generating test cases based on:
     - Document content analysis
     - Selected test types
     - Configured detail level
     - Pattern matching with training data

### 4. **Result Display**
   - View generated test cases with:
     - Detailed steps
     - Expected results
     - Acceptance criteria
     - Priority levels

### 5. **Export**
   - Download test cases as CSV or JSON
   - Use in your testing tools

## 🧠 AI Model Details

### Training Data
The model is initialized with 6 example test cases covering:
- API Testing
- Security Testing
- Usability Testing
- Performance Testing
- Integration Testing
- Unit Testing

### How It Learns
1. **Pattern Analysis** - Analyzes document content for keywords and patterns
2. **Category Detection** - Identifies relevant testing categories
3. **Template Matching** - Matches extracted information to appropriate test case templates
4. **Customization** - Adapts templates with specific document content

### Extending the Model
To improve accuracy, extend `backend/app/ml/training_data.py` with more examples:

```python
def get_training_data():
    return [
        {
            "type": "Your Test Type",
            "title": "Test Title",
            "description": "Test Description",
            # ... more fields
        }
    ]
```

## 💾 API Endpoints

### Documents
- `POST /api/documents/upload` - Upload file
- `POST /api/documents/paste` - Process pasted text

### Test Cases
- `POST /api/test-cases/generate` - Generate test cases
- `GET /api/test-cases/types` - Get available test types

## 📋 Example Usage

### Request
```json
{
  "document_content": "User authentication system...",
  "test_types": ["Security Testing", "API Testing"],
  "detail_level": "Detailed",
  "num_test_cases": 5
}
```

### Response
```json
{
  "test_cases": [
    {
      "id": "TC-001",
      "test_type": "Security Testing",
      "title": "Security Testing: Authentication & Authorization",
      "description": "...",
      "objective": "...",
      "steps": [...],
      "expected_results": [...],
      "priority": "Critical"
    }
  ]
}
```

## 🛠️ Technology Stack

### Backend
- **FastAPI** - Modern web framework
- **Pydantic** - Data validation
- **NLP Tools** - spaCy, NLTK for text analysis
- **PyPDF2, python-docx** - Document processing

### Frontend
- **React 18** - UI library
- **Vite** - Build tool
- **CSS3** - Styling

## 📝 Configuration

### Backend Configuration
Modify `backend/app/utils/file_handler.py` to add support for more file types:

```python
async def process_uploaded_file(file: UploadFile) -> str:
    # Add your file type handling
```

### Frontend Configuration
Update API endpoint in components if needed (default: http://localhost:8000)

## 🤝 Contributing

To enhance the model:
1. Add more training examples in `training_data.py`
2. Improve NLP analysis in `nlp_analyzer.py`
3. Enhance test case generation logic in `test_case_generator.py`

## 📄 License

MIT License

## 🎓 Learning Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [React Documentation](https://react.dev/)
- [Pydantic Documentation](https://docs.pydantic.dev/)
- [spaCy Documentation](https://spacy.io/)

## ✨ Future Enhancements

- [ ] Database integration for storing test cases
- [ ] User authentication and project management
- [ ] Advanced NLP with transformer models
- [ ] Integration with popular test frameworks
- [ ] CI/CD pipeline integration
- [ ] Collaborative features
- [ ] Test case versioning and history
- [ ] Analytics dashboard

## 📞 Support

For issues, questions, or suggestions, please create an issue in the repository.

---

**Built with ❤️ using FastAPI and React**
