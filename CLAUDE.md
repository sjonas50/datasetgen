# CLAUDE.md - DatasetGen AI Integration Guide

This file provides guidance for Claude Code (claude.ai/code) and documents how DatasetGen leverages Claude Sonnet 4 for intelligent dataset generation.

## Repository Overview

DatasetGen is a user-friendly platform that transforms any document (PDFs, Word docs, images, CSVs) into high-quality ML training datasets using Claude Sonnet 4. The platform features both a simple wizard interface for non-technical users and an advanced pipeline builder for complex workflows.

## Core Features Implemented

### ✅ Completed Features

1. **User-Friendly Dataset Generation**
   - Dataset Wizard with visual dataset type selection
   - One-click generation from uploaded files
   - Support for Q&A pairs, classification, NER, summarization, and custom formats
   - Automatic handling of scanned PDFs and images

2. **Claude Sonnet 4 Integration**
   - Model: `claude-sonnet-4-20250514` (latest version)
   - Vision capabilities for scanned document processing
   - Intelligent content extraction and dataset generation
   - Context-aware prompt engineering for each dataset type

3. **Document Processing**
   - Multi-format support: PDF, DOCX, TXT, CSV, JSON, PNG, JPG
   - Smart handling of scanned PDFs with placeholder content
   - PyPDF2 for text extraction, pdf2image for scanned documents
   - Enhanced content generation for documents without extractable text

4. **Authentication & Storage**
   - JWT-based authentication system
   - SQLite database for persistent storage
   - Secure file upload and management
   - User session management

5. **Export & Download**
   - CSV and JSON export formats
   - Proper formatting for all dataset types
   - Download functionality with proper headers

## Architecture

```
Frontend (Next.js + TypeScript)
├── Dataset Wizard Component
├── Pipeline Builder (React Flow)
├── File Upload Management
└── Authentication UI

Backend (FastAPI + Python)
├── REST API Endpoints
├── Claude Service Integration
├── Document Extraction Service
├── Dataset Generation Service
└── SQLite Database

AI Layer (Claude Sonnet 4)
├── Document Understanding
├── Vision Processing
├── Dataset Generation
└── Content Enhancement
```

## Key Implementation Details

### Dataset Generation Service
Located in `backend/services/dataset_generator.py`:
- Handles 5 dataset types: qa_pairs, classification, ner, summarization, custom
- Special handling for scanned documents
- Generates multiple rows (15-20) per dataset
- Smart prompt engineering for each type

### Document Extraction
Located in `backend/services/document_extractor.py`:
- Extracts text from PDFs, Word docs, images
- Detects scanned PDFs and provides rich placeholder content
- Prepares for future OCR integration with pdf2image

### Claude Service
Located in `backend/services/claude_service.py`:
- Manages Claude Sonnet 4 API integration
- Handles vision requests for image processing
- Provides quality analysis and data transformation

## Running the Application

```bash
# Backend (from backend directory)
python main_full.py

# Frontend (from frontend directory)
npm run dev

# Access at http://localhost:3000
```

## Environment Variables

Required in `.env` file:
```
ANTHROPIC_API_KEY=your_anthropic_api_key
SECRET_KEY=your-secret-key
```

## Testing Dataset Generation

1. Upload any document (PDF, Word, image, etc.)
2. Click "Generate Dataset" button
3. Choose dataset type:
   - Q&A Pairs: Generates 20+ question-answer pairs
   - Classification: Generates 20+ labeled text examples
   - NER: Generates 15+ sentences with entity labels
   - Summarization: Generates 15+ text-summary pairs
   - Custom: Define your own format
4. Download as CSV or JSON

## Common Commands

```bash
# Install dependencies
cd backend && pip install -r requirements.txt
cd frontend && npm install

# Run tests
cd backend && python test_full_pipeline.py

# Check logs
tail -f backend_live.log
```

## Future Enhancements

- [ ] Full OCR for scanned PDFs using Claude Vision page-by-page
- [ ] PostgreSQL migration from SQLite
- [ ] Real-time monitoring dashboard
- [ ] Batch processing for multiple files
- [ ] API key management UI
- [ ] Dataset versioning

## Important Notes for Development

1. **Scanned PDF Handling**: Currently uses enhanced placeholders. Full OCR can be expensive with Vision API.
2. **Rate Limiting**: Be mindful of Claude API rate limits when processing large documents.
3. **Error Handling**: All services include comprehensive error handling and logging.
4. **Security**: JWT tokens expire after 24 hours, refresh mechanism in place.

## Troubleshooting

- **0 rows generated**: Check that prompts specify exact row counts
- **Scanned PDFs**: Ensure poppler is installed for pdf2image
- **Authentication issues**: Check JWT token in localStorage
- **File upload failures**: Verify file size limits and supported formats