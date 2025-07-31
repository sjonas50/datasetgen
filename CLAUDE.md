# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This repository implements an AI-first platform for automated dataset creation, transforming raw enterprise data into high-quality ML training datasets through intelligent processing, multi-modal fusion, and no-code interfaces.

## Current Architecture

### Backend (FastAPI)
```
backend/
├── api/               # REST API endpoints
│   └── v1/
│       └── endpoints/ # Resource endpoints (auth, datasets, pipelines, connectors)
├── core/              # Core functionality
│   ├── config.py      # Settings and environment configuration
│   ├── database.py    # Database connection and session management
│   ├── logging.py     # Structured logging setup
│   └── security.py    # Authentication and authorization
├── models/            # SQLAlchemy database models
│   ├── user.py        # User authentication model
│   ├── dataset.py     # Dataset metadata and storage
│   └── pipeline.py    # Pipeline configuration and execution history
├── schemas/           # Pydantic schemas for validation
├── services/          # Business logic layer
│   ├── dataset_service.py
│   ├── pipeline_service.py
│   └── connector_service.py
└── main.py           # FastAPI application entry point
```

### Frontend (Next.js + TypeScript)
```
frontend/
├── src/
│   ├── pages/         # Next.js pages (datasets, pipelines, connectors)
│   ├── components/    # Reusable React components
│   ├── services/      # API client services
│   │   └── api.ts     # Axios-based API client
│   └── styles/        # Global styles and Tailwind CSS
├── package.json       # Dependencies (Next.js 14, Ant Design, React Flow)
└── tsconfig.json      # TypeScript configuration
```

### Infrastructure
- **Docker Compose**: Multi-container development environment
- **PostgreSQL 15**: Primary database with pgvector extension ready
- **Redis 7**: Caching and session management
- **Nginx**: Reverse proxy for production

## Implementation Status

### ✅ Completed
- Project structure and development environment
- FastAPI backend with async SQLAlchemy
- Authentication system with JWT tokens
- RESTful API for datasets, pipelines, and connectors
- Database schema for core entities
- Next.js frontend with Ant Design UI
- Basic CRUD operations for all resources
- Docker development environment
- Intelligent CSV/JSON file connectors with LLM enhancement
- LLM integration services (Claude Sonnet 4, OpenAI, Gemini)
- DocETL-inspired pipeline execution engine with Celery
- Pipeline optimizer with 6 optimization strategies
- Multi-modal PDF processing with Claude Vision
- Image analysis for charts, tables, and dashboards
- Multi-modal data fusion engine (MDF Framework)
- Document parser supporting multiple formats

### 🚧 In Progress
- Monitoring and cost tracking dashboard

### 📋 Planned
- Federated learning support
- Enterprise security features (SSO, audit logs)
- Production deployment configurations
- Real-time streaming data support
- Advanced anomaly detection

## Advanced Features Implementation

### Claude Sonnet 4 Integration
The platform will leverage Claude Sonnet 4 (`claude-sonnet-4-20250514`) for:
- **Extended Thinking**: Complex schema inference and data transformation logic
- **Vision Capabilities**: Direct processing of PDFs, images, and charts
- **Tool Use**: Database queries and API calls during data analysis
- **200K Context Window**: Processing large datasets in single passes
- **Interleaved Thinking**: Reasoning between tool calls for optimal processing

### Key Advanced Features

1. **Intelligent Data Understanding**
   - Automatic schema inference with semantic understanding
   - Data quality assessment and remediation suggestions
   - Natural language dataset descriptions

2. **Multi-Modal Processing**
   - PDF table extraction using vision capabilities
   - Image-based data extraction (charts, diagrams)
   - Audio transcription and analysis
   - Video frame analysis for data extraction

3. **DocETL-Inspired Pipelines**
   - Declarative YAML/JSON pipeline definitions
   - Agent-based optimization for efficiency
   - Self-healing pipeline execution
   - Automatic parallelization and caching

4. **No-Code Pipeline Builder**
   - Visual drag-and-drop interface with React Flow
   - Natural language pipeline creation
   - Real-time preview and validation
   - Template library for common patterns

5. **Enterprise Security**
   - PII detection and automatic masking
   - Synthetic data generation for privacy
   - Comprehensive audit logging
   - Role-based access control (RBAC)

See `docs/IMPLEMENTATION_PLAN.md` for detailed implementation roadmap.

## Automated Quality Validation System

### Quality Checks Performed
1. **Missing Values**: Detection and severity assessment
2. **Duplicates**: Full row and key column duplicate detection
3. **Outliers**: Statistical outlier detection (Z-score and IQR methods)
4. **Format Consistency**: Date formats, string casing, whitespace
5. **Data Type Validation**: Schema compliance and type mismatches
6. **PII Detection**: SSN, phone, email, credit card, IP addresses
7. **Business Rules**: Custom validation rules (ranges, patterns)
8. **LLM Deep Analysis**: Semantic inconsistencies and anomalies

### Quality Scoring
- **Overall Score**: 0-100 based on issue severity and data completeness
- **Column Scores**: Individual quality scores per column
- **Severity Levels**: Critical, High, Medium, Low, Info
- **Actionable Recommendations**: LLM-generated improvement suggestions

### Automated Cleaning Operations
- Remove duplicates
- Strip whitespace
- Standardize casing
- Handle missing values (drop/fill strategies)
- Fix data types
- Remove outliers
- Mask PII data

### Example Quality Pipeline
```yaml
steps:
  - name: "quality_check"
    type: "quality_validation"
    config:
      schema:
        required_columns: ["id", "date", "amount"]
        column_types:
          amount: "float"
          date: "datetime"
      business_rules:
        - type: "range"
          column: "amount"
          min: 0
          max: 100000
      min_quality_score: 85.0
      fail_on_critical: true
```

## Multi-Modal Processing Capabilities

### PDF Processing
- **Table Extraction**: High-accuracy table extraction using Claude Vision
- **Form Data**: Extract data from fillable and scanned forms
- **Charts & Graphs**: Extract data points from embedded visualizations
- **Structured Data**: Automatic key-value pair extraction

### Image Analysis
- **Chart Data Extraction**: Convert charts to structured datasets
- **Table Detection**: Extract tabular data from images
- **Dashboard Analysis**: Extract KPIs and metrics from screenshots
- **Text Region Detection**: Identify and extract text from specific regions

### Data Fusion Strategies
1. **Early Fusion**: Combine raw features before processing
2. **Late Fusion**: Process independently, then combine
3. **Hybrid Fusion**: Strategic combination of early and late
4. **Hierarchical Fusion**: Multi-level fusion with intermediate representations
5. **Attention-Based Fusion**: Dynamic weighting of modality contributions

### Pipeline Steps Available

#### Multi-Modal Processing
- `document_parse`: Parse various document formats
- `pdf_table_extract`: Extract tables from PDFs using Claude Vision
- `image_data_extract`: Extract data from images, charts, and dashboards
- `multimodal_fusion`: Fuse data from multiple modalities
- `document_quality_check`: Assess extraction quality

#### Quality Validation & Cleaning
- `quality_validation`: Comprehensive data quality checks with LLM analysis
- `data_cleaning`: Automated cleaning based on quality issues
- `pii_detection`: Detect and mask personally identifiable information
- `data_profiling`: Generate detailed data profiles and statistics

## LLM Service Architecture

### Claude Sonnet 4 Configuration
```python
# Model configuration
CLAUDE_MODEL = "claude-sonnet-4-20250514"
CLAUDE_MAX_TOKENS = 64000
CLAUDE_CONTEXT_WINDOW = 200000

# API endpoints
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
```

### Service Integration Pattern
```python
# Example usage in pipeline
async def process_with_claude(data):
    claude_service = ClaudeService()
    
    # Use extended thinking for complex tasks
    schema = await claude_service.infer_schema_extended(data)
    
    # Use vision for document processing
    if file_type in ['pdf', 'png', 'jpg']:
        content = await claude_service.analyze_with_vision(file_path)
    
    # Generate transformation code
    transform_code = await claude_service.generate_transformation_code(
        source_schema, target_schema
    )
```

## Common Development Commands

### Quick Start
```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop all services
docker-compose down
```

### Backend Development
```bash
cd backend

# Install dependencies
pip install -r requirements.txt

# Run database migrations
alembic upgrade head

# Start development server
uvicorn main:app --reload --port 8000

# Run tests
pytest

# Format code
black .

# Lint code
flake8 .

# Run specific test module
pytest tests/test_multimodal.py -v

# Check type hints
mypy .
```

### Frontend Development
```bash
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev

# Build for production
npm run build

# Run tests
npm test

# Lint code
npm run lint
```

### Database Operations
```bash
# Access PostgreSQL shell
docker-compose exec postgres psql -U platform_user -d dataset_platform

# Create new migration
cd backend
alembic revision --autogenerate -m "Description of changes"

# Apply migrations
alembic upgrade head

# Rollback migration
alembic downgrade -1
```

## Future Development Guidance

When implementing the platform described in the research document, consider:

1. **Technology Stack Selection**: The research recommends GPT-4o or Gemini 2.5 for core data analysis, with LLaMA 4 for on-premises deployments
2. **Architecture Pattern**: Follow the layered architecture approach with DocETL-inspired declarative pipelines
3. **Security First**: Implement Zero Trust Architecture with comprehensive compliance coverage (GDPR, HIPAA, SOC2)
4. **Multi-modal Focus**: Use the MDF Framework's fusion taxonomy for handling diverse data types
5. **User Experience**: Prioritize no-code interfaces with visual pipeline builders and natural language queries

## Implementation Phases

The research outlines a 4-phase implementation roadmap:
- Phase 1: Cloud AutoML baselines and security frameworks
- Phase 2: Open-source tool integration and custom pipelines
- Phase 3: Advanced multi-modal fusion and LLM-assisted feature engineering
- Phase 4: Production deployment with automated monitoring

When code is added to this repository, update this file with build commands, testing procedures, and architectural decisions.