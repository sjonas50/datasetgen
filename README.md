# DatasetGen - AI-First Dataset Creation Platform

Transform any document into high-quality ML training datasets with just a few clicks. DatasetGen uses Claude Sonnet 4 to intelligently process PDFs, Word docs, images, and more - automatically generating Q&A pairs, classification data, NER datasets, and custom formats tailored to your needs.

## 🚀 Key Features

### 🎯 User-Friendly Dataset Generation
- **One-Click Generation**: Upload files → Choose dataset type → Generate
- **Smart Document Processing**: Handles scanned PDFs, images, Word docs, CSVs, and more
- **Multiple Dataset Types**: 
  - Q&A Pairs for chatbots and FAQ systems
  - Text Classification for sentiment analysis
  - Named Entity Recognition (NER) for information extraction
  - Summarization pairs for abstractive models
  - Custom formats based on your requirements

### 🧠 Powered by Claude Sonnet 4
- **Latest AI Model**: Uses Claude Sonnet 4 (claude-sonnet-4-20250514)
- **Vision Capabilities**: Extracts content from images and scanned documents
- **Intelligent Generation**: Creates contextually appropriate training data
- **High-Quality Output**: Generates diverse, realistic examples

### 📊 Two Interfaces
- **Dataset Wizard**: Simple UI for non-technical users - just click and generate
- **Pipeline Builder**: Advanced visual interface for complex workflows
- **Real-Time Monitoring**: Track generation progress and results

### 🔒 Enterprise Ready
- **Authentication**: Secure JWT-based access control  
- **SQLite Storage**: Persistent data storage (upgradeable to PostgreSQL)
- **File Support**: PDF, DOCX, TXT, CSV, JSON, PNG, JPG
- **Export Options**: Download datasets as CSV or JSON

## 🏗️ Architecture

```
┌──────────────────────────────────────────────┐
│          Frontend (Next.js)                  │
│   • Dataset Wizard (Simple UI)               │
│   • Pipeline Builder (Advanced)              │
│   • File Upload & Management                 │
├──────────────────────────────────────────────┤
│        Backend (FastAPI)                     │
│   • REST API                                 │
│   • Authentication (JWT)                     │
│   • File Processing                          │
├──────────────────────────────────────────────┤
│     AI Layer (Claude Sonnet 4)               │
│   • Document Understanding                   │
│   • Vision Processing                        │
│   • Dataset Generation                       │
├──────────────────────────────────────────────┤
│         Data Storage                         │
│   • SQLite Database                          │
│   • Local File Storage                       │
└──────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Node.js 18+
- Poppler (for PDF processing): `brew install poppler` (macOS) or `apt-get install poppler-utils` (Linux)

### Setup Instructions

1. **Clone the repository:**
```bash
git clone https://github.com/yourorg/dataset-gen.git
cd dataset-gen
```

2. **Set up environment variables:**
Create a `.env` file in the root directory:
```env
# Required - Get your key from https://console.anthropic.com
ANTHROPIC_API_KEY=your_anthropic_api_key

# Generate a secret key
SECRET_KEY=your-secret-key-here
```

3. **Install backend dependencies:**
```bash
cd backend
pip install -r requirements.txt
```

4. **Install frontend dependencies:**
```bash
cd ../frontend
npm install
```

5. **Start the application:**

In one terminal, start the backend:
```bash
cd backend
python main_full.py
```

In another terminal, start the frontend:
```bash
cd frontend
npm run dev
```

6. **Access the platform:**
- 🌐 Frontend: http://localhost:3000
- 🔧 Backend API: http://localhost:8000
- 📚 API Documentation: http://localhost:8000/docs

7. **Create an account and start generating datasets!**

## 📁 Project Structure

```
dataset-gen/
├── backend/                      # FastAPI backend
│   ├── api/v1/endpoints/        # REST API endpoints
│   ├── core/                    # Core configuration
│   ├── models/                  # SQLAlchemy models
│   ├── services/               
│   │   ├── llm/                # LLM integrations (Claude, GPT, Gemini)
│   │   ├── pipeline/           # Pipeline execution & optimization
│   │   ├── connectors/         # Data source connectors
│   │   ├── multimodal/         # PDF, image processing
│   │   ├── quality/            # Validation & PII detection
│   │   └── monitoring/         # Metrics collection
│   ├── schemas/                # Pydantic schemas
│   └── tests/                  # Test suites
├── frontend/                    # Next.js frontend
│   ├── src/
│   │   ├── components/         # React components
│   │   │   ├── PipelineBuilder/  # Visual pipeline builder
│   │   │   └── charts/           # Monitoring charts
│   │   ├── pages/             # Next.js pages
│   │   ├── services/          # API clients
│   │   └── styles/            # CSS modules
│   └── public/                # Static assets
├── infrastructure/            
│   ├── docker/               # Dockerfiles
│   └── nginx/                # Nginx configuration
├── examples/                 # Example pipelines
├── docs/                     # Documentation
└── docker-compose.yml        # Container orchestration
```

## 💻 Development

### Manual Setup (without Docker)

**Backend:**
```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Setup database
alembic upgrade head

# Start backend
uvicorn main:app --reload --port 8000
```

**Frontend:**
```bash
cd frontend
npm install
npm run dev
```

**Start required services:**
```bash
# PostgreSQL
docker run -d -p 5432:5432 -e POSTGRES_PASSWORD=platform_pass postgres:15

# Redis
docker run -d -p 6379:6379 redis:7-alpine
```

### Running Tests

```bash
# Backend tests
cd backend
pytest -v
pytest --cov=services tests/  # With coverage

# Frontend tests
cd frontend
npm test
```

## 📊 Usage Examples

### Creating a Pipeline via API

```python
import requests

# Create a pipeline
pipeline = {
    "name": "Customer Data Quality Pipeline",
    "description": "Validate and clean customer data",
    "config": {
        "steps": [
            {
                "type": "quality_validation",
                "config": {
                    "deep_analysis": True,
                    "min_quality_score": 85.0
                }
            },
            {
                "type": "pii_detection",
                "config": {
                    "action": "mask"
                }
            }
        ]
    }
}

response = requests.post(
    "http://localhost:8000/api/v1/pipelines",
    json=pipeline,
    headers={"Authorization": f"Bearer {token}"}
)
```

### Visual Pipeline Builder

1. Navigate to `/pipeline-builder`
2. Drag nodes from the sidebar
3. Connect nodes to create data flow
4. Configure each step's parameters
5. Save and execute the pipeline

## 🔧 Configuration

### Pipeline Optimization Settings

```yaml
optimization:
  enable_caching: true
  cache_ttl: 3600
  enable_parallelization: true
  enable_filter_pushdown: true
  enable_projection_pushdown: true
  batch_size: 1000
```

### LLM Configuration

```python
llm_config:
  provider: "claude"
  model: "claude-sonnet-4-20250514"
  temperature: 0.1
  use_extended_thinking: true
  max_tokens: 4096
```

## 🛡️ Security Features

- **Authentication**: JWT-based with refresh tokens
- **Authorization**: Role-based access control (RBAC)
- **Data Protection**: Encryption at rest and in transit
- **PII Handling**: Automatic detection and masking
- **Audit Logging**: Complete activity tracking
- **API Rate Limiting**: Configurable per endpoint

## 📈 Performance

- **30x Efficiency**: DocETL-inspired optimization
- **Caching**: Redis-based LLM response caching
- **Parallel Processing**: Distributed pipeline execution
- **Resource Management**: Automatic scaling and throttling

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](docs/CONTRIBUTING.md) for details.

### Development Workflow

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- DocETL framework for pipeline optimization concepts
- MDF Framework for multi-modal fusion strategies
- Claude Sonnet 4 for advanced AI capabilities

## 📞 Support

- 📧 Email: support@datasetgen.ai
- 💬 Discord: [Join our community](https://discord.gg/datasetgen)
- 📚 Documentation: [docs.datasetgen.ai](https://docs.datasetgen.ai)
- 🐛 Issues: [GitHub Issues](https://github.com/yourorg/dataset-gen/issues)