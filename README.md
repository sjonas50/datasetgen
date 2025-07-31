# DatasetGen - AI-First Dataset Creation Platform

An advanced platform that transforms raw enterprise data into high-quality ML training datasets through intelligent processing, multi-modal fusion, and visual pipeline building. Built with cutting-edge LLMs including Claude Sonnet 4, GPT-4, and Gemini, achieving 30x efficiency improvements through DocETL-inspired optimization.

## 🚀 Key Features

### Intelligence Layer
- **Claude Sonnet 4 Integration**: Extended thinking and vision capabilities for document understanding
- **Multi-LLM Support**: OpenAI GPT-4, Google Gemini, and Claude models
- **DocETL-Inspired Optimization**: 30x efficiency improvement in pipeline execution
- **Intelligent Schema Detection**: Automatic type inference and data understanding

### Data Processing
- **Multi-Modal Fusion**: MDF Framework implementation with 5 fusion strategies
- **Visual Pipeline Builder**: Drag-and-drop interface with React Flow
- **Quality Validation**: 10 types of automated checks with severity levels
- **PII Detection**: Built-in patterns for SSN, email, phone, credit cards, IP addresses
- **PDF Processing**: Table extraction using Claude Vision API

### Monitoring & Operations
- **Real-Time Dashboard**: Track pipeline executions, costs, and performance
- **Cost Tracking**: Detailed breakdown by LLM provider, pipeline, and resource type
- **Prometheus Metrics**: Production-ready monitoring integration
- **Distributed Execution**: Celery-based pipeline orchestration

### Enterprise Features
- **Security**: JWT authentication, role-based access control
- **Data Connectors**: CSV, JSON with auto-detection and LLM enhancement
- **Caching**: Redis-based caching for LLM responses
- **API Documentation**: Auto-generated OpenAPI/Swagger docs

## 🏗️ Architecture

```
┌──────────────────────────────────────────────┐
│     Frontend (Next.js + React Flow)          │
│   • Visual Pipeline Builder                  │
│   • Monitoring Dashboard                     │
│   • Dataset Management                       │
├──────────────────────────────────────────────┤
│        API Layer (FastAPI)                   │
│   • RESTful Endpoints                        │
│   • WebSocket Support                        │
│   • OpenAPI Documentation                    │
├──────────────────────────────────────────────┤
│     Intelligence Layer                       │
│   • Claude Sonnet 4 (Extended Thinking)      │
│   • GPT-4 / Gemini Integration              │
│   • DocETL Pipeline Optimizer               │
├──────────────────────────────────────────────┤
│     Processing Layer                         │
│   • Multi-Modal Fusion Engine               │
│   • Quality Validation System               │
│   • PII Detection & Masking                │
├──────────────────────────────────────────────┤
│     Data Layer                              │
│   • PostgreSQL (Metadata)                   │
│   • Redis (Caching & Metrics)              │
│   • S3/Local Storage (Datasets)            │
└──────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Docker and Docker Compose (recommended)
- Or manually install:
  - Python 3.11+
  - Node.js 18+
  - PostgreSQL 15+
  - Redis 7+

### Setup Instructions

1. **Clone the repository:**
```bash
git clone https://github.com/yourorg/dataset-gen.git
cd dataset-gen
```

2. **Configure environment:**
```bash
cp .env.example .env
```

Edit `.env` and add your API keys:
```env
# Required for Claude Sonnet 4
ANTHROPIC_API_KEY=your_anthropic_api_key

# Optional for additional LLMs
OPENAI_API_KEY=your_openai_api_key
GOOGLE_API_KEY=your_google_api_key

# Generate secret key
SECRET_KEY=$(openssl rand -hex 32)
```

3. **Start with Docker Compose:**
```bash
docker-compose up -d
```

4. **Access the platform:**
- 🌐 Frontend: http://localhost:3000
- 🔧 Backend API: http://localhost:8000
- 📚 API Documentation: http://localhost:8000/docs
- 📊 Monitoring: http://localhost:3000/monitoring

5. **Create an account:**
- Navigate to http://localhost:3000/register
- Create your account
- Login to access the platform

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