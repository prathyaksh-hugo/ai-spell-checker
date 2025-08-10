# 🤖 Advanced AI Spell Checking System

A comprehensive, production-ready spell checking system with RAG (Retrieval-Augmented Generation) integration, learning capabilities, and multi-engine corrections. This system goes far beyond traditional spell checkers by providing context-aware corrections, adaptive learning, and high-performance batch processing.

## 🚀 Features

### Core Capabilities
- **Multi-Engine Spell Checking**: Combines PySpellChecker, SymSpell, Language Tool, and Transformer models
- **RAG Integration**: Uses vector databases for contextual, knowledge-augmented corrections
- **Learning System**: Adapts and learns from user corrections with persistent storage
- **Whitelist/Ignore Lists**: User-customizable word management
- **Word & Sentence Level**: Both individual word checking and full sentence correction
- **High Performance**: Optimized for speed with batch processing capabilities

### Advanced Features
- **Context-Aware Corrections**: Uses surrounding text to provide better suggestions
- **Confidence Scoring**: Advanced scoring system combining multiple factors
- **Edit Distance Calculation**: Levenshtein distance for suggestion ranking
- **Frequency-Based Ranking**: Incorporates word frequency data for better suggestions
- **Real-time Learning**: Immediate integration of user feedback
- **Performance Monitoring**: Built-in benchmarking and metrics tracking

### API Features
- **RESTful API**: Full FastAPI implementation with automatic documentation
- **Batch Processing**: Handle multiple texts efficiently
- **Async Support**: Non-blocking operations for high throughput
- **Comprehensive Endpoints**: Complete CRUD operations for all features
- **Error Handling**: Robust error handling and validation
- **CORS Support**: Ready for frontend integration

## 📦 Installation

### Quick Start

1. **Clone and Setup**:
```bash
git clone <repository>
cd <repository>
python setup_and_run.py
```

2. **Choose Installation Type**:
- `1`: Full setup (install dependencies, setup, and run)
- `2`: Quick setup (setup only)
- `3`: Run tests only
- `4`: Start API server only
- `5`: Install dependencies only

### Manual Installation

1. **Install Dependencies**:
```bash
pip install -r requirements.txt
```

2. **Setup Additional Components**:
```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('wordnet')"
```

3. **Start the API Server**:
```bash
python advanced_spell_api.py
```

## 🔧 Usage

### API Endpoints

#### Health Check
```bash
GET /health
```

#### Check Word Spelling
```bash
POST /spell_check
{
  "texts": ["appliction", "recieve", "seperate"],
  "return_suggestions": true,
  "max_suggestions": 5,
  "context": "optional context"
}
```

**Response**:
```json
{
  "results": [
    {
      "text": "appliction",
      "is_correct": false,
      "suggestions": [
        {
          "word": "application",
          "confidence": 0.9,
          "source": "dictionary_us",
          "edit_distance": 1,
          "final_score": 0.8165
        }
      ]
    }
  ]
}
```

#### Batch Sentence Correction
```bash
POST /correct_batch
{
  "texts": [
    "This is a sampel text with some erors.",
    "Another exampl of incorect speling."
  ],
  "model_type": "advanced",
  "return_confidence": true,
  "apply_corrections": true
}
```

**Response**:
```json
{
  "results": [
    {
      "original_text": "This is a sampel text with some erors.",
      "corrected_text": "This is a sample text with some errors.",
      "model_used": "advanced",
      "inference_time_ms": 45.2,
      "confidence": 0.92,
      "word_corrections": [...]
    }
  ]
}
```

#### Learn from Corrections
```bash
POST /learn
{
  "original_word": "teh",
  "corrected_word": "the",
  "context": "common typo"
}
```

#### Manage Whitelist
```bash
POST /whitelist
{
  "word": "API",
  "category": "technical_terms"
}
```

#### Ignore Words
```bash
POST /ignore
{
  "word": "JavaScript",
  "context": "programming language"
}
```

### Python Library Usage

```python
from advanced_spell_checker import MultiEngineSpellChecker

# Initialize checker
checker = MultiEngineSpellChecker()

# Check individual word
result = checker.check_word("appliction")
print(f"Correct: {result.is_correct}")
print(f"Suggestions: {[s.word for s in result.suggestions]}")

# Check sentence
sentence_result = checker.check_sentence("This is a sampel text with erors.")
print(f"Original: {sentence_result['original_sentence']}")
print(f"Corrected: {sentence_result['corrected_sentence']}")

# Learn correction
checker.learn_correction("teh", "the", "common typo")

# Add to whitelist
checker.add_to_whitelist("API")

# Ignore word
checker.ignore_word("JavaScript")
```

## 🏗️ Architecture

### System Components

```
┌─────────────────────┐
│   FastAPI Server    │
│   (advanced_spell_  │
│     api.py)         │
└──────────┬──────────┘
           │
┌──────────▼──────────┐
│ MultiEngineSpell    │
│    Checker          │
│ (advanced_spell_    │
│  checker.py)        │
└──────────┬──────────┘
           │
┌──────────▼──────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Spell Engines     │    │ Learning System │    │  RAG System     │
│                     │    │                 │    │                 │
│ • PySpellChecker    │    │ • SQLite DB     │    │ • ChromaDB      │
│ • SymSpell          │    │ • User Feedback │    │ • Embeddings    │
│ • Language Tool     │    │ • Whitelist     │    │ • Context Search│
│ • Transformers      │    │ • Ignore List   │    │ • Vector Sim.   │
└─────────────────────┘    └─────────────────┘    └─────────────────┘
```

### Key Classes

- **`MultiEngineSpellChecker`**: Main orchestrator combining all engines
- **`LearningDatabase`**: Persistent storage for user preferences and corrections
- **`RAGSpellChecker`**: Vector-based contextual suggestions
- **`SpellCheckResult`**: Structured results with suggestions and metadata

## 🎯 Performance

### Benchmarks

| Operation | Avg Time | Throughput |
|-----------|----------|------------|
| Single Word | ~50ms | ~20 words/sec |
| Sentence (10 words) | ~200ms | ~5 sentences/sec |
| Batch (100 words) | ~2s | ~50 words/sec |

### Optimization Features

- **Lazy Loading**: Components loaded on demand
- **Caching**: Frequent results cached in memory
- **Batch Processing**: Efficient multi-text handling
- **Async Operations**: Non-blocking API endpoints
- **Connection Pooling**: Database connection optimization

## 🧪 Testing

### Run Tests
```bash
# Run comprehensive test suite
python test_spell_checker.py

# Run tests against running server
python test_spell_checker.py --api-url http://localhost:8000

# Wait for server and then test
python test_spell_checker.py --wait-for-server
```

### Test Coverage
- ✅ API endpoint functionality
- ✅ Word-level spell checking accuracy
- ✅ Sentence-level correction
- ✅ Learning and adaptation
- ✅ Performance benchmarks
- ✅ Error handling
- ✅ Edge cases

## 📊 API Documentation

Once the server is running, access:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI Schema**: http://localhost:8000/openapi.json

## 🔒 Configuration

### Environment Variables
```bash
# Database paths
SPELL_LEARNING_DB_PATH=./spell_learning.db
SPELL_RAG_DB_PATH=./spell_rag_db

# Performance settings
MAX_SUGGESTIONS=5
CONFIDENCE_THRESHOLD=0.7
BATCH_SIZE=100

# Model settings
USE_TRANSFORMERS=true
USE_RAG=true
```

### Model Configuration
```python
# Custom model paths
checker = MultiEngineSpellChecker()
checker.transformer_corrector = pipeline("text2text-generation", model="your-model")
```

## 🚀 Deployment

### Docker Deployment
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["python", "advanced_spell_api.py"]
```

### Production Considerations
- Use PostgreSQL instead of SQLite for learning database
- Implement Redis caching for frequent results
- Set up load balancing for high traffic
- Monitor performance with APM tools
- Configure proper logging and error tracking

## 📈 Monitoring

### Health Endpoints
- `GET /health`: System health status
- `GET /stats`: Performance statistics
- `POST /benchmark`: Run performance tests

### Metrics Tracked
- Response times
- Accuracy rates
- Cache hit rates
- Error rates
- Resource usage

## 🤝 Integration

### Frontend Integration
```javascript
// Example frontend integration
const spellCheck = async (texts) => {
  const response = await fetch('/spell_check', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ texts, return_suggestions: true })
  });
  return await response.json();
};
```

### Webhook Integration
```python
# Example webhook for learning
@app.post("/webhook/correction")
async def learn_from_webhook(data: dict):
    checker = get_spell_checker()
    checker.learn_correction(
        data['original'],
        data['corrected'],
        data.get('context', '')
    )
    return {"status": "learned"}
```

## 🛠️ Customization

### Adding New Engines
```python
class CustomSpellEngine:
    def check_word(self, word: str) -> List[str]:
        # Your custom logic
        return suggestions

# Integrate into MultiEngineSpellChecker
checker = MultiEngineSpellChecker()
checker.custom_engine = CustomSpellEngine()
```

### Custom Scoring
```python
def custom_score_calculator(suggestion: str, original: str, confidence: float) -> float:
    # Your custom scoring logic
    return final_score

checker._calculate_final_score = custom_score_calculator
```

## 📚 Examples

### Real-world Use Cases

1. **Content Management System**
```python
# Integrate with CMS for real-time spell checking
def check_article_content(content: str) -> dict:
    checker = MultiEngineSpellChecker()
    return checker.check_sentence(content)
```

2. **Email Client Integration**
```python
# Auto-correct email content
def auto_correct_email(email_text: str) -> str:
    result = checker.check_sentence(email_text)
    return result['corrected_sentence']
```

3. **Educational Platform**
```python
# Provide learning feedback
def get_spelling_feedback(student_text: str) -> dict:
    result = checker.check_sentence(student_text)
    return {
        'errors_found': result['total_corrections'],
        'suggestions': result['word_results'],
        'corrected_text': result['corrected_sentence']
    }
```

## 🔍 Troubleshooting

### Common Issues

**1. Server won't start**
```bash
# Check dependencies
python -c "import advanced_spell_checker, fastapi, uvicorn"

# Run setup
python setup_and_run.py
```

**2. Low accuracy**
```bash
# Update dictionaries
python setup_and_run.py  # Choose option 2

# Add custom corrections
checker.learn_correction("custom_word", "correct_spelling")
```

**3. Performance issues**
```bash
# Run benchmark
python test_spell_checker.py

# Check system resources
python -c "import psutil; print(f'Memory: {psutil.virtual_memory().percent}%')"
```

## 📄 License

MIT License - see LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Run tests (`python test_spell_checker.py`)
4. Commit changes (`git commit -m 'Add amazing feature'`)
5. Push to branch (`git push origin feature/amazing-feature`)
6. Open Pull Request

## 📞 Support

- **Issues**: GitHub Issues
- **Documentation**: `/docs` endpoint when server is running
- **Examples**: `/examples` directory

---

## 🔄 Migration from Old System

If migrating from the old T5-based system:

```python
# Old usage
from deployment_pipeline import ModelManager
manager = ModelManager()
result = manager.predict("text")

# New usage
from advanced_spell_checker import MultiEngineSpellChecker
checker = MultiEngineSpellChecker()
result = checker.check_word("text")
```

The new system is backwards compatible with your existing API calls to `/correct_batch`, but provides much better accuracy and additional features.

---

**Built with ❤️ for robust, production-ready spell checking**