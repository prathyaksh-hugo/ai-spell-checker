# 🚀 Deployment Guide - Advanced AI Spell Checking System

## Quick Start (5-Minute Setup)

```bash
# 1. Clone and navigate to project
cd /workspace

# 2. Run the setup script
python3 setup_and_run.py
# Choose option 1: Full setup (install dependencies, setup, and run)

# 3. That's it! Your API is running at http://localhost:8000
```

## Test Your Deployment

Once the server is running, test it:

```bash
# Test basic functionality
curl -X POST "http://localhost:8000/spell_check" \
  -H "Content-Type: application/json" \
  -d '{"texts": ["appliction", "recieve", "seperate"]}'

# Test sentence correction (matches your original format)
curl -X POST "http://localhost:8000/correct_batch" \
  -H "Content-Type: application/json" \
  -d '{
    "texts": [
      "This is a sampel text with some erors.",
      "Anothr exampl with incorect speling."
    ],
    "model_type": "advanced",
    "return_confidence": true
  }'
```

## Expected Response Format

The new system provides the exact response format you requested:

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
          "final_score": 0.8165603305785124
        },
        {
          "word": "affliction",
          "confidence": 0.9,
          "source": "dictionary_us",
          "edit_distance": 2,
          "final_score": 0.8037
        }
      ]
    }
  ]
}
```

## Production Deployment

### Option 1: Direct Python
```bash
# Install dependencies
pip install -r requirements.txt

# Start the server
python3 advanced_spell_api.py
```

### Option 2: Using uvicorn
```bash
# For production with multiple workers
uvicorn advanced_spell_api:app --host 0.0.0.0 --port 8000 --workers 4
```

### Option 3: Docker
```bash
# Create Dockerfile (already provided in the system)
docker build -t advanced-spell-checker .
docker run -p 8000:8000 advanced-spell-checker
```

## Learning and Customization

### Learn from User Corrections
```bash
curl -X POST "http://localhost:8000/learn" \
  -H "Content-Type: application/json" \
  -d '{
    "original_word": "teh",
    "corrected_word": "the",
    "context": "common typo"
  }'
```

### Add Words to Whitelist
```bash
curl -X POST "http://localhost:8000/whitelist" \
  -H "Content-Type: application/json" \
  -d '{
    "word": "API",
    "category": "technical_terms"
  }'
```

### Ignore Words
```bash
curl -X POST "http://localhost:8000/ignore" \
  -H "Content-Type: application/json" \
  -d '{
    "word": "JavaScript",
    "context": "programming language"
  }'
```

## Integration with Frontend

### JavaScript Example
```javascript
async function checkSpelling(texts) {
  const response = await fetch('/spell_check', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      texts: texts,
      return_suggestions: true,
      max_suggestions: 5
    })
  });
  
  return await response.json();
}

// Usage
const results = await checkSpelling(['appliction', 'recieve']);
console.log(results.results[0].suggestions);
```

### React Component Example
```jsx
import React, { useState } from 'react';

function SpellChecker() {
  const [text, setText] = useState('');
  const [suggestions, setSuggestions] = useState([]);

  const checkSpelling = async () => {
    const response = await fetch('/spell_check', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ texts: [text] })
    });
    
    const data = await response.json();
    setSuggestions(data.results[0].suggestions);
  };

  return (
    <div>
      <input 
        value={text} 
        onChange={(e) => setText(e.target.value)}
        placeholder="Enter text to check"
      />
      <button onClick={checkSpelling}>Check Spelling</button>
      
      {suggestions.map((suggestion, index) => (
        <div key={index}>
          {suggestion.word} (confidence: {suggestion.confidence})
        </div>
      ))}
    </div>
  );
}
```

## Performance Optimization

### Environment Variables
```bash
# Set these for production
export MAX_SUGGESTIONS=5
export CONFIDENCE_THRESHOLD=0.7
export BATCH_SIZE=100
export USE_TRANSFORMERS=true
export USE_RAG=true
```

### Database Configuration
```python
# For high-traffic production, use PostgreSQL
# Update advanced_spell_checker.py:
class LearningDatabase:
    def __init__(self, db_url="postgresql://user:pass@localhost/spellcheck"):
        # Use SQLAlchemy for PostgreSQL connection
        pass
```

## Monitoring and Health Checks

### Health Endpoint
```bash
curl http://localhost:8000/health
```

### Performance Statistics
```bash
curl http://localhost:8000/stats
```

### Run Benchmark
```bash
curl -X POST http://localhost:8000/benchmark
```

## Troubleshooting

### Common Issues

1. **Server won't start**
   ```bash
   # Check dependencies
   python3 -c "import fastapi, uvicorn"
   
   # Run setup again
   python3 setup_and_run.py
   ```

2. **Poor accuracy**
   ```bash
   # Update dictionaries
   python3 setup_and_run.py  # Choose option 2
   
   # Add domain-specific words to whitelist
   curl -X POST "http://localhost:8000/whitelist" \
     -H "Content-Type: application/json" \
     -d '{"word": "YourDomainWord"}'
   ```

3. **Memory issues**
   ```bash
   # Check system resources
   python3 -c "
   import psutil
   print(f'Memory: {psutil.virtual_memory().percent}%')
   print(f'Available: {psutil.virtual_memory().available / (1024**3):.1f} GB')
   "
   ```

### Logs and Debugging
```bash
# Check logs
tail -f advanced_spell_checker.log

# Enable debug mode
export LOG_LEVEL=DEBUG
python3 advanced_spell_api.py
```

## Migration from Original System

If you're migrating from the old T5-based system:

### Old API calls still work:
```bash
# This still works (backwards compatible)
curl -X POST "http://localhost:8000/correct_batch" \
  -H "Content-Type: application/json" \
  -d '{
    "texts": ["This is a sampel text."],
    "model_type": "dynamic_int8",
    "return_confidence": false
  }'
```

### But now you get much better results:
- ✅ Actual spelling corrections (old system returned unchanged text)
- ✅ Multiple suggestion sources
- ✅ Confidence scores and edit distances
- ✅ Learning capabilities
- ✅ Context-aware suggestions

## Backup and Recovery

### Backup Learning Data
```bash
# Copy the learning database
cp spell_learning.db spell_learning_backup.db

# Export learned corrections
sqlite3 spell_learning.db "SELECT * FROM user_corrections;" > corrections_backup.sql
```

### Restore Learning Data
```bash
# Restore from backup
cp spell_learning_backup.db spell_learning.db

# Import corrections
sqlite3 spell_learning.db < corrections_backup.sql
```

## Advanced Configuration

### Custom Model Integration
```python
# In advanced_spell_checker.py, add your custom model:
from your_model import YourSpellChecker

class MultiEngineSpellChecker:
    def __init__(self):
        # ... existing code ...
        self.your_custom_model = YourSpellChecker()
    
    def check_word(self, word: str, context: str = "") -> SpellCheckResult:
        # Add your model to the checking process
        custom_suggestions = self.your_custom_model.get_suggestions(word)
        # ... integrate with existing suggestions ...
```

### RAG Knowledge Base
```python
# Add domain-specific corrections to RAG
corrections = [
    {"incorrect": "yourdomain_word", "correct": "correct_word", "context": "domain context"},
    # ... more corrections
]

checker = MultiEngineSpellChecker()
checker.rag_checker.add_knowledge(corrections)
```

## API Documentation

Once deployed, full API documentation is available at:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Support

- **Issues**: Create GitHub issues for bugs
- **Documentation**: README_ADVANCED.md
- **Examples**: demo_advanced_spell_checker.py
- **Tests**: test_spell_checker.py

---

## 🎯 Summary

You now have a production-ready spell checking system that:

✅ **Fixes your original problems**:
- Actually corrects spelling (unlike the old system)
- Provides proper suggestions with confidence scores
- Returns populated model information

✅ **Adds powerful new features**:
- Learning from user corrections
- RAG-powered contextual suggestions
- Multi-engine processing
- Whitelist/ignore functionality

✅ **Maintains compatibility**:
- Your existing API calls still work
- Same endpoint structure (/correct_batch)
- Enhanced response format

✅ **Production ready**:
- Comprehensive testing
- Performance optimization
- Health monitoring
- Error handling

**Start using it now**: `python3 setup_and_run.py` and choose option 1!