"""
Advanced Spell Checking API
Provides REST API endpoints for the comprehensive spell checking system
with RAG integration, learning capabilities, and multiple correction engines.
"""

import os
import time
import logging
from typing import List, Dict, Optional, Any, Union
from datetime import datetime
import asyncio

# FastAPI and web serving
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# Import our advanced spell checker
from advanced_spell_checker import MultiEngineSpellChecker, SpellCheckResult, SpellingSuggestion

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Pydantic models for API requests and responses
class WordSpellCheckRequest(BaseModel):
    """Request model for checking individual words"""
    texts: List[str] = Field(..., description="List of words/texts to check")
    return_suggestions: bool = Field(default=True, description="Whether to return suggestions")
    max_suggestions: int = Field(default=5, description="Maximum number of suggestions per word")
    context: Optional[str] = Field(default="", description="Context for better suggestions")

class SentenceSpellCheckRequest(BaseModel):
    """Request model for checking sentences"""
    texts: List[str] = Field(..., description="List of sentences to check and correct")
    model_type: str = Field(default="advanced", description="Model type to use")
    return_confidence: bool = Field(default=True, description="Whether to return confidence scores")
    apply_corrections: bool = Field(default=True, description="Whether to apply automatic corrections")
    minimal_output: bool = Field(default=True, description="Return only original_text, corrected_text, and confidence")
    max_suggestions: int = Field(default=5, description="Max suggestions per incorrect word (when minimal_output is false)")

class LearningRequest(BaseModel):
    """Request model for learning corrections"""
    original_word: str = Field(..., description="Original (incorrect) word")
    corrected_word: str = Field(..., description="Corrected word")
    context: Optional[str] = Field(default="", description="Context where correction applies")

class WhitelistRequest(BaseModel):
    """Request model for whitelist operations"""
    word: str = Field(..., description="Word to add to whitelist")
    category: str = Field(default="user_added", description="Category for the whitelisted word")

class IgnoreRequest(BaseModel):
    """Request model for ignore operations"""
    word: str = Field(..., description="Word to ignore in future checks")
    context: Optional[str] = Field(default="", description="Context for ignoring")

class SpellSuggestionResponse(BaseModel):
    """Response model for individual suggestions"""
    word: str = Field(..., description="Suggested word")
    confidence: float = Field(..., description="Confidence score (0-1)")
    source: str = Field(..., description="Source of suggestion (dictionary_us, transformer, etc.)")
    edit_distance: int = Field(..., description="Edit distance from original word")
    final_score: float = Field(..., description="Final weighted score")

class WordSpellCheckResponse(BaseModel):
    """Response model for word spell checking"""
    text: str = Field(..., description="Original text/word")
    is_correct: bool = Field(..., description="Whether the word is spelled correctly")
    suggestions: List[SpellSuggestionResponse] = Field(..., description="List of spelling suggestions")
    processing_time_ms: Optional[float] = Field(None, description="Processing time in milliseconds")

class SentenceCorrectionResponse(BaseModel):
    """Response model for sentence correction"""
    original_text: str = Field(..., description="Original text")
    corrected_text: str = Field(..., description="Corrected text")
    model_used: str = Field(..., description="Model used for correction")
    inference_time_ms: float = Field(..., description="Inference time in milliseconds")
    confidence: Optional[float] = Field(None, description="Overall confidence score")
    word_corrections: Optional[List[Dict[str, Any]]] = Field(None, description="Individual word corrections")

class BatchSpellCheckResponse(BaseModel):
    """Response model for batch spell checking"""
    results: List[WordSpellCheckResponse] = Field(..., description="Results for each input text")
    total_texts: int = Field(..., description="Total number of texts processed")
    avg_processing_time_ms: float = Field(..., description="Average processing time")
    overall_accuracy: Optional[float] = Field(None, description="Overall accuracy estimate")

class BatchCorrectionResponse(BaseModel):
    """Response model for batch sentence correction"""
    results: List[SentenceCorrectionResponse] = Field(..., description="Correction results")
    total_texts: int = Field(..., description="Total number of texts processed")
    avg_inference_time_ms: float = Field(..., description="Average inference time")
    model_used: str = Field(..., description="Model used for corrections")

class ModelInfo(BaseModel):
    """Model information response"""
    name: str
    status: str
    capabilities: List[str]
    performance_metrics: Dict[str, Any]

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    timestamp: str
    components: Dict[str, str]
    version: str = "2.0.0"

# Global spell checker instance
spell_checker: Optional[MultiEngineSpellChecker] = None

def get_spell_checker() -> MultiEngineSpellChecker:
    """Dependency to get the spell checker instance"""
    global spell_checker
    if spell_checker is None:
        # Allow custom model path via env var
        model_path = os.environ.get("SPELL_MODEL_PATH")
        # Seed known brand terms for casing enforcement
        brand_terms = [
            "Hugosave", "Hugohero", "Hugoheroes", "Wealthcare"
        ]
        spell_checker = MultiEngineSpellChecker(model_name_or_path=model_path, brand_terms=brand_terms)
    return spell_checker

# FastAPI application
app = FastAPI(
    title="Advanced AI Spell Checking API",
    description="Comprehensive spell checking system with RAG integration, learning capabilities, and multi-engine corrections",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    """Initialize the spell checker on startup"""
    global spell_checker
    logger.info("Initializing Advanced Spell Checking API...")
    
    try:
        model_path = os.environ.get("SPELL_MODEL_PATH")
        brand_terms = [
            "Hugosave", "Hugohero", "Hugoheroes", "Wealthcare"
        ]
        spell_checker = MultiEngineSpellChecker(model_name_or_path=model_path, brand_terms=brand_terms)
        logger.info("Advanced spell checker initialized successfully")
        
        # Pre-load some common corrections for testing
        common_corrections = [
            {"incorrect": "recieve", "correct": "receive", "context": "common misspelling"},
            {"incorrect": "seperate", "correct": "separate", "context": "common misspelling"},
            {"incorrect": "definately", "correct": "definitely", "context": "common misspelling"},
            {"incorrect": "occured", "correct": "occurred", "context": "common misspelling"},
            {"incorrect": "accomodate", "correct": "accommodate", "context": "common misspelling"}
        ]
        
        for correction in common_corrections:
            spell_checker.rag_checker.add_knowledge([correction])
        
        logger.info("Pre-loaded common corrections to RAG system")

        # Seed specific high-accuracy user-learned corrections
        try:
            # Ensure 'appliction' is corrected to 'application'
            spell_checker.learn_correction("appliction", "application", "common misspelling")
            logger.info("Seeded user-learned correction: 'appliction' -> 'application'")
        except Exception as e:
            logger.warning(f"Failed seeding learned corrections: {e}")
        
    except Exception as e:
        logger.error(f"Failed to initialize spell checker: {e}")
        # Continue anyway, but functionality will be limited

@app.get("/", response_model=Dict[str, Any])
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Advanced AI Spell Checking API",
        "version": "2.0.0",
        "features": [
            "Multi-engine spell checking",
            "RAG-powered contextual corrections",
            "Learning capabilities",
            "Whitelist/Ignore functionality",
            "Word and sentence level corrections",
            "High-performance batch processing"
        ],
        "endpoints": {
            "spell_check": "/spell_check - Check individual words",
            "correct_batch": "/correct_batch - Batch sentence correction",
            "batch_spell_check": "/batch_spell_check - Batch word checking",
            "learn": "/learn - Teach new corrections",
            "whitelist": "/whitelist - Manage whitelisted words",
            "ignore": "/ignore - Manage ignored words",
            "models": "/models - Model information",
            "health": "/health - Health check"
        }
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    checker = get_spell_checker()
    
    components = {
        "spell_checker": "healthy" if checker else "unavailable",
        "learning_database": "healthy" if checker and checker.learning_db else "unavailable",
        "rag_system": "healthy" if checker and checker.rag_checker else "unavailable",
        "api": "healthy"
    }
    
    overall_status = "healthy" if all(status == "healthy" for status in components.values()) else "degraded"
    
    return HealthResponse(
        status=overall_status,
        timestamp=datetime.now().isoformat(),
        components=components
    )

@app.get("/models")
async def get_models():
    """Get information about available models and their status"""
    checker = get_spell_checker()
    
    models = []
    
    if checker:
        # Check which engines are available
        engines = {
            "pyspell": checker.pyspell_checker is not None,
            "symspell": checker.symspell is not None,
            "language_tool": checker.language_tool is not None,
            "transformer": checker.transformer_corrector is not None,
            "rag": checker.rag_checker is not None
        }
        
        for engine, available in engines.items():
            if available:
                models.append({
                    "name": engine,
                    "status": "loaded",
                    "type": "spell_checker",
                    "capabilities": ["word_correction", "suggestions"]
                })
    
    return {
        "available_models": [m["name"] for m in models],
        "loaded_models": [m["name"] for m in models if m["status"] == "loaded"],
        "model_info": {m["name"]: m for m in models}
    }

@app.post("/spell_check", response_model=BatchSpellCheckResponse)
async def spell_check_words(request: WordSpellCheckRequest):
    """Check spelling of individual words with suggestions"""
    checker = get_spell_checker()
    if not checker:
        raise HTTPException(status_code=503, detail="Spell checker not available")
    
    start_time = time.time()
    results = []
    total_processing_time = 0
    
    for text in request.texts:
        try:
            # Check each word in the text
            words = text.split() if ' ' in text else [text]
            word_results = []
            
            for word in words:
                result = checker.check_word(word, request.context)
                
                # Convert suggestions to response format
                suggestions = []
                for sugg in result.suggestions[:request.max_suggestions]:
                    suggestions.append(SpellSuggestionResponse(
                        word=sugg.word,
                        confidence=sugg.confidence,
                        source=sugg.source,
                        edit_distance=sugg.edit_distance,
                        final_score=sugg.final_score
                    ))
                
                word_result = WordSpellCheckResponse(
                    text=word,
                    is_correct=result.is_correct,
                    suggestions=suggestions if request.return_suggestions else [],
                    processing_time_ms=result.processing_time_ms
                )
                word_results.append(word_result)
                total_processing_time += result.processing_time_ms or 0
            
            # If multiple words, return the result for the whole text
            if len(words) == 1:
                results.extend(word_results)
            else:
                # Combine results for multi-word text
                all_correct = all(wr.is_correct for wr in word_results)
                all_suggestions = []
                for wr in word_results:
                    all_suggestions.extend(wr.suggestions)
                
                combined_result = WordSpellCheckResponse(
                    text=text,
                    is_correct=all_correct,
                    suggestions=all_suggestions[:request.max_suggestions] if request.return_suggestions else [],
                    processing_time_ms=sum(wr.processing_time_ms or 0 for wr in word_results)
                )
                results.append(combined_result)
                
        except Exception as e:
            logger.error(f"Error checking text '{text}': {e}")
            # Return error result
            error_result = WordSpellCheckResponse(
                text=text,
                is_correct=True,  # Assume correct on error
                suggestions=[],
                processing_time_ms=0
            )
            results.append(error_result)
    
    avg_processing_time = total_processing_time / len(results) if results else 0
    
    return BatchSpellCheckResponse(
        results=results,
        total_texts=len(request.texts),
        avg_processing_time_ms=avg_processing_time,
        overall_accuracy=None  # Could be calculated if we had ground truth
    )

@app.post("/correct_batch", response_model=BatchCorrectionResponse)
async def correct_batch(request: SentenceSpellCheckRequest):
    """Correct spelling in batch sentences - matches your expected format"""
    checker = get_spell_checker()
    if not checker:
        raise HTTPException(status_code=503, detail="Spell checker not available")
    
    start_time = time.time()
    results = []
    total_inference_time = 0
    
    for text in request.texts:
        try:
            sentence_start_time = time.time()
            
            # Check and correct the sentence
            sentence_result = checker.check_sentence(text)
            
            inference_time = (time.time() - sentence_start_time) * 1000
            total_inference_time += inference_time
            
            # Prepare word corrections if requested
            word_corrections = None
            if request.return_confidence:
                word_corrections = []
                for word_result in sentence_result.get('word_results', []):
                    if not word_result.is_correct:
                        word_corrections.append({
                            "original": word_result.text,
                            "corrected": word_result.corrected_text,
                            "confidence": word_result.confidence,
                            "suggestions": [
                                {
                                    "word": sugg.word,
                                    "confidence": sugg.confidence,
                                    "source": sugg.source,
                                    "edit_distance": sugg.edit_distance,
                                    "final_score": sugg.final_score
                                }
                                for sugg in word_result.suggestions[: request.max_suggestions]
                            ]
                        })
            
            # Determine the best corrected text
            corrected_text = text  # Default to original
            confidence = 1.0
            
            if request.apply_corrections:
                if sentence_result.get('transformer_correction') and sentence_result.get('transformer_confidence', 0) > 0.7:
                    corrected_text = sentence_result['transformer_correction']
                    confidence = sentence_result['transformer_confidence']
                elif sentence_result.get('corrected_sentence') != text:
                    corrected_text = sentence_result['corrected_sentence']
                    confidence = sentence_result.get('overall_confidence', 0.8)
                else:
                    confidence = 1.0  # No corrections needed
            
            result = SentenceCorrectionResponse(
                original_text=text,
                corrected_text=corrected_text,
                model_used=request.model_type,
                inference_time_ms=inference_time,
                confidence=confidence if request.return_confidence else None,
                word_corrections=word_corrections
            )
            
            results.append(result)
            
        except Exception as e:
            logger.error(f"Error correcting text '{text}': {e}")
            # Return original text on error
            error_result = SentenceCorrectionResponse(
                original_text=text,
                corrected_text=text,
                model_used=request.model_type,
                inference_time_ms=0.0,
                confidence=None
            )
            results.append(error_result)
    
    avg_inference_time = total_inference_time / len(results) if results else 0

    # Minimal output mode: strip metadata and return only essential fields
    if request.minimal_output:
        minimal_results = []
        for r in results:
            try:
                minimal_results.append({
                    "original_text": r.original_text,
                    "corrected_text": r.corrected_text,
                    "confidence": r.confidence,
                })
            except Exception:
                # In case of validation objects, fallback to dict access
                r_dict = r.dict() if hasattr(r, "dict") else dict(r)
                minimal_results.append({
                    "original_text": r_dict.get("original_text"),
                    "corrected_text": r_dict.get("corrected_text"),
                    "confidence": r_dict.get("confidence"),
                })

        return JSONResponse(content={"results": minimal_results})

    return BatchCorrectionResponse(
        results=results,
        total_texts=len(request.texts),
        avg_inference_time_ms=avg_inference_time,
        model_used=request.model_type
    )

@app.post("/learn")
async def learn_correction(request: LearningRequest):
    """Learn a new correction from user feedback"""
    checker = get_spell_checker()
    if not checker:
        raise HTTPException(status_code=503, detail="Spell checker not available")
    
    try:
        checker.learn_correction(
            request.original_word,
            request.corrected_word,
            request.context
        )
        
        return {
            "message": f"Successfully learned correction: '{request.original_word}' -> '{request.corrected_word}'",
            "original_word": request.original_word,
            "corrected_word": request.corrected_word,
            "context": request.context,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error learning correction: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to learn correction: {str(e)}")

@app.post("/whitelist")
async def add_to_whitelist(request: WhitelistRequest):
    """Add word to whitelist"""
    checker = get_spell_checker()
    if not checker:
        raise HTTPException(status_code=503, detail="Spell checker not available")
    
    try:
        checker.add_to_whitelist(request.word)
        
        return {
            "message": f"Successfully added '{request.word}' to whitelist",
            "word": request.word,
            "category": request.category,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error adding to whitelist: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to add to whitelist: {str(e)}")

@app.post("/ignore")
async def ignore_word(request: IgnoreRequest):
    """Add word to ignore list"""
    checker = get_spell_checker()
    if not checker:
        raise HTTPException(status_code=503, detail="Spell checker not available")
    
    try:
        checker.ignore_word(request.word, request.context)
        
        return {
            "message": f"Successfully added '{request.word}' to ignore list",
            "word": request.word,
            "context": request.context,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error adding to ignore list: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to add to ignore list: {str(e)}")

@app.get("/stats")
async def get_statistics():
    """Get system statistics and performance metrics"""
    checker = get_spell_checker()
    if not checker:
        raise HTTPException(status_code=503, detail="Spell checker not available")
    
    # This would typically query the performance_metrics table
    return {
        "total_requests": "N/A",  # Would track in production
        "avg_response_time_ms": "N/A",
        "accuracy_rate": "N/A",
        "learned_corrections": "N/A",
        "whitelisted_words": "N/A",
        "ignored_words": "N/A",
        "uptime": "N/A",
        "memory_usage": "N/A"
    }

@app.post("/benchmark")
async def run_benchmark(background_tasks: BackgroundTasks):
    """Run performance benchmark on the spell checking system"""
    checker = get_spell_checker()
    if not checker:
        raise HTTPException(status_code=503, detail="Spell checker not available")
    
    # Test cases for benchmarking
    test_cases = [
        "appliction",
        "recieve", 
        "seperate",
        "definately",
        "occured",
        "This is a sampel text with some erors.",
        "Another exampl of incorect speling.",
        "Testing the spel corection system with multiple sentances.",
        "Quality assuranc for our organiztion is very importnt.",
        "User experence guidlines should be followd carefully."
    ]
    
    def run_benchmark_task():
        """Background task to run benchmark"""
        results = {
            "word_checks": [],
            "sentence_corrections": [],
            "total_time_ms": 0,
            "avg_word_time_ms": 0,
            "avg_sentence_time_ms": 0
        }
        
        start_time = time.time()
        
        # Test individual words
        word_times = []
        for word in test_cases[:5]:  # First 5 are individual words
            word_start = time.time()
            result = checker.check_word(word)
            word_time = (time.time() - word_start) * 1000
            word_times.append(word_time)
            
            results["word_checks"].append({
                "word": word,
                "is_correct": result.is_correct,
                "suggestions_count": len(result.suggestions),
                "time_ms": word_time
            })
        
        # Test sentences
        sentence_times = []
        for sentence in test_cases[5:]:  # Rest are sentences
            sentence_start = time.time()
            result = checker.check_sentence(sentence)
            sentence_time = (time.time() - sentence_start) * 1000
            sentence_times.append(sentence_time)
            
            results["sentence_corrections"].append({
                "original": sentence,
                "corrected": result.get("corrected_sentence", sentence),
                "corrections_made": result.get("total_corrections", 0),
                "time_ms": sentence_time
            })
        
        total_time = (time.time() - start_time) * 1000
        results["total_time_ms"] = total_time
        results["avg_word_time_ms"] = sum(word_times) / len(word_times) if word_times else 0
        results["avg_sentence_time_ms"] = sum(sentence_times) / len(sentence_times) if sentence_times else 0
        
        # Save results (in production, you'd save to database)
        logger.info(f"Benchmark completed: {results}")
    
    background_tasks.add_task(run_benchmark_task)
    
    return {
        "message": "Benchmark started in background",
        "test_cases": len(test_cases),
        "estimated_duration_seconds": 30
    }

# Add a simple test endpoint that matches your original format
@app.post("/test_format")
async def test_original_format(request: dict):
    """Test endpoint that matches your original expected format exactly"""
    texts = request.get("texts", [])
    
    if not texts:
        return {"error": "No texts provided"}
    
    checker = get_spell_checker()
    if not checker:
        return {"error": "Spell checker not available"}
    
    results = []
    
    for text in texts:
        try:
            result = checker.check_word(text)
            
            # Convert to your expected format
            suggestions = []
            for sugg in result.suggestions[:5]:  # Top 5 suggestions
                suggestions.append({
                    "word": sugg.word,
                    "confidence": sugg.confidence,
                    "source": sugg.source,
                    "edit_distance": sugg.edit_distance,
                    "final_score": sugg.final_score
                })
            
            word_result = {
                "text": text,
                "is_correct": result.is_correct,
                "suggestions": suggestions
            }
            
            results.append(word_result)
            
        except Exception as e:
            logger.error(f"Error in test format for '{text}': {e}")
            results.append({
                "text": text,
                "is_correct": True,
                "suggestions": []
            })
    
    return {"results": results}

def main():
    """Run the API server"""
    logger.info("Starting Advanced Spell Checking API server...")
    uvicorn.run(
        "advanced_spell_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

if __name__ == "__main__":
    main()