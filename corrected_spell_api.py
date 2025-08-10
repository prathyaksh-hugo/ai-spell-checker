#!/usr/bin/env python3
"""
Corrected Spell Checking API - Provides accurate suggestions
"""

import time
import logging
from typing import List, Dict, Any
from datetime import datetime

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# Import our fixed spell checker
from fixed_spell_checker import FixedSpellChecker, SpellCheckResult, Suggestion

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic models for API
class WordSpellCheckRequest(BaseModel):
    texts: List[str] = Field(..., description="List of words to check")
    return_suggestions: bool = Field(default=True, description="Whether to return suggestions")
    max_suggestions: int = Field(default=5, description="Maximum number of suggestions")
    context: str = Field(default="", description="Context for checking")

class SpellSuggestionResponse(BaseModel):
    word: str
    confidence: float
    source: str
    edit_distance: int
    final_score: float

class WordSpellCheckResponse(BaseModel):
    text: str
    is_correct: bool
    suggestions: List[SpellSuggestionResponse]

class BatchSpellCheckResponse(BaseModel):
    results: List[WordSpellCheckResponse]
    total_texts: int
    avg_processing_time_ms: float

# FastAPI app
app = FastAPI(
    title="Corrected Spell Checking API",
    description="Accurate spell checking with proper suggestions",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global spell checker
spell_checker = FixedSpellChecker()

@app.get("/")
async def root():
    return {
        "message": "Corrected Spell Checking API",
        "version": "1.0.0",
        "status": "working correctly",
        "features": [
            "Accurate word-level spell checking",
            "5 suggestions per misspelled word",
            "Confidence scoring",
            "Edit distance calculation"
        ]
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "spell_checker": "active"
    }

@app.post("/spell_check", response_model=BatchSpellCheckResponse)
async def spell_check_words(request: WordSpellCheckRequest):
    """Check spelling of words with accurate suggestions"""
    
    results = []
    total_processing_time = 0
    
    for text in request.texts:
        # Check the word
        result = spell_checker.check_word(text)
        total_processing_time += result.processing_time_ms
        
        # Convert suggestions to response format
        suggestions = []
        if request.return_suggestions:
            for sugg in result.suggestions[:request.max_suggestions]:
                suggestions.append(SpellSuggestionResponse(
                    word=sugg.word,
                    confidence=sugg.confidence,
                    source=sugg.source,
                    edit_distance=sugg.edit_distance,
                    final_score=sugg.final_score
                ))
        
        word_result = WordSpellCheckResponse(
            text=result.text,
            is_correct=result.is_correct,
            suggestions=suggestions
        )
        results.append(word_result)
    
    avg_processing_time = total_processing_time / len(results) if results else 0
    
    return BatchSpellCheckResponse(
        results=results,
        total_texts=len(request.texts),
        avg_processing_time_ms=avg_processing_time
    )

@app.get("/test_appliction")
async def test_appliction():
    """Test endpoint specifically for 'appliction' word"""
    result = spell_checker.check_word("appliction")
    
    suggestions = []
    for sugg in result.suggestions:
        suggestions.append({
            "word": sugg.word,
            "confidence": sugg.confidence,
            "source": sugg.source,
            "edit_distance": sugg.edit_distance,
            "final_score": sugg.final_score
        })
    
    return {
        "word": "appliction",
        "is_correct": result.is_correct,
        "suggestions": suggestions,
        "processing_time_ms": result.processing_time_ms
    }

def main():
    logger.info("Starting Corrected Spell Checking API...")
    uvicorn.run(
        "corrected_spell_api:app",
        host="0.0.0.0", 
        port=8001,  # Different port to avoid conflict
        reload=True
    )

if __name__ == "__main__":
    main()