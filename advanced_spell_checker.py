"""
Advanced AI-Powered Spell Checking System
Features:
- Multi-engine spell checking (PySpellChecker, SymSpell, Language Tool, Transformers)
- RAG (Retrieval-Augmented Generation) integration
- Learning capabilities with whitelist/blacklist
- Context-aware corrections
- Confidence scoring and suggestions
- Sentence-level and word-level corrections
- Performance optimization
"""

import os
import json
import time
import sqlite3
import logging
from typing import List, Dict, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import asyncio
import re

# Core libraries
import numpy as np
import pandas as pd
from tqdm import tqdm

# NLP and spell checking libraries (import independently; do not fail all on one missing)
HAVE_PYSPELLCHECKER = False
HAVE_SYMSPELL = False
HAVE_LANGUAGE_TOOL = False

try:
    from spellchecker import SpellChecker  # pyspellchecker
    HAVE_PYSPELLCHECKER = True
except ImportError as e:
    logging.warning(f"PySpellChecker not available: {e}")

try:
    from symspellpy import SymSpell, Verbosity
    HAVE_SYMSPELL = True
except ImportError as e:
    logging.warning(f"SymSpell not available: {e}")

try:
    import language_tool_python
    HAVE_LANGUAGE_TOOL = True
except ImportError as e:
    logging.warning(f"language_tool_python not available: {e}")

# Optional extras (do not gate functionality)
try:
    import nltk  # noqa: F401
except ImportError:
    pass
try:
    from textblob import TextBlob  # noqa: F401
except ImportError:
    pass

# ML libraries
try:
    import torch
    from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
    from sentence_transformers import SentenceTransformer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# Vector database
try:
    import chromadb
    from chromadb.config import Settings
    VECTOR_DB_AVAILABLE = True
except ImportError:
    VECTOR_DB_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SpellingSuggestion:
    """A spelling suggestion with metadata"""
    word: str
    confidence: float
    source: str  # dictionary_us, transformer, context, etc.
    edit_distance: int
    final_score: float
    frequency_score: Optional[float] = None
    context_score: Optional[float] = None

@dataclass
class SpellCheckResult:
    """Result of spell checking a single word or text"""
    text: str
    is_correct: bool
    suggestions: List[SpellingSuggestion]
    corrected_text: Optional[str] = None
    confidence: Optional[float] = None
    processing_time_ms: Optional[float] = None

@dataclass
class ContextualCorrection:
    """Context-aware correction with RAG integration"""
    original_text: str
    corrected_text: str
    confidence: float
    context_used: List[str]
    reasoning: str

class LearningDatabase:
    """Database for storing learned corrections and user preferences"""
    
    def __init__(self, db_path: str = "./spell_learning.db"):
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Initialize the learning database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # User corrections table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_corrections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                original_word TEXT NOT NULL,
                corrected_word TEXT NOT NULL,
                context TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                frequency INTEGER DEFAULT 1,
                confidence REAL DEFAULT 1.0
            )
        """)
        
        # Whitelist table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS whitelist (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                word TEXT UNIQUE NOT NULL,
                added_date DATETIME DEFAULT CURRENT_TIMESTAMP,
                category TEXT DEFAULT 'user_added'
            )
        """)
        
        # Ignored words table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ignored_words (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                word TEXT UNIQUE NOT NULL,
                added_date DATETIME DEFAULT CURRENT_TIMESTAMP,
                context TEXT
            )
        """)
        
        # Performance metrics
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS performance_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                operation_type TEXT NOT NULL,
                processing_time_ms REAL NOT NULL,
                accuracy REAL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        conn.commit()
        conn.close()
    
    def add_user_correction(self, original: str, corrected: str, context: str = None):
        """Add a user correction to the learning database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Check if correction already exists
        cursor.execute("""
            SELECT frequency FROM user_corrections 
            WHERE original_word = ? AND corrected_word = ?
        """, (original, corrected))
        
        result = cursor.fetchone()
        if result:
            # Increment frequency
            cursor.execute("""
                UPDATE user_corrections 
                SET frequency = frequency + 1, timestamp = CURRENT_TIMESTAMP
                WHERE original_word = ? AND corrected_word = ?
            """, (original, corrected))
        else:
            # Add new correction
            cursor.execute("""
                INSERT INTO user_corrections (original_word, corrected_word, context)
                VALUES (?, ?, ?)
            """, (original, corrected, context))
        
        conn.commit()
        conn.close()
    
    def add_to_whitelist(self, word: str, category: str = "user_added"):
        """Add word to whitelist"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR IGNORE INTO whitelist (word, category)
            VALUES (?, ?)
        """, (word.lower(), category))
        
        conn.commit()
        conn.close()
    
    def add_to_ignored(self, word: str, context: str = None):
        """Add word to ignored list"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR IGNORE INTO ignored_words (word, context)
            VALUES (?, ?)
        """, (word.lower(), context))
        
        conn.commit()
        conn.close()
    
    def is_whitelisted(self, word: str) -> bool:
        """Check if word is in whitelist"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT 1 FROM whitelist WHERE word = ?", (word.lower(),))
        result = cursor.fetchone() is not None
        
        conn.close()
        return result
    
    def is_ignored(self, word: str) -> bool:
        """Check if word should be ignored"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT 1 FROM ignored_words WHERE word = ?", (word.lower(),))
        result = cursor.fetchone() is not None
        
        conn.close()
        return result
    
    def get_user_correction(self, word: str) -> Optional[str]:
        """Get user's preferred correction for a word"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT corrected_word, frequency 
            FROM user_corrections 
            WHERE original_word = ? 
            ORDER BY frequency DESC, timestamp DESC 
            LIMIT 1
        """, (word.lower(),))
        
        result = cursor.fetchone()
        conn.close()
        
        return result[0] if result else None

class RAGSpellChecker:
    """RAG-powered spell checker with external knowledge integration"""
    
    def __init__(self, vector_db_path: str = "./spell_rag_db"):
        self.vector_db_path = vector_db_path
        self.sentence_transformer = None
        self.chroma_client = None
        self.collection = None
        
        if VECTOR_DB_AVAILABLE:
            self._init_vector_db()
        if TRANSFORMERS_AVAILABLE:
            self._init_sentence_transformer()
    
    def _init_vector_db(self):
        """Initialize ChromaDB for RAG"""
        try:
            self.chroma_client = chromadb.PersistentClient(path=self.vector_db_path)
            self.collection = self.chroma_client.get_or_create_collection(
                name="spell_corrections",
                metadata={"description": "Spell correction knowledge base"}
            )
            logger.info("Vector database initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize vector database: {e}")
    
    def _init_sentence_transformer(self):
        """Initialize sentence transformer for embeddings"""
        try:
            self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Sentence transformer initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize sentence transformer: {e}")
    
    def add_knowledge(self, corrections: List[Dict[str, str]]):
        """Add spell correction knowledge to RAG database"""
        if not self.collection or not self.sentence_transformer:
            return
        
        try:
            documents = []
            metadatas = []
            ids = []
            
            for i, correction in enumerate(corrections):
                doc = f"Correct '{correction['incorrect']}' to '{correction['correct']}'"
                if 'context' in correction:
                    doc += f" in context: {correction['context']}"
                
                documents.append(doc)
                metadatas.append(correction)
                ids.append(f"correction_{i}_{int(time.time())}")
            
            embeddings = self.sentence_transformer.encode(documents)
            
            self.collection.add(
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids
            )
            
            logger.info(f"Added {len(corrections)} corrections to RAG database")
        except Exception as e:
            logger.error(f"Failed to add knowledge to RAG: {e}")
    
    def get_contextual_suggestions(self, word: str, context: str = "", top_k: int = 5) -> List[Dict]:
        """Get contextual suggestions using RAG"""
        if not self.collection or not self.sentence_transformer:
            return []
        
        try:
            query = f"Correct '{word}'"
            if context:
                query += f" in context: {context}"
            
            query_embedding = self.sentence_transformer.encode([query])
            
            results = self.collection.query(
                query_embeddings=query_embedding,
                n_results=top_k
            )
            
            suggestions = []
            for i, (doc, metadata, distance) in enumerate(zip(
                results['documents'][0], 
                results['metadatas'][0], 
                results['distances'][0]
            )):
                suggestions.append({
                    'suggested_word': metadata.get('correct', ''),
                    'confidence': 1.0 - distance,  # Convert distance to confidence
                    'context': metadata.get('context', ''),
                    'source': 'rag'
                })
            
            return suggestions
        except Exception as e:
            logger.error(f"Failed to get RAG suggestions: {e}")
            return []

class MultiEngineSpellChecker:
    """Multi-engine spell checker combining multiple approaches"""
    
    def __init__(self, model_name_or_path: Optional[str] = None, brand_terms: Optional[List[str]] = None):
        self.learning_db = LearningDatabase()
        self.rag_checker = RAGSpellChecker()
        
        # Optional custom transformer model path/name
        self.model_name_or_path = model_name_or_path or os.environ.get("SPELL_MODEL_PATH")
        
        # Initialize spell checking engines
        self.pyspell_checker = None
        self.symspell = None
        self.language_tool = None
        self.transformer_corrector = None
        
        # Initialize any traditional checkers that are available
        self._init_traditional_checkers()
        if TRANSFORMERS_AVAILABLE:
            self._init_transformer_corrector()
        
        # Brand terms map for enforcing capitalization (key: lowercase, value: canonical)
        self.brand_terms_case_map = self._build_brand_terms_case_map(brand_terms)
        
        # Word frequency data for scoring
        self.word_frequencies = self._load_word_frequencies()
    
    def _init_traditional_checkers(self):
        """Initialize any available traditional spell checking libraries independently"""
        loaded_engines = []

        # PySpellChecker
        if HAVE_PYSPELLCHECKER:
            try:
                self.pyspell_checker = SpellChecker()
                loaded_engines.append("pyspell")
            except Exception as e:
                self.pyspell_checker = None
                logger.warning(f"Failed to initialize PySpellChecker: {e}")

        # SymSpell
        if HAVE_SYMSPELL:
            try:
                self.symspell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
                dictionary_path = "./symspell_frequency_dictionary_en_82_765.txt"
                if not os.path.exists(dictionary_path):
                    # Create a basic dictionary if not available
                    self._create_basic_dictionary(dictionary_path)
                self.symspell.load_dictionary(dictionary_path, term_index=0, count_index=1)
                loaded_engines.append("symspell")
            except Exception as e:
                self.symspell = None
                logger.warning(f"Failed to initialize SymSpell: {e}")

        # Language Tool
        if HAVE_LANGUAGE_TOOL:
            try:
                self.language_tool = language_tool_python.LanguageTool('en-US')
                loaded_engines.append("language_tool")
            except Exception as e:
                self.language_tool = None
                logger.warning(f"Failed to initialize LanguageTool: {e}")

        if loaded_engines:
            logger.info(f"Traditional spell checkers initialized: {', '.join(loaded_engines)}")
        else:
            logger.warning("No traditional spell checkers could be initialized. Suggestions may be limited.")
    
    def _init_transformer_corrector(self):
        """Initialize transformer-based spell corrector"""
        # If user provided a custom fine-tuned model path/name, try it first
        candidate_models: List[Tuple[str, Optional[str]]] = []
        if self.model_name_or_path:
            candidate_models.append((self.model_name_or_path, None))
        
        # Fallbacks
        candidate_models.extend([
            ("ai-forever/T5-large-spell", None),
            ("oliverguhr/spelling-correction-english-base", "oliverguhr/spelling-correction-english-base"),
            ("t5-small", None),
        ])
        
        for model_name, tokenizer_name in candidate_models:
            try:
                self.transformer_corrector = pipeline(
                    "text2text-generation",
                    model=model_name,
                    tokenizer=(tokenizer_name or model_name)
                )
                logger.info(f"Transformer corrector initialized: {model_name}")
                return
            except Exception as e:
                logger.warning(f"Failed to initialize transformer model '{model_name}': {e}")
        self.transformer_corrector = None

    def _build_brand_terms_case_map(self, brand_terms: Optional[List[str]]) -> Dict[str, str]:
        """Build a lowercase->canonical case map for brand terms and common hashtags."""
        # Default key terms commonly used in Hugosave brand guide
        default_terms = [
            "Hugosave", "Hugohero", "Hugoheroes", "Wealthcare"
        ]
        terms = brand_terms or default_terms
        case_map: Dict[str, str] = {}
        for term in terms:
            case_map[term.lower()] = term
            # Also track hashtag variant
            case_map[f"#{term.lower()}"] = f"#{term}"
        return case_map

    def enforce_brand_casing(self, text: str) -> str:
        """Ensure brand terms are correctly capitalized in the provided text."""
        if not text:
            return text
        result = text
        # Replace longer keys first to avoid partial overlaps
        for key in sorted(self.brand_terms_case_map.keys(), key=len, reverse=True):
            canonical = self.brand_terms_case_map[key]
            if key.startswith('#'):
                # Hashtag replacement (case-insensitive)
                pattern = re.compile(re.escape(key), flags=re.IGNORECASE)
                result = pattern.sub(canonical, result)
            else:
                # Word boundary replacement (case-insensitive)
                pattern = re.compile(rf"\b{re.escape(key)}\b", flags=re.IGNORECASE)
                result = pattern.sub(canonical, result)
        return result
    
    def _create_basic_dictionary(self, path: str):
        """Create a basic frequency dictionary"""
        # Basic English words with frequencies
        basic_words = {
            'the': 1000000, 'be': 500000, 'to': 400000, 'of': 350000, 'and': 300000,
            'a': 250000, 'in': 200000, 'that': 150000, 'have': 140000, 'I': 130000,
            'it': 120000, 'for': 110000, 'not': 100000, 'on': 95000, 'with': 90000,
            'he': 85000, 'as': 80000, 'you': 75000, 'do': 70000, 'at': 65000
        }
        
        # Enrich with common corrections and technical terms
        enriched_words = {
            # Correct forms for common misspellings
            'application': 400000,
            'applications': 200000,
            'receive': 500000,
            'separate': 300000,
            'definitely': 200000,
            'occurred': 150000,
            'accommodate': 120000,
            'acknowledge': 80000,
            'beginning': 250000,
            'calendar': 120000,
            'cemetery': 50000,
            'changeable': 40000,
            'consensus': 60000,
            'dilemma': 70000,
            'existence': 180000,
            'experience': 400000,
            'independent': 200000,
            'occasion': 100000,
            'privilege': 80000,
            'tomorrow': 150000,
            'until': 500000,
            'weird': 80000,
            'affliction': 90000,
            # Useful base words often proposed
            'javascript': 30000,
            'python': 50000,
            'technical': 60000,
            'document': 120000,
        }
        basic_words.update(enriched_words)

        with open(path, 'w') as f:
            for word, freq in basic_words.items():
                f.write(f"{word.lower()} {freq}\n")
    
    def _load_word_frequencies(self) -> Dict[str, int]:
        """Load word frequency data for scoring"""
        # In a real implementation, you'd load from a comprehensive frequency corpus
        # For now, return a basic dictionary
        return {
            'the': 1000000, 'be': 500000, 'to': 400000, 'of': 350000, 'and': 300000,
            'a': 250000, 'in': 200000, 'that': 150000, 'have': 140000, 'I': 130000
        }
    
    def _calculate_edit_distance(self, word1: str, word2: str) -> int:
        """Calculate Levenshtein distance between two words"""
        if len(word1) > len(word2):
            word1, word2 = word2, word1
        
        distances = range(len(word1) + 1)
        for i2, c2 in enumerate(word2):
            new_distances = [i2 + 1]
            for i1, c1 in enumerate(word1):
                if c1 == c2:
                    new_distances.append(distances[i1])
                else:
                    new_distances.append(1 + min(distances[i1], distances[i1 + 1], new_distances[-1]))
            distances = new_distances
        return distances[-1]
    
    def _calculate_final_score(self, suggestion: str, original: str, confidence: float, source: str) -> float:
        """Calculate final score for a suggestion"""
        edit_distance = self._calculate_edit_distance(original.lower(), suggestion.lower())
        max_length = max(len(original), len(suggestion))
        
        # Edit distance penalty
        edit_penalty = edit_distance / max_length if max_length > 0 else 1.0
        
        # Frequency boost
        frequency_boost = 0.0
        if suggestion.lower() in self.word_frequencies:
            # Normalize frequency score
            max_freq = max(self.word_frequencies.values())
            frequency_boost = self.word_frequencies[suggestion.lower()] / max_freq * 0.2
        
        # Source confidence adjustment
        source_weights = {
            'user_learned': 1.0,
            'dictionary_us': 0.9,
            'transformer': 0.8,
            'symspell': 0.85,
            'pyspell': 0.8,
            'language_tool': 0.9,
            'rag': 0.75
        }
        
        source_weight = source_weights.get(source, 0.7)
        
        # Calculate final score
        final_score = (confidence * source_weight * (1 - edit_penalty) + frequency_boost)
        return min(1.0, max(0.0, final_score))
    
    def check_word(self, word: str, context: str = "") -> SpellCheckResult:
        """Check spelling of a single word using multiple engines"""
        start_time = time.time()
        
        # Enforce brand casing as a first-class correction
        brand_canonical = self.brand_terms_case_map.get(word.lower())
        if brand_canonical and word != brand_canonical:
            suggestion = SpellingSuggestion(
                word=brand_canonical,
                confidence=1.0,
                source="brand_rule",
                edit_distance=self._calculate_edit_distance(word, brand_canonical),
                final_score=1.0
            )
            return SpellCheckResult(
                text=word,
                is_correct=False,
                suggestions=[suggestion],
                corrected_text=brand_canonical,
                confidence=1.0,
                processing_time_ms=(time.time() - start_time) * 1000
            )

        # Check if word is ignored or whitelisted
        if self.learning_db.is_ignored(word):
            return SpellCheckResult(
                text=word,
                is_correct=True,
                suggestions=[],
                processing_time_ms=(time.time() - start_time) * 1000
            )
        
        if self.learning_db.is_whitelisted(word):
            return SpellCheckResult(
                text=word,
                is_correct=True,
                suggestions=[],
                processing_time_ms=(time.time() - start_time) * 1000
            )
        
        # Check for user-learned corrections
        user_correction = self.learning_db.get_user_correction(word)
        if user_correction:
            suggestion = SpellingSuggestion(
                word=user_correction,
                confidence=1.0,
                source="user_learned",
                edit_distance=self._calculate_edit_distance(word, user_correction),
                final_score=1.0
            )
            return SpellCheckResult(
                text=word,
                is_correct=False,
                suggestions=[suggestion],
                corrected_text=user_correction,
                confidence=1.0,
                processing_time_ms=(time.time() - start_time) * 1000
            )
        
        all_suggestions = []
        is_correct = True
        
        # PySpellChecker
        if self.pyspell_checker:
            try:
                if word.lower() not in self.pyspell_checker:
                    is_correct = False
                    candidates = self.pyspell_checker.candidates(word)
                    if candidates:
                        for candidate in list(candidates)[:5]:  # Top 5
                            suggestion = SpellingSuggestion(
                                word=candidate,
                                confidence=0.9,
                                source="dictionary_us",
                                edit_distance=self._calculate_edit_distance(word, candidate),
                                final_score=0.0  # Will be calculated later
                            )
                            all_suggestions.append(suggestion)
            except Exception as e:
                logger.warning(f"PySpellChecker error for '{word}': {e}")
        
        # SymSpell
        if self.symspell:
            try:
                symspell_suggestions = self.symspell.lookup(word, Verbosity.TOP, max_edit_distance=2)
                if not symspell_suggestions or symspell_suggestions[0].term != word:
                    is_correct = False
                    for sym_suggestion in symspell_suggestions[:5]:
                        # Calculate confidence based on edit distance and frequency
                        max_distance = 2
                        confidence = 1.0 - (sym_suggestion.distance / max_distance)
                        
                        suggestion = SpellingSuggestion(
                            word=sym_suggestion.term,
                            confidence=confidence,
                            source="symspell",
                            edit_distance=sym_suggestion.distance,
                            final_score=0.0,
                            frequency_score=sym_suggestion.count
                        )
                        all_suggestions.append(suggestion)
            except Exception as e:
                logger.warning(f"SymSpell error for '{word}': {e}")
        
        # Language Tool
        if self.language_tool:
            try:
                matches = self.language_tool.check(word)
                if matches:
                    is_correct = False
                    for match in matches:
                        for replacement in match.replacements[:3]:  # Top 3 per match
                            suggestion = SpellingSuggestion(
                                word=replacement,
                                confidence=0.9,
                                source="language_tool",
                                edit_distance=self._calculate_edit_distance(word, replacement),
                                final_score=0.0
                            )
                            all_suggestions.append(suggestion)
            except Exception as e:
                logger.warning(f"Language Tool error for '{word}': {e}")
        
        # RAG suggestions (filtered by edit distance)
        rag_suggestions = self.rag_checker.get_contextual_suggestions(word, context, top_k=3)
        for rag_sugg in rag_suggestions:
            if rag_sugg['suggested_word']:
                # Clamp confidence to [0, 1]
                rag_conf = rag_sugg['confidence']
                if rag_conf is None:
                    rag_conf = 0.5
                rag_conf = max(0.0, min(1.0, float(rag_conf)))
                ed = self._calculate_edit_distance(word, rag_sugg['suggested_word'])
                # Skip far-off suggestions to avoid irrelevant corrections
                if ed > 3:
                    continue
                suggestion = SpellingSuggestion(
                    word=rag_sugg['suggested_word'],
                    confidence=rag_conf,
                    source="rag",
                    edit_distance=ed,
                    final_score=0.0,
                    context_score=rag_conf
                )
                all_suggestions.append(suggestion)
        
        # Remove duplicates and calculate final scores
        unique_suggestions = {}
        for suggestion in all_suggestions:
            key = suggestion.word.lower()
            if key not in unique_suggestions or suggestion.confidence > unique_suggestions[key].confidence:
                suggestion.final_score = self._calculate_final_score(
                    suggestion.word, word, suggestion.confidence, suggestion.source
                )
                unique_suggestions[key] = suggestion
        
        # Sort by final score
        final_suggestions = sorted(unique_suggestions.values(), key=lambda x: x.final_score, reverse=True)
        
        # Determine best correction and adjust correctness if only suggestion sources fired
        corrected_text = None
        confidence = None
        if final_suggestions:
            top = final_suggestions[0]
            # Only accept as correction if the final score is reasonable
            if top.final_score is not None and top.final_score >= 0.3 and top.word.lower() != word.lower():
                corrected_text = top.word
                confidence = top.final_score
                if is_correct:
                    is_correct = False
        
        processing_time = (time.time() - start_time) * 1000
        
        return SpellCheckResult(
            text=word,
            is_correct=is_correct,
            suggestions=final_suggestions[:5],  # Top 5 suggestions
            corrected_text=corrected_text,
            confidence=confidence,
            processing_time_ms=processing_time
        )
    
    def check_sentence(self, sentence: str) -> Dict[str, Any]:
        """Check spelling of an entire sentence with context"""
        start_time = time.time()
        
        # Tokenize sentence
        words = re.findall(r'\b\w+\b', sentence)
        word_positions = []
        for word in words:
            start_pos = sentence.find(word)
            word_positions.append((word, start_pos, start_pos + len(word)))
        
        results = []
        corrected_sentence = sentence
        total_corrections = 0
        
        for word, start_pos, end_pos in word_positions:
            # Get context (words before and after)
            word_index = words.index(word)
            context_before = " ".join(words[max(0, word_index-3):word_index])
            context_after = " ".join(words[word_index+1:min(len(words), word_index+4)])
            context = f"{context_before} {context_after}".strip()
            
            word_result = self.check_word(word, context)
            results.append(word_result)
            
            # Apply correction to sentence
            if not word_result.is_correct and word_result.corrected_text:
                corrected_sentence = corrected_sentence.replace(word, word_result.corrected_text, 1)
                total_corrections += 1
        
        # Try transformer-based sentence correction
        transformer_correction = None
        transformer_confidence = 0.0
        
        # Always attempt a transformer-based sentence correction when available
        if self.transformer_corrector:
            try:
                # Different models may expect different prefixes; try without and with a prefix
                inputs_to_try = [sentence, f"correct: {sentence}", f"fix: {sentence}"]
                best_text = None
                best_conf = -1.0
                for input_text in inputs_to_try:
                    transformer_result = self.transformer_corrector(input_text, max_length=512, num_return_sequences=1)
                    if transformer_result and 'generated_text' in transformer_result[0]:
                        candidate = transformer_result[0]['generated_text']
                        # Confidence based on similarity to original (higher is better), but prefer meaningful changes
                        denom = max(len(sentence), len(candidate)) or 1
                        similarity = 1.0 - (self._calculate_edit_distance(sentence, candidate) / denom)
                        # Encourage changes by slightly penalizing exact matches
                        adjusted_conf = similarity - (0.05 if candidate.strip() == sentence.strip() else 0.0)
                        if adjusted_conf > best_conf:
                            best_conf = adjusted_conf
                            best_text = candidate
                if best_text is not None:
                    transformer_correction = best_text
                    transformer_confidence = max(0.0, min(1.0, best_conf))
            except Exception as e:
                logger.warning(f"Transformer correction error: {e}")
        
        # Apply brand casing rules on outputs
        corrected_sentence = self.enforce_brand_casing(corrected_sentence)
        if transformer_correction:
            transformer_correction = self.enforce_brand_casing(transformer_correction)

        processing_time = (time.time() - start_time) * 1000
        
        return {
            "original_sentence": sentence,
            "corrected_sentence": corrected_sentence,
            "transformer_correction": transformer_correction,
            "transformer_confidence": transformer_confidence,
            "word_results": results,
            "total_corrections": total_corrections,
            "processing_time_ms": processing_time,
            "overall_confidence": sum(r.confidence for r in results if r.confidence) / len(results) if results else 0.0
        }
    
    def learn_correction(self, original: str, corrected: str, context: str = ""):
        """Learn a new correction from user feedback"""
        self.learning_db.add_user_correction(original, corrected, context)
        
        # Add to RAG knowledge base
        correction_data = {
            'incorrect': original,
            'correct': corrected,
            'context': context,
            'timestamp': datetime.now().isoformat()
        }
        self.rag_checker.add_knowledge([correction_data])
        
        logger.info(f"Learned correction: '{original}' -> '{corrected}'")
    
    def add_to_whitelist(self, word: str):
        """Add word to whitelist"""
        self.learning_db.add_to_whitelist(word)
        logger.info(f"Added '{word}' to whitelist")
    
    def ignore_word(self, word: str, context: str = ""):
        """Add word to ignore list"""
        self.learning_db.add_to_ignored(word, context)
        logger.info(f"Added '{word}' to ignore list")

# Example usage and testing
def main():
    """Test the advanced spell checker"""
    checker = MultiEngineSpellChecker()
    
    # Test words
    test_words = ["appliction", "recieve", "seperate", "fhdhfdhdf", "hello", "worng"]
    
    print("=== Word-Level Spell Checking ===")
    for word in test_words:
        result = checker.check_word(word)
        print(f"\nWord: '{word}'")
        print(f"Is Correct: {result.is_correct}")
        print(f"Processing Time: {result.processing_time_ms:.2f}ms")
        
        if result.suggestions:
            print("Suggestions:")
            for i, suggestion in enumerate(result.suggestions[:3], 1):
                print(f"  {i}. {suggestion.word} (confidence: {suggestion.confidence:.3f}, "
                      f"score: {suggestion.final_score:.3f}, source: {suggestion.source})")
    
    # Test sentence
    print("\n=== Sentence-Level Spell Checking ===")
    test_sentence = "This is a sampel text with some erors and worng speling."
    sentence_result = checker.check_sentence(test_sentence)
    
    print(f"Original: {sentence_result['original_sentence']}")
    print(f"Corrected: {sentence_result['corrected_sentence']}")
    print(f"Total Corrections: {sentence_result['total_corrections']}")
    print(f"Processing Time: {sentence_result['processing_time_ms']:.2f}ms")
    print(f"Overall Confidence: {sentence_result['overall_confidence']:.3f}")
    
    # Test learning
    print("\n=== Learning Example ===")
    checker.learn_correction("teh", "the", "common typo")
    checker.add_to_whitelist("API")
    checker.ignore_word("JavaScript")
    
    # Test learned correction
    learned_result = checker.check_word("teh")
    print(f"Learned correction for 'teh': {learned_result.corrected_text}")

if __name__ == "__main__":
    main()