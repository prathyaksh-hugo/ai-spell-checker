# main.py - Production-Ready Intelligent Spell Checker
import asyncio
import logging
import time
import sqlite3
from contextlib import asynccontextmanager
from typing import List, Dict, Optional, Tuple, Any
import json
import hashlib
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
import threading

# Core imports
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
import re
from difflib import SequenceMatcher
from collections import Counter, defaultdict
from cachetools import TTLCache
import numpy as np

# Optional imports with fallbacks
try:
    import enchant
    ENCHANT_AVAILABLE = True
except ImportError:
    enchant = None
    ENCHANT_AVAILABLE = False

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    redis = None
    REDIS_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SEMANTIC_MODEL_AVAILABLE = True
except ImportError:
    SentenceTransformer = None
    SEMANTIC_MODEL_AVAILABLE = False

try:
    import phonetics
    PHONETICS_AVAILABLE = True
except ImportError:
    phonetics = None
    PHONETICS_AVAILABLE = False

try:
    from rapidfuzz import fuzz, process
    RAPIDFUZZ_AVAILABLE = True
except ImportError:
    fuzz = process = None
    RAPIDFUZZ_AVAILABLE = False

# --- CONFIGURATION MANAGEMENT ---
class Config:
    # Model settings
    PRIMARY_MODEL = 'ai-forever/T5-large-spell'
    BACKUP_MODEL = 't5-small'
    SEMANTIC_MODEL = 'all-MiniLM-L6-v2'
    
    # Performance settings
    MAX_WORKERS = 4
    CACHE_TTL = 3600  # 1 hour
    CACHE_SIZE = 10000
    BATCH_SIZE = 32
    MAX_SUGGESTIONS = 10
    
    # Quality thresholds
    MIN_CONFIDENCE = 0.3
    MIN_SIMILARITY = 0.4
    MAX_EDIT_DISTANCE = 3
    
    # Redis settings (optional)
    REDIS_URL = "redis://localhost:6379"
    USE_REDIS = False
    
    # Security
    RATE_LIMIT = 1000  # requests per hour per IP
    MAX_TEXT_LENGTH = 1000

# --- ENHANCED LOGGING ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('spellchecker.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- PERFORMANCE MONITORING ---
class PerformanceMonitor:
    def __init__(self):
        self.request_count = 0
        self.total_processing_time = 0
        self.error_count = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.lock = threading.Lock()
    
    def record_request(self, processing_time: float, from_cache: bool = False, error: bool = False):
        with self.lock:
            self.request_count += 1
            self.total_processing_time += processing_time
            if error:
                self.error_count += 1
            if from_cache:
                self.cache_hits += 1
            else:
                self.cache_misses += 1
    
    def get_stats(self) -> Dict:
        with self.lock:
            return {
                "total_requests": self.request_count,
                "average_processing_time": self.total_processing_time / max(self.request_count, 1),
                "error_rate": self.error_count / max(self.request_count, 1),
                "cache_hit_rate": self.cache_hits / max(self.request_count, 1),
                "uptime_minutes": (datetime.now() - start_time).total_seconds() / 60
            }

# --- ENHANCED CACHING SYSTEM ---
class CacheManager:
    def __init__(self):
        self.local_cache = TTLCache(maxsize=Config.CACHE_SIZE, ttl=Config.CACHE_TTL)
        self.redis_client = None
        
        # Try to connect to Redis
        if Config.USE_REDIS and REDIS_AVAILABLE:
            try:
                self.redis_client = redis.Redis.from_url(Config.REDIS_URL, decode_responses=True)
                self.redis_client.ping()
                logger.info("Redis cache connected successfully")
            except Exception as e:
                logger.warning(f"Redis connection failed: {e}. Using local cache only.")
                self.redis_client = None
    
    def _get_cache_key(self, text: str, language: str = "en") -> str:
        """Generate consistent cache key"""
        key_data = f"{text.lower().strip()}:{language}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    async def get(self, text: str, language: str = "en") -> Optional[Dict]:
        """Get cached result"""
        key = self._get_cache_key(text, language)
        
        # Try Redis first
        if self.redis_client:
            try:
                cached = await asyncio.get_event_loop().run_in_executor(
                    None, self.redis_client.get, key
                )
                if cached:
                    return json.loads(cached)
            except Exception as e:
                logger.warning(f"Redis get error: {e}")
        
        # Fallback to local cache
        return self.local_cache.get(key)
    
    async def set(self, text: str, result: Dict, language: str = "en"):
        """Set cache result"""
        key = self._get_cache_key(text, language)
        result_json = json.dumps(result)
        
        # Set in Redis
        if self.redis_client:
            try:
                await asyncio.get_event_loop().run_in_executor(
                    None, self.redis_client.setex, key, Config.CACHE_TTL, result_json
                )
            except Exception as e:
                logger.warning(f"Redis set error: {e}")
        
        # Set in local cache
        self.local_cache[key] = result

# --- CUSTOM DICTIONARY MANAGER ---
class CustomDictionaryManager:
    def __init__(self):
        self.db_path = "custom_dictionary.db"
        self.init_database()
        self.custom_words = set()
        self.domain_corrections = {}
        self.load_custom_data()
    
    def init_database(self):
        """Initialize SQLite database for custom dictionaries"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS custom_words (
                    word TEXT PRIMARY KEY,
                    frequency INTEGER DEFAULT 1,
                    domain TEXT,
                    added_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS corrections (
                    incorrect TEXT PRIMARY KEY,
                    correct TEXT,
                    confidence REAL,
                    usage_count INTEGER DEFAULT 1,
                    last_used TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
    
    def load_custom_data(self):
        """Load custom dictionaries from database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Load custom words
                cursor = conn.execute("SELECT word FROM custom_words")
                self.custom_words = {row[0].lower() for row in cursor.fetchall()}
                
                # Load domain corrections
                cursor = conn.execute("SELECT incorrect, correct, confidence FROM corrections")
                for incorrect, correct, confidence in cursor.fetchall():
                    self.domain_corrections[incorrect.lower()] = (correct, confidence)
                    
                logger.info(f"Loaded {len(self.custom_words)} custom words and {len(self.domain_corrections)} corrections")
        except Exception as e:
            logger.error(f"Error loading custom data: {e}")
    
    def add_word(self, word: str, domain: str = "general"):
        """Add word to custom dictionary"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO custom_words (word, domain) VALUES (?, ?)",
                    (word.lower(), domain)
                )
                self.custom_words.add(word.lower())
        except Exception as e:
            logger.error(f"Error adding custom word: {e}")
    
    def add_correction(self, incorrect: str, correct: str, confidence: float = 1.0):
        """Add custom correction"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO corrections (incorrect, correct, confidence) VALUES (?, ?, ?)",
                    (incorrect.lower(), correct.lower(), confidence)
                )
                self.domain_corrections[incorrect.lower()] = (correct.lower(), confidence)
        except Exception as e:
            logger.error(f"Error adding correction: {e}")
    
    def is_custom_word(self, word: str) -> bool:
        """Check if word is in custom dictionary"""
        return word.lower() in self.custom_words
    
    def get_custom_correction(self, word: str) -> Optional[Tuple[str, float]]:
        """Get custom correction if available"""
        return self.domain_corrections.get(word.lower())

# --- ENHANCED AI MODELS MANAGER ---
class AIModelsManager:
    def __init__(self):
        self.primary_model = None
        self.primary_tokenizer = None
        self.semantic_model = None
        self.backup_model = None
        self.backup_tokenizer = None
        self.model_lock = threading.Lock()
        self.load_models()
    
    def load_models(self):
        """Load all AI models"""
        try:
            logger.info("Loading primary T5 model...")
            self.primary_model = T5ForConditionalGeneration.from_pretrained(Config.PRIMARY_MODEL)
            self.primary_tokenizer = T5Tokenizer.from_pretrained(Config.PRIMARY_MODEL)
            
            if SEMANTIC_MODEL_AVAILABLE:
                logger.info("Loading semantic similarity model...")
                self.semantic_model = SentenceTransformer(Config.SEMANTIC_MODEL)
            
            logger.info("All models loaded successfully!")
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise
    
    def get_ai_suggestions(self, word: str, context: str = "", num_suggestions: int = 5) -> List[Tuple[str, float]]:
        """Get AI suggestions with confidence scores"""
        suggestions = []
        
        # Prepare input with context if available
        if context:
            input_text = f"Context: {context}. Correct spelling: {word}"
        else:
            input_text = f"Correct spelling: {word}"
        
        try:
            with self.model_lock:
                inputs = self.primary_tokenizer.encode(
                    input_text, 
                    return_tensors='pt', 
                    max_length=128, 
                    truncation=True
                )
                
                with torch.no_grad():
                    outputs = self.primary_model.generate(
                        inputs,
                        max_length=inputs.shape[-1] + 15,
                        num_beams=num_suggestions * 2,
                        num_return_sequences=num_suggestions,
                        early_stopping=True,
                        repetition_penalty=2.0,
                        do_sample=False
                    )
                
                # Extract suggestions
                for sequence in outputs:
                    decoded = self.primary_tokenizer.decode(sequence, skip_special_tokens=True)
                    cleaned = self._clean_ai_output(decoded, word)
                    if cleaned and self._is_valid_suggestion(cleaned, word):
                        confidence = 0.8  # Default confidence
                        suggestions.append((cleaned, confidence))
        
        except Exception as e:
            logger.error(f"AI suggestion error: {e}")
        
        return suggestions[:num_suggestions]
    
    def get_semantic_similarity(self, word1: str, word2: str) -> float:
        """Get semantic similarity between two words"""
        if not SEMANTIC_MODEL_AVAILABLE or not self.semantic_model:
            return 0.0
        
        try:
            embeddings = self.semantic_model.encode([word1, word2])
            similarity = np.dot(embeddings[0], embeddings[1]) / (
                np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
            )
            return float(similarity)
        except Exception as e:
            logger.error(f"Semantic similarity error: {e}")
            return 0.0
    
    def _clean_ai_output(self, raw_output: str, original: str) -> str:
        """Enhanced cleaning of AI output"""
        cleaned = raw_output.strip()
        
        # Remove prefixes
        prefixes = [
            "Context:", "Correct spelling:", "Fix:", "Spelling correction:",
            "Corrected:", "Should be:", "->"
        ]
        
        for prefix in prefixes:
            if cleaned.startswith(prefix):
                cleaned = cleaned[len(prefix):].strip()
                break
        
        # Extract first valid word
        words = re.findall(r'\b[a-zA-Z]+\b', cleaned)
        if words:
            candidate = words[0].lower()
            if len(candidate) > 1 and candidate != original.lower():
                return candidate
        
        return ""
    
    def _is_valid_suggestion(self, suggestion: str, original: str) -> bool:
        """Validate suggestion quality"""
        if not suggestion or len(suggestion) < 2:
            return False
        
        # Filter obvious artifacts
        invalid_words = {'spelling', 'correct', 'fix', 'context', 'should', 'be'}
        if suggestion.lower() in invalid_words:
            return False
        
        # Check minimum similarity
        similarity = SequenceMatcher(None, original.lower(), suggestion.lower()).ratio()
        return similarity > Config.MIN_SIMILARITY

# --- ADVANCED SUGGESTION ENGINE ---
class AdvancedSuggestionEngine:
    def __init__(self, ai_manager: AIModelsManager, dict_manager: CustomDictionaryManager):
        self.ai_manager = ai_manager
        self.dict_manager = dict_manager
        
        # Initialize dictionaries
        if ENCHANT_AVAILABLE:
            try:
                self.us_dict = enchant.Dict("en_US")
                self.uk_dict = enchant.Dict("en_GB")
            except:
                self.us_dict = self.uk_dict = None
                logger.warning("Enchant dictionaries not available")
        else:
            self.us_dict = self.uk_dict = None
            logger.warning("Enchant not available")
    
    def get_comprehensive_suggestions(self, word: str, context: str = "", language: str = "en") -> List[Dict]:
        """Get comprehensive suggestions from all sources"""
        suggestions = []
        
        # 1. Check custom corrections first
        custom_correction = self.dict_manager.get_custom_correction(word)
        if custom_correction:
            suggestions.append({
                'word': custom_correction[0],
                'confidence': custom_correction[1],
                'source': 'custom',
                'edit_distance': self._edit_distance(word, custom_correction[0])
            })
        
        # 2. Dictionary suggestions
        dict_suggestions = self._get_dictionary_suggestions(word)
        suggestions.extend(dict_suggestions)
        
        # 3. AI suggestions with context
        ai_suggestions = self.ai_manager.get_ai_suggestions(word, context)
        for suggestion, confidence in ai_suggestions:
            suggestions.append({
                'word': suggestion,
                'confidence': confidence,
                'source': 'ai',
                'edit_distance': self._edit_distance(word, suggestion)
            })
        
        # 4. Phonetic suggestions
        phonetic_suggestions = self._get_phonetic_suggestions(word)
        suggestions.extend(phonetic_suggestions)
        
        # 5. Fuzzy matching suggestions
        fuzzy_suggestions = self._get_fuzzy_suggestions(word)
        suggestions.extend(fuzzy_suggestions)
        
        # 6. Rank and filter suggestions
        final_suggestions = self._rank_and_filter_suggestions(suggestions, word)
        
        return final_suggestions[:Config.MAX_SUGGESTIONS]
    
    def _get_dictionary_suggestions(self, word: str) -> List[Dict]:
        """Get suggestions from dictionaries"""
        suggestions = []
        
        if self.us_dict:
            us_suggestions = self.us_dict.suggest(word)[:5]
            for suggestion in us_suggestions:
                suggestions.append({
                    'word': suggestion.lower(),
                    'confidence': 0.9,
                    'source': 'dictionary_us',
                    'edit_distance': self._edit_distance(word, suggestion)
                })
        
        if self.uk_dict:
            uk_suggestions = self.uk_dict.suggest(word)[:5]
            for suggestion in uk_suggestions:
                if suggestion.lower() not in [s['word'] for s in suggestions]:
                    suggestions.append({
                        'word': suggestion.lower(),
                        'confidence': 0.85,
                        'source': 'dictionary_uk',
                        'edit_distance': self._edit_distance(word, suggestion)
                    })
        
        return suggestions
    
    def _get_phonetic_suggestions(self, word: str) -> List[Dict]:
        """Get phonetically similar suggestions"""
        suggestions = []
        
        if not PHONETICS_AVAILABLE:
            return suggestions
        
        try:
            # Generate phonetic variations
            phonetic_variants = self._generate_phonetic_variants(word)
            
            for variant in phonetic_variants:
                if self._is_valid_word(variant) and variant != word.lower():
                    suggestions.append({
                        'word': variant,
                        'confidence': 0.7,
                        'source': 'phonetic',
                        'edit_distance': self._edit_distance(word, variant)
                    })
        except Exception as e:
            logger.error(f"Phonetic suggestion error: {e}")
        
        return suggestions[:3]
    
    def _get_fuzzy_suggestions(self, word: str) -> List[Dict]:
        """Get fuzzy matching suggestions"""
        suggestions = []
        
        if not RAPIDFUZZ_AVAILABLE:
            return suggestions
        
        # Common English words for fuzzy matching
        common_words = [
            'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had',
            'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 'how',
            'its', 'may', 'new', 'now', 'old', 'see', 'two', 'way', 'who', 'boy',
            'did', 'man', 'car', 'use', 'her', 'oil', 'sit', 'set', 'run', 'eat'
        ]
        
        # Add custom words
        if hasattr(self.dict_manager, 'custom_words'):
            common_words.extend(list(self.dict_manager.custom_words)[:100])
        
        try:
            # Use rapidfuzz for fast fuzzy matching
            matches = process.extract(word, common_words, limit=5, scorer=fuzz.WRatio)
            
            for match, score in matches:
                if score > 70:  # Only high-confidence matches
                    suggestions.append({
                        'word': match,
                        'confidence': score / 100.0,
                        'source': 'fuzzy',
                        'edit_distance': self._edit_distance(word, match)
                    })
        except Exception as e:
            logger.error(f"Fuzzy matching error: {e}")
        
        return suggestions
    
    def _generate_phonetic_variants(self, word: str) -> List[str]:
        """Generate phonetic variants of a word"""
        variants = []
        
        phonetic_rules = [
            ('ph', 'f'), ('f', 'ph'), ('c', 'k'), ('k', 'c'),
            ('z', 's'), ('s', 'z'), ('i', 'y'), ('y', 'i'),
            ('ei', 'ie'), ('ie', 'ei'), ('tion', 'sion'), ('sion', 'tion')
        ]
        
        for old, new in phonetic_rules:
            if old in word.lower():
                variant = word.lower().replace(old, new)
                variants.append(variant)
        
        return variants
    
    def _is_valid_word(self, word: str) -> bool:
        """Check if word is valid"""
        if len(word) < 2 or not word.isalpha():
            return False
        
        # Check in dictionaries
        if self.us_dict and self.us_dict.check(word):
            return True
        if self.uk_dict and self.uk_dict.check(word):
            return True
        
        # Check in custom dictionary
        if self.dict_manager.is_custom_word(word):
            return True
        
        return False
    
    def _edit_distance(self, word1: str, word2: str) -> int:
        """Calculate edit distance between two words"""
        m, n = len(word1), len(word2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if word1[i-1] == word2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
        
        return dp[m][n]
    
    def _rank_and_filter_suggestions(self, suggestions: List[Dict], original_word: str) -> List[Dict]:
        """Rank and filter suggestions by quality"""
        # Remove duplicates
        seen = set()
        unique_suggestions = []
        for suggestion in suggestions:
            if suggestion['word'] not in seen:
                unique_suggestions.append(suggestion)
                seen.add(suggestion['word'])
        
        # Filter by quality thresholds
        filtered = [
            s for s in unique_suggestions 
            if s['confidence'] >= Config.MIN_CONFIDENCE and 
            s['edit_distance'] <= Config.MAX_EDIT_DISTANCE
        ]
        
        # Calculate composite scores
        for suggestion in filtered:
            # Combine multiple factors
            edit_penalty = suggestion['edit_distance'] / max(len(original_word), len(suggestion['word']))
            length_penalty = abs(len(original_word) - len(suggestion['word'])) / max(len(original_word), len(suggestion['word']))
            
            # Source-based weighting
            source_weights = {
                'custom': 1.0,
                'dictionary_us': 0.95,
                'dictionary_uk': 0.9,
                'ai': 0.8,
                'phonetic': 0.7,
                'fuzzy': 0.6
            }
            
            source_weight = source_weights.get(suggestion['source'], 0.5)
            
            # Final composite score
            suggestion['final_score'] = (
                suggestion['confidence'] * source_weight * 
                (1 - edit_penalty * 0.3) * 
                (1 - length_penalty * 0.2)
            )
        
        # Sort by final score
        filtered.sort(key=lambda x: x['final_score'], reverse=True)
        
        return filtered

# --- RATE LIMITING ---
class RateLimiter:
    def __init__(self):
        self.requests = defaultdict(list)
        self.lock = threading.Lock()
    
    def is_allowed(self, client_ip: str) -> bool:
        """Check if request is allowed based on rate limit"""
        now = datetime.now()
        hour_ago = now - timedelta(hours=1)
        
        with self.lock:
            # Clean old requests
            self.requests[client_ip] = [
                req_time for req_time in self.requests[client_ip] 
                if req_time > hour_ago
            ]
            
            # Check rate limit
            if len(self.requests[client_ip]) >= Config.RATE_LIMIT:
                return False
            
            # Add current request
            self.requests[client_ip].append(now)
            return True

# --- ENHANCED REQUEST/RESPONSE MODELS ---
class SpellCheckRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=Config.MAX_TEXT_LENGTH)
    context: Optional[str] = Field(None, max_length=500)
    language: Optional[str] = Field("en", pattern="^(en|en_US|en_GB)$")  # Fixed: regex -> pattern
    max_suggestions: Optional[int] = Field(5, ge=1, le=Config.MAX_SUGGESTIONS)
    include_confidence: Optional[bool] = True
    
    @validator('text')
    def validate_text(cls, v):
        if not v.strip():
            raise ValueError('Text cannot be empty')
        return v.strip()

class SpellCheckResponse(BaseModel):
    is_correct: bool
    original_text: str
    suggestions: List[Dict[str, Any]]
    processing_time_ms: float
    language_detected: Optional[str]
    cache_hit: bool
    confidence_threshold: float

class BatchSpellCheckRequest(BaseModel):
    texts: List[str] = Field(..., min_items=1, max_items=100)
    context: Optional[str] = None
    language: Optional[str] = "en"
    max_suggestions: Optional[int] = 5

class AddWordRequest(BaseModel):
    word: str = Field(..., min_length=1, max_length=100)
    domain: str = Field("general", max_length=50)

class AddCorrectionRequest(BaseModel):
    incorrect: str = Field(..., min_length=1, max_length=100)
    correct: str = Field(..., min_length=1, max_length=100)
    confidence: float = Field(1.0, ge=0.0, le=1.0)

# --- UTILITY FUNCTIONS ---
def get_client_ip(request: Request) -> str:
    """Extract client IP from request"""
    return request.client.host

async def check_rate_limit(request: Request) -> str:
    """Check rate limiting"""
    client_ip = get_client_ip(request)
    if not rate_limiter.is_allowed(client_ip):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    return client_ip

# --- MAIN APPLICATION SETUP ---
start_time = datetime.now()
performance_monitor = PerformanceMonitor()
cache_manager = CacheManager()
dict_manager = CustomDictionaryManager()
ai_manager = AIModelsManager()
suggestion_engine = AdvancedSuggestionEngine(ai_manager, dict_manager)
rate_limiter = RateLimiter()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    logger.info("Starting Enhanced Spell Checker API")
    yield
    logger.info("Shutting down Enhanced Spell Checker API")

app = FastAPI(
    title="Advanced Spell Checker API",
    description="Production-ready intelligent spell checking service",
    version="2.0.0",
    lifespan=lifespan
)

# --- MIDDLEWARE ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]  # Configure appropriately for production
)

# --- API ENDPOINTS ---
@app.post("/spellcheck", response_model=SpellCheckResponse)
async def spell_check(
    spell_request: SpellCheckRequest,
    background_tasks: BackgroundTasks,
    request: Request,
    client_ip: str = Depends(check_rate_limit)
):
    """Enhanced spell checking endpoint"""
    start_time = time.time()
    cache_hit = False
    
    try:
        # Check cache first
        cached_result = await cache_manager.get(spell_request.text, spell_request.language)
        if cached_result:
            cache_hit = True
            performance_monitor.record_request(0, from_cache=True)
            return SpellCheckResponse(**cached_result, cache_hit=True)
        
        # Check if word is already correct
        word = spell_request.text.lower().strip()
        is_correct = dict_manager.is_custom_word(word)
        
        if not is_correct and suggestion_engine.us_dict:
            is_correct = suggestion_engine.us_dict.check(word) or suggestion_engine.uk_dict.check(word)
        
        suggestions = []
        if not is_correct:
            # Get comprehensive suggestions
            suggestion_results = suggestion_engine.get_comprehensive_suggestions(
                word, spell_request.context or "", spell_request.language
            )
            
            # Format suggestions
            suggestions = [
                {
                    "word": s["word"],
                    "confidence": s["final_score"],
                    "source": s["source"],
                    "edit_distance": s["edit_distance"]
                }
                for s in suggestion_results[:spell_request.max_suggestions]
            ]
        
        # Prepare response
        processing_time = (time.time() - start_time) * 1000
        response_data = {
            "is_correct": is_correct,
            "original_text": spell_request.text,
            "suggestions": suggestions,
            "processing_time_ms": processing_time,
            "language_detected": spell_request.language,
            "cache_hit": cache_hit,
            "confidence_threshold": Config.MIN_CONFIDENCE
        }
        
        # Cache result
        background_tasks.add_task(
            cache_manager.set, spell_request.text, response_data, spell_request.language
        )
        
        # Record performance
        performance_monitor.record_request(processing_time / 1000)
        
        return SpellCheckResponse(**response_data)
        
    except Exception as e:
        performance_monitor.record_request((time.time() - start_time) * 1000, error=True)
        logger.error(f"Spell check error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/batch-spellcheck")
async def batch_spell_check(
    batch_request: BatchSpellCheckRequest,
    request: Request,
    client_ip: str = Depends(check_rate_limit)
):
    """Batch spell checking endpoint"""
    start_time = time.time()
    
    try:
        results = []
        
        # Process in parallel using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=Config.MAX_WORKERS) as executor:
            # Create individual requests
            individual_requests = [
                SpellCheckRequest(
                    text=text,
                    context=batch_request.context,
                    language=batch_request.language,
                    max_suggestions=batch_request.max_suggestions
                )
                for text in batch_request.texts
            ]
            
            # Process concurrently
            futures = [
                executor.submit(
                    suggestion_engine.get_comprehensive_suggestions,
                    req.text.lower().strip(),
                    req.context or "",
                    req.language
                )
                for req in individual_requests
            ]
            
            # Collect results
            for i, future in enumerate(futures):
                try:
                    suggestions = future.result(timeout=10)  # 10 second timeout
                    results.append({
                        "text": batch_request.texts[i],
                        "is_correct": len(suggestions) == 0,
                        "suggestions": suggestions[:batch_request.max_suggestions]
                    })
                except Exception as e:
                    logger.error(f"Batch processing error for '{batch_request.texts[i]}': {e}")
                    results.append({
                        "text": batch_request.texts[i],
                        "error": str(e)
                    })
        
        processing_time = (time.time() - start_time) * 1000
        
        return {
            "results": results,
            "total_processed": len(batch_request.texts),
            "processing_time_ms": processing_time
        }
        
    except Exception as e:
        logger.error(f"Batch spell check error: {e}")
        raise HTTPException(status_code=500, detail="Batch processing failed")

@app.post("/add-custom-word")
async def add_custom_word(
    word_request: AddWordRequest,
    request: Request,
    client_ip: str = Depends(check_rate_limit)
):
    """Add word to custom dictionary"""
    try:
        dict_manager.add_word(word_request.word, word_request.domain)
        return {
            "message": f"Word '{word_request.word}' added to custom dictionary", 
            "domain": word_request.domain
        }
    except Exception as e:
        logger.error(f"Error adding custom word: {e}")
        raise HTTPException(status_code=500, detail="Failed to add custom word")

@app.post("/add-custom-correction")
async def add_custom_correction(
    correction_request: AddCorrectionRequest,
    request: Request,
    client_ip: str = Depends(check_rate_limit)
):
    """Add custom correction"""
    try:
        dict_manager.add_correction(
            correction_request.incorrect, 
            correction_request.correct, 
            correction_request.confidence
        )
        return {
            "message": f"Correction '{correction_request.incorrect}' -> '{correction_request.correct}' added",
            "confidence": correction_request.confidence
        }
    except Exception as e:
        logger.error(f"Error adding custom correction: {e}")
        raise HTTPException(status_code=500, detail="Failed to add custom correction")

@app.get("/stats")
async def get_stats():
    """Get system statistics"""
    return {
        "performance": performance_monitor.get_stats(),
        "cache_info": {
            "local_cache_size": len(cache_manager.local_cache),
            "redis_connected": cache_manager.redis_client is not None
        },
        "dictionary_info": {
            "custom_words": len(dict_manager.custom_words),
            "custom_corrections": len(dict_manager.domain_corrections),
            "enchant_available": suggestion_engine.us_dict is not None
        },
        "features": {
            "enchant_available": ENCHANT_AVAILABLE,
            "redis_available": REDIS_AVAILABLE,
            "semantic_model_available": SEMANTIC_MODEL_AVAILABLE,
            "phonetics_available": PHONETICS_AVAILABLE,
            "rapidfuzz_available": RAPIDFUZZ_AVAILABLE
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "uptime_seconds": (datetime.now() - start_time).total_seconds()
    }

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "Advanced Spell Checker API",
        "version": "2.0.0",
        "status": "running",
        "features": [
            "Multi-source suggestions",
            "Custom dictionaries",
            "Batch processing",
            "Intelligent caching",
            "Rate limiting",
            "Performance monitoring"
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        workers=1,
        access_log=True
    )
