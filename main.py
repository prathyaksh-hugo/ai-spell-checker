# main.py - Complete Hugosave Brand-Optimized Spell Checker with Quantization
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
import os

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
    # Model settings - Hugosave optimized
    PRIMARY_MODEL = os.getenv('PRIMARY_MODEL', './hugosave-quantized-model')
    FALLBACK_MODEL = 'ai-forever/T5-large-spell'
    SEMANTIC_MODEL = 'all-MiniLM-L6-v2'
    
    # Quantization settings
    QUANTIZATION_METHOD = os.getenv('QUANTIZATION_METHOD', 'dynamic')
    USE_HUGOSAVE_MODEL = os.getenv('USE_HUGOSAVE_MODEL', 'true').lower() == 'true'
    
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
    REDIS_URL = os.getenv('REDIS_URL', "redis://localhost:6379")
    USE_REDIS = os.getenv('USE_REDIS', 'false').lower() == 'true'
    
    # Security
    RATE_LIMIT = 1000  # requests per hour per IP
    MAX_TEXT_LENGTH = 1000

# --- ENHANCED LOGGING ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('hugosave_spellchecker.log'),
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
        self.hugosave_corrections = 0
        self.lock = threading.Lock()
    
    def record_request(self, processing_time: float, from_cache: bool = False, error: bool = False, hugosave_correction: bool = False):
        with self.lock:
            self.request_count += 1
            self.total_processing_time += processing_time
            if error:
                self.error_count += 1
            if from_cache:
                self.cache_hits += 1
            else:
                self.cache_misses += 1
            if hugosave_correction:
                self.hugosave_corrections += 1
    
    def get_stats(self) -> Dict:
        with self.lock:
            return {
                "total_requests": self.request_count,
                "average_processing_time": self.total_processing_time / max(self.request_count, 1),
                "error_rate": self.error_count / max(self.request_count, 1),
                "cache_hit_rate": self.cache_hits / max(self.request_count, 1),
                "hugosave_correction_rate": self.hugosave_corrections / max(self.request_count, 1),
                "uptime_minutes": (datetime.now() - start_time).total_seconds() / 60
            }

# --- HUGOSAVE BRAND GUIDELINES MANAGER ---
class HugosaveBrandManager:
    def __init__(self):
        self.brand_terms = self._load_hugosave_terms()
        self.ux_writing_rules = self._load_ux_rules()
        self.style_rules = self._load_style_rules()
    
    def _load_hugosave_terms(self) -> Dict[str, str]:
        """Load Hugosave brand terms"""
        return {
            'hugosave': 'Hugosave',
            'hugohero': 'Hugohero',
            'hugoheroes': 'Hugoheroes',
            'wealthcare': 'Wealthcare',
            'wealthcare®': 'Wealthcare®',
            'homescreen': 'Homescreen',
            'save account': 'Save Account',
            'spend account': 'Spend Account',
            'cash account': 'Cash Account',
            'multi-currency account': 'Multi-currency Account',
            'debit card': 'Debit Card',
            'hugosave visa platinum debit card': 'Hugosave Visa Platinum Debit Card',
            'pots': 'Pots',
            'rewards centre': 'Rewards Centre',
            'quests': 'Quests',
            'referrals': 'Referrals',
            'roundups': 'Roundups',
            'auto top-up': 'Auto Top-up',
            'invest-as-you-spend': 'Invest-as-you-spend',
            'net worth': 'Net Worth',
            'portfolio composition': 'Portfolio Composition',
            'investment personality quiz': 'Investment Personality Quiz',
            'defender': 'Defender',
            'mediator': 'Mediator',
            'adventurer': 'Adventurer',
            'kyc': 'KYC',
            'singpass': 'Singpass',
            'edda': 'eDDA',
            't&cs': 'T&Cs',
            'hugohub': 'HugoHub',
            'customer': 'Customer',
            'end-customer': 'End-Customer',
            'wallet': 'Wallet'
        }
    
    def _load_ux_rules(self) -> Dict[str, str]:
        """Load UX writing rules"""
        return {
            'login': 'log in',
            'logout': 'log out',
            'signup': 'sign up',
            'setup': 'set up',
            'backup': 'back up',
            'click on': 'select',
            'please click': 'select',
            'kindly': '',
            'utilize': 'use',
            'facilitate': 'help',
            'commence': 'start',
            'terminate': 'end'
        }
    
    def _load_style_rules(self) -> Dict[str, str]:
        """Load British English style rules"""
        return {
            'color': 'colour',
            'realize': 'realise',
            'organize': 'organise',
            'analyze': 'analyse',
            'center': 'centre',
            'favor': 'favour',
            'honor': 'honour',
            'labor': 'labour',
            'neighbor': 'neighbour'
        }
    
    def get_hugosave_correction(self, word: str) -> Optional[Tuple[str, str, float]]:
        """Get Hugosave-specific correction"""
        word_lower = word.lower()
        
        # Check brand terms
        if word_lower in self.brand_terms:
            return (self.brand_terms[word_lower], 'hugosave_brand', 1.0)
        
        # Check UX writing rules
        if word_lower in self.ux_writing_rules:
            return (self.ux_writing_rules[word_lower], 'ux_writing', 0.95)
        
        # Check style rules
        if word_lower in self.style_rules:
            return (self.style_rules[word_lower], 'british_english', 0.9)
        
        return None
    
    def is_hugosave_term(self, word: str) -> bool:
        """Check if word is a Hugosave term"""
        return word.lower() in {**self.brand_terms, **self.ux_writing_rules, **self.style_rules}

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
        key_data = f"{text.lower().strip()}:{language}:hugosave"
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
        self.db_path = "hugosave_dictionary.db"
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

# --- HUGOSAVE QUANTIZED AI MODELS MANAGER ---
class HugosaveQuantizedAIManager:
    def __init__(self, quantized_model_path: str = None):
        self.model_path = quantized_model_path or Config.PRIMARY_MODEL
        self.quantized_model = None
        self.tokenizer = None
        self.model_lock = threading.Lock()
        self.model_info = {}
        
        self.load_hugosave_model()
    
    def load_hugosave_model(self):
        """Load Hugosave-specific quantized model"""
        logger.info("🚀 Loading Hugosave quantized model...")
        
        try:
            # Check if Hugosave model exists
            if os.path.exists(self.model_path) and Config.USE_HUGOSAVE_MODEL:
                self._load_quantized_hugosave_model()
            else:
                logger.warning("Hugosave model not found, falling back to base model")
                self._load_fallback_model()
                
        except Exception as e:
            logger.error(f"Error loading Hugosave model: {e}")
            logger.info("Loading fallback model...")
            self._load_fallback_model()
    
    def _load_quantized_hugosave_model(self):
        """Load the quantized Hugosave model"""
        try:
            self.tokenizer = T5Tokenizer.from_pretrained(self.model_path)
            
            # Try to load quantized model state
            quantized_state_path = f"{self.model_path}/quantized_model.pth"
            if os.path.exists(quantized_state_path):
                # Load base model and apply quantization
                base_model = T5ForConditionalGeneration.from_pretrained(
                    self.model_path.replace('-quantized', '-fine-tuned')
                )
                
                self.quantized_model = torch.quantization.quantize_dynamic(
                    base_model,
                    {torch.nn.Linear, torch.nn.Embedding},
                    dtype=torch.qint8
                )
            else:
                # Load model and apply quantization
                base_model = T5ForConditionalGeneration.from_pretrained(self.model_path)
                self.quantized_model = torch.quantization.quantize_dynamic(
                    base_model,
                    {torch.nn.Linear, torch.nn.Embedding},
                    dtype=torch.qint8
                )
            
            self.model_info = {
                'model_type': 'hugosave_quantized',
                'quantization': 'dynamic_int8',
                'hugosave_optimized': True,
                'memory_footprint': self._get_model_size_mb(self.quantized_model)
            }
            
            logger.info("✅ Hugosave quantized model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading quantized Hugosave model: {e}")
            raise
    
    def _load_fallback_model(self):
        """Load fallback model if Hugosave model fails"""
        self.tokenizer = T5Tokenizer.from_pretrained(Config.FALLBACK_MODEL)
        base_model = T5ForConditionalGeneration.from_pretrained(Config.FALLBACK_MODEL)
        
        # Apply quantization to fallback model
        self.quantized_model = torch.quantization.quantize_dynamic(
            base_model,
            {torch.nn.Linear, torch.nn.Embedding},
            dtype=torch.qint8
        )
        
        self.model_info = {
            'model_type': 'fallback_quantized',
            'quantization': 'dynamic_int8',
            'hugosave_optimized': False,
            'memory_footprint': self._get_model_size_mb(self.quantized_model)
        }
        
        logger.info("✅ Fallback quantized model loaded")
    
    def get_ai_suggestions(self, word: str, context: str = "", num_suggestions: int = 5) -> List[Tuple[str, float]]:
        """Get suggestions using Hugosave-optimized quantized model"""
        suggestions = []
        
        # Enhanced input format for Hugosave context
        if context:
            input_text = f"Context: {context}. Spelling correction: {word}"
        else:
            input_text = f"Spelling correction: {word}"
        
        try:
            with self.model_lock:
                inputs = self.tokenizer.encode(
                    input_text,
                    return_tensors='pt',
                    max_length=128,
                    truncation=True
                )
                
                with torch.no_grad():
                    outputs = self.quantized_model.generate(
                        inputs,
                        max_length=inputs.shape[-1] + 10,
                        num_beams=3,  # Optimized for speed
                        num_return_sequences=min(num_suggestions, 3),
                        early_stopping=True,
                        repetition_penalty=1.5,
                        do_sample=False,
                        use_cache=True
                    )
                
                for output in outputs:
                    decoded = self.tokenizer.decode(output, skip_special_tokens=True)
                    cleaned = self._clean_ai_output(decoded, word)
                    if cleaned and self._is_valid_suggestion(cleaned, word):
                        # Higher confidence for Hugosave-trained model
                        confidence = 0.95 if self.model_info.get('hugosave_optimized') else 0.8
                        suggestions.append((cleaned, confidence))
        
        except Exception as e:
            logger.error(f"Hugosave AI suggestion error: {e}")
        
        return suggestions[:num_suggestions]
    
    def _get_model_size_mb(self, model):
        """Calculate model size in MB"""
        param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
        buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
        return round((param_size + buffer_size) / 1024 / 1024, 2)
    
    def _clean_ai_output(self, raw_output: str, original: str) -> str:
        """Enhanced cleaning of AI output"""
        cleaned = raw_output.strip()
        
        prefixes = [
            "Context:", "Spelling correction:", "Fix:", "Corrected:",
            "Should be:", "->", "Brand correction:", "Hugosave:"
        ]
        
        for prefix in prefixes:
            if cleaned.startswith(prefix):
                cleaned = cleaned[len(prefix):].strip()
                break
        
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
        
        invalid_words = {'spelling', 'correct', 'fix', 'context', 'should', 'be', 'hugosave', 'brand'}
        if suggestion.lower() in invalid_words:
            return False
        
        similarity = SequenceMatcher(None, original.lower(), suggestion.lower()).ratio()
        return similarity > Config.MIN_SIMILARITY

# --- ADVANCED SUGGESTION ENGINE ---
class AdvancedSuggestionEngine:
    def __init__(self, ai_manager: HugosaveQuantizedAIManager, dict_manager: CustomDictionaryManager, brand_manager: HugosaveBrandManager):
        self.ai_manager = ai_manager
        self.dict_manager = dict_manager
        self.brand_manager = brand_manager
        
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
        """Get comprehensive suggestions with Hugosave brand priority"""
        suggestions = []
        
        # 1. PRIORITY: Check Hugosave brand guidelines first
        hugosave_correction = self.brand_manager.get_hugosave_correction(word)
        if hugosave_correction:
            correct_word, category, confidence = hugosave_correction
            suggestions.append({
                'word': correct_word,
                'confidence': confidence,
                'source': f'hugosave_{category}',
                'edit_distance': self._edit_distance(word, correct_word),
                'hugosave_brand': True
            })
        
        # 2. Check custom corrections
        custom_correction = self.dict_manager.get_custom_correction(word)
        if custom_correction:
            suggestions.append({
                'word': custom_correction[0],
                'confidence': custom_correction[1],
                'source': 'custom',
                'edit_distance': self._edit_distance(word, custom_correction[0])
            })
        
        # 3. Dictionary suggestions
        dict_suggestions = self._get_dictionary_suggestions(word)
        suggestions.extend(dict_suggestions)
        
        # 4. AI suggestions with context (Hugosave-optimized)
        ai_suggestions = self.ai_manager.get_ai_suggestions(word, context)
        for suggestion, confidence in ai_suggestions:
            suggestions.append({
                'word': suggestion,
                'confidence': confidence,
                'source': 'hugosave_ai',
                'edit_distance': self._edit_distance(word, suggestion)
            })
        
        # 5. Phonetic and fuzzy suggestions (if no high-quality suggestions found)
        if len(suggestions) < 3:
            phonetic_suggestions = self._get_phonetic_suggestions(word)
            suggestions.extend(phonetic_suggestions)
            
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
        
        # Include Hugosave terms in fuzzy matching
        common_words = [
            'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had',
            'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 'how'
        ]
        
        # Add Hugosave brand terms
        hugosave_terms = list(self.brand_manager.brand_terms.values())
        common_words.extend(hugosave_terms)
        
        # Add custom words
        if hasattr(self.dict_manager, 'custom_words'):
            common_words.extend(list(self.dict_manager.custom_words)[:100])
        
        try:
            matches = process.extract(word, common_words, limit=5, scorer=fuzz.WRatio)
            
            for match, score in matches:
                if score > 70:
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
        
        # Check if it's a Hugosave term
        if self.brand_manager.is_hugosave_term(word):
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
        """Rank and filter suggestions with Hugosave priority"""
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
        
        # Calculate composite scores with Hugosave priority
        for suggestion in filtered:
            edit_penalty = suggestion['edit_distance'] / max(len(original_word), len(suggestion['word']))
            length_penalty = abs(len(original_word) - len(suggestion['word'])) / max(len(original_word), len(suggestion['word']))
            
            # Enhanced source-based weighting (Hugosave gets highest priority)
            source_weights = {
                'hugosave_hugosave_brand': 1.0,      # Highest priority
                'hugosave_ux_writing': 0.98,
                'hugosave_british_english': 0.95,
                'hugosave_ai': 0.92,
                'custom': 0.9,
                'dictionary_us': 0.85,
                'dictionary_uk': 0.8,
                'phonetic': 0.7,
                'fuzzy': 0.6
            }
            
            source_weight = source_weights.get(suggestion['source'], 0.5)
            
            # Hugosave brand bonus
            hugosave_bonus = 0.1 if suggestion.get('hugosave_brand', False) else 0.0
            
            # Final composite score
            suggestion['final_score'] = (
                suggestion['confidence'] * source_weight * 
                (1 - edit_penalty * 0.2) * 
                (1 - length_penalty * 0.1) +
                hugosave_bonus
            )
        
        # Sort by final score (Hugosave terms will naturally rank higher)
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
    language: Optional[str] = Field("en", pattern="^(en|en_US|en_GB)$")
    max_suggestions: Optional[int] = Field(5, ge=1, le=Config.MAX_SUGGESTIONS)
    include_confidence: Optional[bool] = True
    hugosave_priority: Optional[bool] = True
    
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
    hugosave_optimized: bool
    hugosave_correction: bool

class BatchSpellCheckRequest(BaseModel):
    texts: List[str] = Field(..., min_items=1, max_items=100)
    context: Optional[str] = None
    language: Optional[str] = "en"
    max_suggestions: Optional[int] = 5
    hugosave_priority: Optional[bool] = True

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
brand_manager = HugosaveBrandManager()
ai_manager = HugosaveQuantizedAIManager()
suggestion_engine = AdvancedSuggestionEngine(ai_manager, dict_manager, brand_manager)
rate_limiter = RateLimiter()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    logger.info("🚀 Starting Hugosave Enhanced Spell Checker API")
    logger.info(f"📊 Model Info: {ai_manager.model_info}")
    yield
    logger.info("🛑 Shutting down Hugosave Enhanced Spell Checker API")

app = FastAPI(
    title="Hugosave Advanced Spell Checker API",
    description="Production-ready spell checking with Hugosave brand guidelines and quantization",
    version="3.0.0",
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
    """Enhanced spell checking with Hugosave brand guidelines"""
    start_time_request = time.time()
    cache_hit = False
    hugosave_correction = False
    
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
        
        # Check Hugosave brand terms
        if not is_correct:
            hugosave_correction_result = brand_manager.get_hugosave_correction(word)
            if hugosave_correction_result and hugosave_correction_result[0].lower() == word:
                is_correct = True
        
        # Check dictionaries
        if not is_correct and suggestion_engine.us_dict:
            is_correct = suggestion_engine.us_dict.check(word) or suggestion_engine.uk_dict.check(word)
        
        suggestions = []
        if not is_correct:
            # Get comprehensive suggestions with Hugosave priority
            suggestion_results = suggestion_engine.get_comprehensive_suggestions(
                word, spell_request.context or "", spell_request.language
            )
            
            # Check if any suggestion is a Hugosave correction
            hugosave_correction = any(s.get('hugosave_brand', False) for s in suggestion_results)
            
            # Format suggestions
            suggestions = [
                {
                    "word": s["word"],
                    "confidence": s["final_score"],
                    "source": s["source"],
                    "edit_distance": s["edit_distance"],
                    "hugosave_brand": s.get("hugosave_brand", False)
                }
                for s in suggestion_results[:spell_request.max_suggestions]
            ]
        
        # Prepare response
        processing_time = (time.time() - start_time_request) * 1000
        response_data = {
            "is_correct": is_correct,
            "original_text": spell_request.text,
            "suggestions": suggestions,
            "processing_time_ms": processing_time,
            "language_detected": spell_request.language,
            "cache_hit": cache_hit,
            "confidence_threshold": Config.MIN_CONFIDENCE,
            "hugosave_optimized": ai_manager.model_info.get('hugosave_optimized', False),
            "hugosave_correction": hugosave_correction
        }
        
        # Cache result
        background_tasks.add_task(
            cache_manager.set, spell_request.text, response_data, spell_request.language
        )
        
        # Record performance
        performance_monitor.record_request(processing_time / 1000, hugosave_correction=hugosave_correction)
        
        return SpellCheckResponse(**response_data)
        
    except Exception as e:
        performance_monitor.record_request((time.time() - start_time_request) * 1000, error=True)
        logger.error(f"Spell check error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/batch-spellcheck")
async def batch_spell_check(
    batch_request: BatchSpellCheckRequest,
    request: Request,
    client_ip: str = Depends(check_rate_limit)
):
    """Batch spell checking with Hugosave optimization"""
    start_time_batch = time.time()
    
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
                    max_suggestions=batch_request.max_suggestions,
                    hugosave_priority=batch_request.hugosave_priority
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
                    suggestions = future.result(timeout=10)
                    
                    # Check for Hugosave corrections
                    hugosave_correction = any(s.get('hugosave_brand', False) for s in suggestions)
                    
                    results.append({
                        "text": batch_request.texts[i],
                        "is_correct": len(suggestions) == 0,
                        "suggestions": suggestions[:batch_request.max_suggestions],
                        "hugosave_correction": hugosave_correction
                    })
                except Exception as e:
                    logger.error(f"Batch processing error for '{batch_request.texts[i]}': {e}")
                    results.append({
                        "text": batch_request.texts[i],
                        "error": str(e)
                    })
        
        processing_time = (time.time() - start_time_batch) * 1000
        
        return {
            "results": results,
            "total_processed": len(batch_request.texts),
            "processing_time_ms": processing_time,
            "hugosave_optimized": True
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

@app.get("/hugosave-terms")
async def get_hugosave_terms():
    """Get all Hugosave brand terms"""
    return {
        "brand_terms": brand_manager.brand_terms,
        "ux_writing_rules": brand_manager.ux_writing_rules,
        "style_rules": brand_manager.style_rules,
        "total_terms": len(brand_manager.brand_terms) + len(brand_manager.ux_writing_rules) + len(brand_manager.style_rules)
    }

@app.get("/stats")
async def get_stats():
    """Get comprehensive system statistics"""
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
        "hugosave_info": {
            "brand_terms": len(brand_manager.brand_terms),
            "ux_rules": len(brand_manager.ux_writing_rules),
            "style_rules": len(brand_manager.style_rules),
            "model_optimized": ai_manager.model_info.get('hugosave_optimized', False)
        },
        "model_info": ai_manager.model_info,
        "features": {
            "enchant_available": ENCHANT_AVAILABLE,
            "redis_available": REDIS_AVAILABLE,
            "semantic_model_available": SEMANTIC_MODEL_AVAILABLE,
            "phonetics_available": PHONETICS_AVAILABLE,
            "rapidfuzz_available": RAPIDFUZZ_AVAILABLE,
            "hugosave_quantized": True
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "uptime_seconds": (datetime.now() - start_time).total_seconds(),
        "hugosave_optimized": ai_manager.model_info.get('hugosave_optimized', False),
        "model_type": ai_manager.model_info.get('model_type', 'unknown')
    }

@app.get("/benchmark")
async def benchmark_performance():
    """Benchmark Hugosave spell checker performance"""
    try:
        hugosave_test_words = ["hugosave", "hugohero", "login", "color", "realize", "centre"]
        
        start_time_bench = time.time()
        results = []
        
        for word in hugosave_test_words:
            word_start = time.time()
            suggestions = suggestion_engine.get_comprehensive_suggestions(word)
            word_time = (time.time() - word_start) * 1000
            
            results.append({
                "word": word,
                "processing_time_ms": round(word_time, 2),
                "suggestions_count": len(suggestions),
                "hugosave_correction": any(s.get('hugosave_brand', False) for s in suggestions)
            })
        
        total_time = (time.time() - start_time_bench) * 1000
        avg_time = total_time / len(hugosave_test_words)
        
        return {
            "benchmark_results": results,
            "total_time_ms": round(total_time, 2),
            "average_time_ms": round(avg_time, 2),
            "model_info": ai_manager.model_info,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Benchmark failed: {str(e)}")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "Hugosave Advanced Spell Checker API",
        "version": "3.0.0",
        "status": "running",
        "hugosave_optimized": ai_manager.model_info.get('hugosave_optimized', False),
        "features": [
            "Hugosave brand guidelines integration",
            "Quantized AI models for fast inference",
            "Multi-source suggestions",
            "Custom dictionaries",
            "Batch processing",
            "Intelligent caching",
            "Rate limiting",
            "Performance monitoring",
            "British English style guide",
            "UX writing rules"
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
