#!/usr/bin/env python3
"""
Setup and Run Script for Advanced AI Spell Checking System

This script handles:
1. Installing dependencies
2. Setting up NLTK data
3. Initializing the system
4. Starting the API server
5. Running comprehensive tests
"""

import os
import sys
import subprocess
import logging
import time
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_command(command, description):
    """Run a command and handle errors"""
    logger.info(f"Running: {description}")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        logger.info(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ {description} failed: {e.stderr}")
        return False

def install_dependencies():
    """Install required dependencies"""
    logger.info("🚀 Installing dependencies...")
    
    # Install main requirements
    if not run_command("pip install -r requirements.txt", "Installing main requirements"):
        logger.warning("Some packages might have failed to install, continuing...")
    
    # Install additional spell checking libraries
    additional_packages = [
        "pyenchant",  # For enchant support
        "language_tool_python",
        "symspellpy",
        "spellchecker",
        "nltk",
        "chromadb",
        "sentence-transformers"
    ]
    
    for package in additional_packages:
        run_command(f"pip install {package}", f"Installing {package}")
    
    logger.info("✅ Dependencies installation completed")

def setup_nltk_data():
    """Download required NLTK data"""
    logger.info("📚 Setting up NLTK data...")
    
    try:
        import nltk
        
        # Download required NLTK data
        nltk_data = [
            'punkt',
            'stopwords',
            'wordnet',
            'averaged_perceptron_tagger',
            'brown'
        ]
        
        for data in nltk_data:
            try:
                nltk.download(data, quiet=True)
                logger.info(f"Downloaded NLTK {data}")
            except Exception as e:
                logger.warning(f"Failed to download NLTK {data}: {e}")
        
        logger.info("✅ NLTK data setup completed")
        
    except ImportError:
        logger.warning("NLTK not available, skipping NLTK data setup")

def create_symspell_dictionary():
    """Create a basic SymSpell dictionary if not present"""
    logger.info("📖 Creating SymSpell dictionary...")
    
    dict_path = "./symspell_frequency_dictionary_en_82_765.txt"
    
    if not os.path.exists(dict_path):
        # Create a comprehensive dictionary with common English words
        common_words = {
            'the': 22038615, 'be': 12545825, 'and': 10741073, 'of': 10343885, 'a': 10144200,
            'to': 8965071, 'in': 6592456, 'he': 4081574, 'have': 4034417, 'it': 3872788,
            'that': 3430996, 'for': 3281454, 'they': 3181543, 'I': 3086225, 'with': 2683014,
            'as': 2670186, 'not': 2529480, 'on': 2481623, 'she': 2192375, 'at': 2113096,
            'by': 1945934, 'this': 1942065, 'we': 1861704, 'you': 1831504, 'do': 1798089,
            'but': 1776767, 'from': 1763142, 'or': 1663815, 'which': 1507145, 'one': 1465516,
            'would': 1386063, 'all': 1356013, 'will': 1320118, 'there': 1295491, 'say': 1266638,
            'who': 1244281, 'make': 1142932, 'when': 1020214, 'can': 1019255, 'more': 1000896,
            'if': 987620, 'no': 977636, 'man': 935607, 'out': 929542, 'other': 915746,
            'so': 915201, 'what': 894029, 'time': 888589, 'up': 883979, 'go': 850192,
            'about': 803791, 'than': 748027, 'into': 720177, 'could': 676787, 'state': 676787,
            'only': 660172, 'new': 652787, 'year': 642263, 'some': 640234, 'take': 611846,
            'come': 611846, 'these': 596637, 'know': 590294, 'see': 582007, 'use': 580509,
            'get': 574303, 'may': 564687, 'way': 551504, 'day': 541200, 'work': 530596,
            'life': 527506, 'system': 522893, 'each': 507992, 'right': 502011, 'program': 499852,
            'hear': 498779, 'question': 481043, 'during': 477822, 'where': 473642, 'much': 463938,
            'place': 459747, 'important': 456542, 'public': 452986, 'become': 451832, 'same': 449617,
            'few': 442185, 'house': 441850, 'world': 440006, 'still': 439997, 'should': 437874,
            'school': 436365, 'people': 434749, 'never': 434041, 'made': 433337, 'through': 432929,
            
            # Common misspellings and their corrections
            'recieve': 1000, 'receive': 500000,
            'seperate': 1000, 'separate': 300000,
            'definately': 1000, 'definitely': 200000,
            'occured': 1000, 'occurred': 150000,
            'accomodate': 1000, 'accommodate': 100000,
            'acknowlege': 1000, 'acknowledge': 80000,
            'begining': 1000, 'beginning': 250000,
            'calender': 1000, 'calendar': 120000,
            'cemetary': 1000, 'cemetery': 50000,
            'changable': 1000, 'changeable': 40000,
            'collectible': 45000, 'collectable': 35000,
            'concensus': 1000, 'consensus': 60000,
            'dilemna': 1000, 'dilemma': 70000,
            'existance': 1000, 'existence': 180000,
            'expierence': 1000, 'experience': 400000,
            'independant': 1000, 'independent': 200000,
            'occassion': 1000, 'occasion': 100000,
            'priviledge': 1000, 'privilege': 80000,
            'thier': 1000, 'their': 3000000,
            'tommorrow': 1000, 'tomorrow': 150000,
            'untill': 1000, 'until': 500000,
            'wierd': 1000, 'weird': 80000
        }
        
        with open(dict_path, 'w', encoding='utf-8') as f:
            for word, freq in common_words.items():
                f.write(f"{word} {freq}\n")
        
        logger.info(f"✅ Created SymSpell dictionary with {len(common_words)} entries")
    else:
        logger.info("SymSpell dictionary already exists")

def test_system():
    """Run comprehensive system tests"""
    logger.info("🧪 Running system tests...")
    
    try:
        from advanced_spell_checker import MultiEngineSpellChecker
        
        # Initialize checker
        checker = MultiEngineSpellChecker()
        
        # Test basic functionality
        test_words = ["appliction", "recieve", "seperate", "hello", "world"]
        logger.info("Testing word-level spell checking...")
        
        for word in test_words:
            result = checker.check_word(word)
            status = "✅ CORRECT" if result.is_correct else f"❌ INCORRECT (suggestions: {len(result.suggestions)})"
            logger.info(f"  '{word}': {status}")
        
        # Test sentence correction
        logger.info("Testing sentence-level correction...")
        test_sentence = "This is a sampel text with some erors and worng speling."
        sentence_result = checker.check_sentence(test_sentence)
        
        logger.info(f"  Original: {sentence_result['original_sentence']}")
        logger.info(f"  Corrected: {sentence_result['corrected_sentence']}")
        logger.info(f"  Corrections made: {sentence_result['total_corrections']}")
        
        # Test learning functionality
        logger.info("Testing learning functionality...")
        checker.learn_correction("teh", "the", "common typo")
        checker.add_to_whitelist("API")
        checker.ignore_word("JavaScript")
        
        logger.info("✅ System tests completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ System tests failed: {e}")
        return False

def start_api_server():
    """Start the API server"""
    logger.info("🚀 Starting Advanced Spell Checking API server...")
    
    try:
        from advanced_spell_api import app
        import uvicorn
        
        logger.info("Server starting on http://0.0.0.0:8000")
        logger.info("API documentation available at: http://localhost:8000/docs")
        logger.info("Alternative docs at: http://localhost:8000/redoc")
        
        uvicorn.run(
            "advanced_spell_api:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level="info"
        )
        
    except Exception as e:
        logger.error(f"❌ Failed to start API server: {e}")
        return False

def check_system_requirements():
    """Check if system meets requirements"""
    logger.info("🔍 Checking system requirements...")
    
    # Check Python version
    python_version = sys.version_info
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
        logger.error(f"❌ Python 3.8+ required, found {python_version.major}.{python_version.minor}")
        return False
    
    logger.info(f"✅ Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    # Check available memory (if psutil is available)
    try:
        import psutil
        memory = psutil.virtual_memory()
        logger.info(f"✅ Available memory: {memory.available / (1024**3):.1f} GB")
        
        if memory.available < 2 * (1024**3):  # Less than 2GB
            logger.warning("⚠️  Low memory detected. System may run slowly.")
    except ImportError:
        logger.info("psutil not available, skipping memory check")
    
    return True

def main():
    """Main setup and run function"""
    print("=" * 80)
    print("🤖 ADVANCED AI SPELL CHECKING SYSTEM")
    print("=" * 80)
    print()
    
    # Check system requirements
    if not check_system_requirements():
        sys.exit(1)
    
    # Ask user what they want to do
    print("What would you like to do?")
    print("1. Full setup (install dependencies, setup, and run)")
    print("2. Quick setup (setup only)")
    print("3. Run tests only")
    print("4. Start API server only")
    print("5. Install dependencies only")
    
    choice = input("\nEnter your choice (1-5): ").strip()
    
    if choice == "1":
        # Full setup
        logger.info("🚀 Starting full setup...")
        install_dependencies()
        setup_nltk_data()
        create_symspell_dictionary()
        
        if test_system():
            logger.info("🎉 Setup completed successfully!")
            print("\n" + "=" * 60)
            print("✅ SETUP COMPLETE!")
            print("=" * 60)
            print("🌐 Starting API server...")
            start_api_server()
        else:
            logger.error("❌ Setup failed during testing")
            
    elif choice == "2":
        # Quick setup
        setup_nltk_data()
        create_symspell_dictionary()
        logger.info("✅ Quick setup completed")
        
    elif choice == "3":
        # Run tests only
        test_system()
        
    elif choice == "4":
        # Start API server only
        start_api_server()
        
    elif choice == "5":
        # Install dependencies only
        install_dependencies()
        
    else:
        logger.error("Invalid choice. Exiting.")
        sys.exit(1)

if __name__ == "__main__":
    main()