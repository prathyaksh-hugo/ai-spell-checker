"""
PDF Data Extractor for T5 Fine-tuning
Extracts text from PDF documents (like brand guides) and converts to training format
"""

import os
import json
import re
import logging
from typing import List, Dict, Optional, Union, Tuple
from pathlib import Path
import pandas as pd

# PDF processing
try:
    import PyPDF2
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False
    logging.warning("PDF libraries not available. Install with: pip install PyPDF2 pdfplumber")

# Text processing
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

# OCR support
try:
    import pytesseract
    from PIL import Image
    import fitz  # PyMuPDF
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    logging.warning("OCR libraries not available. Install with: pip install pytesseract Pillow PyMuPDF")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BrandGuideProcessor:
    """Process brand guide content into spell correction training data"""
    
    def __init__(self):
        self.terminology_patterns = [
            r'[A-Z][a-z]+(?:[A-Z][a-z]*)*',  # CamelCase
            r'[A-Z]{2,}',  # Acronyms
            r'[A-Z][a-z]+\s+[A-Z][a-z]+',  # Proper nouns
        ]
        
    def extract_terminology(self, text: str) -> List[str]:
        """Extract important terminology from brand guide text"""
        terms = set()
        
        # Extract defined terms (often in quotes or after colons)
        defined_terms = re.findall(r'["'"]([^"'"]+)["'"]', text)
        terms.update(defined_terms)
        
        # Extract capitalized terms
        for pattern in self.terminology_patterns:
            matches = re.findall(pattern, text)
            terms.update(matches)
        
        # Extract terms from lists (bullet points, numbered lists)
        list_items = re.findall(r'[•\-\*]\s*([^\n]+)', text)
        for item in list_items:
            # Clean and extract meaningful terms
            clean_item = re.sub(r'[^\w\s]', '', item).strip()
            if len(clean_item.split()) <= 3 and len(clean_item) > 2:
                terms.add(clean_item)
        
        return list(terms)
    
    def extract_writing_rules(self, text: str) -> List[Dict[str, str]]:
        """Extract writing rules and examples from brand guide"""
        rules = []
        
        # Find Do/Don't examples
        do_dont_pattern = r'Do:\s*"([^"]+)".*?Don\'t:\s*"([^"]+)"'
        matches = re.findall(do_dont_pattern, text, re.DOTALL | re.IGNORECASE)
        
        for correct, incorrect in matches:
            rules.append({
                "correct": correct.strip(),
                "incorrect": incorrect.strip(),
                "rule_type": "do_dont"
            })
        
        # Find preference patterns (Prefer: ... Avoid: ...)
        prefer_pattern = r'Prefer:\s*"([^"]+)".*?Avoid:\s*"([^"]+)"'
        matches = re.findall(prefer_pattern, text, re.DOTALL | re.IGNORECASE)
        
        for correct, incorrect in matches:
            rules.append({
                "correct": correct.strip(),
                "incorrect": incorrect.strip(),
                "rule_type": "preference"
            })
        
        return rules
    
    def create_spelling_variations(self, term: str) -> List[str]:
        """Create common spelling variations for terminology"""
        variations = []
        
        # Common misspellings
        common_patterns = [
            (r'ie', 'ei'),  # believe -> beleive
            (r'ei', 'ie'),  # receive -> recieve
            (r'tion', 'sion'),  # action -> acsion
            (r'ance', 'ence'),  # guidance -> guidence
            (r'double consonant', 'single'),  # occurrence -> occurence
        ]
        
        # British vs American spelling
        brit_us_patterns = [
            (r'ise$', 'ize'),  # realise -> realize
            (r'isation$', 'ization'),  # organisation -> organization
            (r'our$', 'or'),  # colour -> color
            (r're$', 'er'),  # centre -> center
        ]
        
        # Apply patterns
        for old, new in brit_us_patterns:
            if re.search(old, term):
                variation = re.sub(old, new, term)
                variations.append(variation)
        
        # Character-level errors
        if len(term) > 3:
            # Missing character
            for i in range(1, len(term)):
                variation = term[:i] + term[i+1:]
                variations.append(variation)
            
            # Extra character
            common_extra = ['e', 'a', 'i', 'o', 'u']
            for i in range(len(term)):
                for char in common_extra:
                    variation = term[:i] + char + term[i:]
                    variations.append(variation)
        
        return list(set(variations))

class PDFDataExtractor:
    """Extract text from PDF documents"""
    
    def __init__(self, use_ocr: bool = False):
        self.use_ocr = use_ocr and OCR_AVAILABLE
        self.brand_processor = BrandGuideProcessor()
    
    def extract_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF using multiple methods"""
        text = ""
        
        try:
            # Method 1: pdfplumber (best for text extraction)
            if PDFPLUMBER_AVAILABLE:
                text = self._extract_with_pdfplumber(pdf_path)
                if text.strip():
                    logger.info("Successfully extracted text using pdfplumber")
                    return text
            
            # Method 2: PyPDF2 (fallback)
            text = self._extract_with_pypdf2(pdf_path)
            if text.strip():
                logger.info("Successfully extracted text using PyPDF2")
                return text
            
            # Method 3: OCR (for scanned PDFs)
            if self.use_ocr:
                text = self._extract_with_ocr(pdf_path)
                logger.info("Successfully extracted text using OCR")
                return text
                
        except Exception as e:
            logger.error(f"Failed to extract text from PDF: {e}")
        
        return text
    
    def _extract_with_pdfplumber(self, pdf_path: str) -> str:
        """Extract text using pdfplumber"""
        text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text
    
    def _extract_with_pypdf2(self, pdf_path: str) -> str:
        """Extract text using PyPDF2"""
        text = ""
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text() + "\n"
        return text
    
    def _extract_with_ocr(self, pdf_path: str) -> str:
        """Extract text using OCR"""
        if not OCR_AVAILABLE:
            return ""
        
        text = ""
        doc = fitz.open(pdf_path)
        
        for page_num in range(doc.page_count):
            page = doc[page_num]
            pix = page.get_pixmap()
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            page_text = pytesseract.image_to_string(img)
            text += page_text + "\n"
        
        doc.close()
        return text
    
    def process_brand_guide_text(self, text: str) -> List[Dict]:
        """Process brand guide text into training examples"""
        training_data = []
        
        # Extract terminology
        terms = self.brand_processor.extract_terminology(text)
        logger.info(f"Extracted {len(terms)} terminology terms")
        
        # Extract writing rules
        rules = self.brand_processor.extract_writing_rules(text)
        logger.info(f"Extracted {len(rules)} writing rules")
        
        # Create training examples from terminology
        for term in terms:
            variations = self.brand_processor.create_spelling_variations(term)
            
            for variation in variations[:3]:  # Limit variations per term
                if variation != term and len(variation) > 2:
                    training_data.append({
                        "input_text": f"correct: {variation}",
                        "target_text": term,
                        "source": f"terminology_{term}",
                        "error_type": "spelling_variation"
                    })
        
        # Create training examples from rules
        for rule in rules:
            training_data.append({
                "input_text": f"correct: {rule['incorrect']}",
                "target_text": rule['correct'],
                "source": f"rule_{rule['rule_type']}",
                "error_type": "style_guide"
            })
        
        # Extract sentences and create context-aware examples
        sentences = sent_tokenize(text)
        for i, sentence in enumerate(sentences[:50]):  # Limit for processing
            # Clean sentence
            clean_sentence = re.sub(r'[^\w\s\-\'\.]', '', sentence).strip()
            
            if len(clean_sentence.split()) >= 3 and len(clean_sentence) < 200:
                # Create a version with potential errors
                error_sentence = self._introduce_common_errors(clean_sentence)
                
                if error_sentence != clean_sentence:
                    training_data.append({
                        "input_text": f"correct: {error_sentence}",
                        "target_text": clean_sentence,
                        "source": f"sentence_{i}",
                        "error_type": "contextual"
                    })
        
        return training_data
    
    def _introduce_common_errors(self, text: str) -> str:
        """Introduce common spelling/grammar errors"""
        words = text.split()
        error_words = []
        
        for word in words:
            # 20% chance to introduce an error
            if len(word) > 3 and len(word) < 15 and word.isalpha():
                import random
                if random.random() < 0.2:
                    # Common error patterns
                    error_patterns = [
                        lambda w: w.replace('ie', 'ei') if 'ie' in w else w,
                        lambda w: w.replace('ei', 'ie') if 'ei' in w else w,
                        lambda w: w[:-1] if w.endswith('e') else w + 'e',
                        lambda w: w.replace('tion', 'sion') if 'tion' in w else w,
                        lambda w: w.replace('ance', 'ence') if 'ance' in w else w,
                    ]
                    
                    error_fn = random.choice(error_patterns)
                    error_word = error_fn(word)
                    error_words.append(error_word)
                else:
                    error_words.append(word)
            else:
                error_words.append(word)
        
        return ' '.join(error_words)

class MultiSourceDataManager:
    """Manage training data from multiple sources"""
    
    def __init__(self, data_dir: str = "./training_data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        self.pdf_extractor = PDFDataExtractor()
        self.data_sources = {}
        
    def add_pdf_source(self, pdf_path: str, source_name: str) -> List[Dict]:
        """Add training data from PDF source"""
        logger.info(f"Processing PDF source: {source_name}")
        
        # Extract text from PDF
        text = self.pdf_extractor.extract_from_pdf(pdf_path)
        
        if not text.strip():
            logger.error(f"No text extracted from {pdf_path}")
            return []
        
        # Process into training data
        training_data = self.pdf_extractor.process_brand_guide_text(text)
        
        # Save source data
        source_file = self.data_dir / f"{source_name}_data.json"
        with open(source_file, 'w', encoding='utf-8') as f:
            json.dump(training_data, f, indent=2, ensure_ascii=False)
        
        self.data_sources[source_name] = {
            "file": str(source_file),
            "count": len(training_data),
            "source_type": "pdf"
        }
        
        logger.info(f"Extracted {len(training_data)} training examples from {source_name}")
        return training_data
    
    def add_text_source(self, text: str, source_name: str) -> List[Dict]:
        """Add training data from raw text"""
        logger.info(f"Processing text source: {source_name}")
        
        # Process text directly
        training_data = self.pdf_extractor.process_brand_guide_text(text)
        
        # Save source data
        source_file = self.data_dir / f"{source_name}_data.json"
        with open(source_file, 'w', encoding='utf-8') as f:
            json.dump(training_data, f, indent=2, ensure_ascii=False)
        
        self.data_sources[source_name] = {
            "file": str(source_file),
            "count": len(training_data),
            "source_type": "text"
        }
        
        logger.info(f"Extracted {len(training_data)} training examples from {source_name}")
        return training_data
    
    def combine_all_sources(self, output_file: str = "combined_training_data.json") -> List[Dict]:
        """Combine training data from all sources"""
        combined_data = []
        
        for source_name, source_info in self.data_sources.items():
            with open(source_info["file"], 'r', encoding='utf-8') as f:
                source_data = json.load(f)
                
            # Add source metadata
            for example in source_data:
                example["data_source"] = source_name
                
            combined_data.extend(source_data)
        
        # Save combined data
        output_path = self.data_dir / output_file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(combined_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Combined {len(combined_data)} examples from {len(self.data_sources)} sources")
        logger.info(f"Saved to: {output_path}")
        
        # Create summary
        self._create_data_summary(combined_data, output_path.parent / "data_summary.json")
        
        return combined_data
    
    def _create_data_summary(self, data: List[Dict], summary_path: Path):
        """Create a summary of the training data"""
        summary = {
            "total_examples": len(data),
            "sources": {},
            "error_types": {},
            "example_distribution": {}
        }
        
        for example in data:
            source = example.get("data_source", "unknown")
            error_type = example.get("error_type", "unknown")
            
            # Count by source
            if source not in summary["sources"]:
                summary["sources"][source] = 0
            summary["sources"][source] += 1
            
            # Count by error type
            if error_type not in summary["error_types"]:
                summary["error_types"][error_type] = 0
            summary["error_types"][error_type] += 1
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Data summary saved to: {summary_path}")

def process_hugosave_brand_guide():
    """Process the Hugosave brand guide data provided by user"""
    
    hugosave_text = """
    Hugosave — Brand Voice & UX Writing Guide
    Purpose: A concise, practical guide that captures Hugosave's brand voice, content rules, UX writing patterns, and approved terminology. Use this as the single source of truth for product copy, UI text, microcopy, notifications, and localisation handoffs.
    Overview
    Brand voice pillars: Friendly · Credible · Empowering

    Primary audience: Everyday people learning to save, spend and invest with confidence.

    Writing goals: Clear, usable, action-oriented, and inclusive. Copy should reduce friction, set expectations, and guide users to successful outcomes.

    Tone of Voice
    Friendly: Warm, conversational, approachable.

    Credible: Accurate, precise, no overpromising. Use data or clear instructions where needed.

    Empowering: Encourage user action; explain next steps and choices.

    How tone shifts by journey:
    Onboarding — friendly + educational (explain benefits simply).

    Transaction success — celebratory, concise, clear next steps.

    Errors — empathetic, actionable, solution-oriented.

    Security/privacy flows — formal and precise.

    Content Style Guide
    Language: British English (use 'u' in words like colour; use s rather than z).

    Case: Sentence case for headings and UI labels (e.g., "Add money").

    Oxford comma: Use a comma before 'and' in lists of three or more.

    Person: Prefer second person for instructions and CTAs ("Add money to your Save account"). Use first person only when emphasizing ownership ("My Save Account").

    Contractions: Use moderate contractions that feel natural in conversational UI copy (e.g., "you'll", "we're"), but avoid very informal ones ("gonna", "wanna").

    Tone level: Aim for an 8th-grade reading level—short sentences, common words.

    Grammar, Punctuation & Formatting Rules
    Sentence structure: Keep sentences short and direct. One idea per sentence.

    Numbers: Use numerals (1, 2, 3) for lists and amounts; spelled-out words only when necessary for tone.

    Dates: Monday, 24 March 2025 for long form; Mon, 24 Mar '25 allowed for compact displays. Avoid ambiguous formats.

    Time: 12-hour clock with am/pm lowercase and no space (e.g., 2:18pm).

    Currency: Prefix currency symbol, with a space: S$ 100.00 (follow example in UX pattern). Show decimals up to two places unless product requires more.

    Units: Use decimal point every three digits from right for grams/numeric formatting when relevant (match product examples).

    Terminology & Naming Conventions
    Defined Proper Nouns (capitalised):

    Hugosave, Wealthcare, Wealthcare®, Smarter Spending

    Hugohero, Hugoheroes, #hugohero, #hugoheroes

    Homescreen, Save Account, Spend Account, Cash Account, Multi-currency Account

    Debit Card, Hugosave Visa Platinum Debit Card

    Pots, Rewards Centre, Quests, Referrals, Roundups, Auto Top-up, Invest-as-you-spend

    Net Worth, Portfolio Composition, Investment Personality Quiz (Defender, Mediator, Adventurer)

    KYC, Singpass, eDDA, T&Cs

    HugoHub-specific terms: HugoHub, Customer, End-Customer, Wallet

    Principles:
    Treat defined terms as proper nouns (capitalise consistently).

    Use sentence case for UI copy, but capitalise defined nouns (e.g., "Your Save Account").

    Localisation & Translation Guidance
    Provide translators with short context strings (where text appears) and character limits.

    Avoid idioms, metaphors, and culturally specific references.

    Keep placeholders and tokens consistent (e.g., {amount}, {date}).

    Tips for translation: avoid compact wordplay that breaks in other languages; provide notes for ambiguous terms (e.g., "Pot" meaning a savings bucket).

    Coordination: share the Approved Terms List and a glossary to localisation teams before translation runs.

    Accessibility & Inclusivity
    Use plain language and short sentences.

    Provide clear labels for input fields and buttons (avoid vague labels like "Submit").

    Screen reader tips: include context where needed — e.g., "Add money — opens deposit modal" as aria-labels when appropriate.

    Use inclusive examples and avoid stereotypes and slang.

    Microcopy & UI Elements
    Headings: Short, descriptive. Prefer phrases like "Add money" not full sentences.

    Buttons / CTAs: Action-oriented, no more than 20 characters where possible. Primary CTAs are concise and start with verb (e.g., "Top up", "Withdraw").

    Input labels & placeholders: Labels must be persistent. Placeholders show example input, not instructions.

    Tooltips & help: Keep single-line, factual and non-judgemental.

    Empty states / Onboarding carousels: Explain benefit + clear next step.

    Error messages & Alerts
    Style: Empathetic, solution oriented, short. Explain what happened and how to fix it.

    User-caused error example: "We couldn't verify that account number. Check the number and try again."

    System failure example: "Something went wrong on our end — try again in a few minutes."

    Transaction failure: Provide next steps and contact channels. Avoid technical jargon.

    Notifications & Emails
    Keep subject lines concise and benefit-oriented.

    Use present tense and active voice (e.g., "You've received S$ 10 cash back").

    Provide clear calls to action and explain why the message matters.

    Character Limits (UI guidelines)
    CTA / Button: 20 characters (aim lower when possible)

    Group buttons on card (Payees/Buttons): 9–10 characters for group buttons

    Microcopy / Tooltips: 100 characters max

    Headings: 30 characters recommended

    Product-Specific Vocabulary & Examples
    Approved short phrases: "Add money", "Top up", "Withdraw", "View PIN", "Manage Card", "Schedule".

    Prefer: "Add money to your Save account" (Do)

    Avoid: "Add money to my Save account" (Don't)

    Do / Don't Examples (Quick Reference)
    Do: "Add money to your Save account"

    Don't: "Add money to my Save account"

    Do: "Save changes?" (Confirmational short prompt)

    Don't: "Would you like to save your changes?" (Too wordy for UI)

    Do: "Message has been sent"

    Don't: "Message sent" (ambiguous tense) — prefer the Do example for clarity.

    Do: "Sign in"

    Don't: "Log in" / "Sign-in" / "Login" (use "Sign in" as primary)

    Writing for Different User Journeys
    Onboarding: Focus on benefit and next step. Use short, encouraging lines.

    Successful actions: Celebrate concisely and show next actions (e.g., "Done — money added to Save account. View balance").

    Failures (app side): Apologise, explain, and describe next steps.

    Failures (user side): Be specific and instructive ("You entered an incorrect account number").

    Short copy: headings, command labels, links
    Use verbs for commands ("View PIN", "Manage Card").

    Keep link text meaningful out of context (avoid "click here").

    Guidance for Translators & Localisation teams
    Share glossary with defined nouns and examples of Do/Don't translations.

    Provide the UI context and character limits. For languages with longer strings, give expanded space where possible.

    Keep placeholders intact (e.g., {amount}) and provide notes for inflection or grammar requirements.

    Screen Reader & Accessibility Writing Tips
    Use explicit labels and avoid ambiguous link names.

    Keep context in short strings for dynamic updates (ARIA-live usage).

    For icons-only buttons, include an aria-label that matches the visible CTA in context.

    Appendix — Approved Terms & Short Glossary
    Hugosave, Wealthcare, Smarter Spending, Hugohero, Hugoheroes, Save Account, Spend Account, Cash Account, Pots, Rewards Centre, Quests, Referrals, Roundups, Invest-as-you-spend, Debit Card, Net Worth, Portfolio Composition, Investment Personality Quiz, Defender, Mediator, Adventurer, KYC, Singpass, eDDA, T&Cs.
    """
    
    # Initialize data manager
    data_manager = MultiSourceDataManager()
    
    # Process the Hugosave brand guide
    training_data = data_manager.add_text_source(hugosave_text, "hugosave_brand_guide")
    
    # Combine all sources (in this case, just one)
    combined_data = data_manager.combine_all_sources()
    
    return combined_data, data_manager

def main():
    """Main function to demonstrate PDF and text processing"""
    logger.info("Starting PDF/Text Data Extraction Pipeline")
    
    # Process the Hugosave brand guide
    training_data, data_manager = process_hugosave_brand_guide()
    
    logger.info(f"Generated {len(training_data)} training examples")
    
    # Copy to main training file for the pipeline
    import shutil
    shutil.copy2("./training_data/combined_training_data.json", "./training_data.json")
    
    logger.info("Training data ready for T5 fine-tuning!")
    
    # Show some examples
    logger.info("\nSample Training Examples:")
    for i, example in enumerate(training_data[:5]):
        logger.info(f"Example {i+1}:")
        logger.info(f"  Input: {example['input_text']}")
        logger.info(f"  Target: {example['target_text']}")
        logger.info(f"  Type: {example['error_type']}")
        logger.info("")

if __name__ == "__main__":
    main()