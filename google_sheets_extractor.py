"""
Google Sheets Data Extractor for T5 Fine-tuning
Extracts brand and UX guidelines from Google Sheets to prepare training data
"""

import gspread
import pandas as pd
import json
import os
from google.oauth2.service_account import Credentials
from typing import List, Dict, Tuple
import re

class GoogleSheetsExtractor:
    def __init__(self, credentials_path: str = None):
        """
        Initialize Google Sheets extractor
        
        Args:
            credentials_path: Path to Google service account credentials JSON file
        """
        self.credentials_path = credentials_path
        self.gc = None
        self.setup_authentication()
    
    def setup_authentication(self):
        """Setup Google Sheets authentication"""
        try:
            if self.credentials_path and os.path.exists(self.credentials_path):
                # Using service account credentials
                scope = [
                    'https://spreadsheets.google.com/feeds',
                    'https://www.googleapis.com/auth/drive'
                ]
                creds = Credentials.from_service_account_file(
                    self.credentials_path, scopes=scope
                )
                self.gc = gspread.authorize(creds)
            else:
                # Using OAuth2 (requires user authentication)
                self.gc = gspread.oauth()
                
        except Exception as e:
            print(f"Authentication failed: {e}")
            print("Please ensure you have proper Google Sheets credentials set up")
            raise
    
    def extract_data(self, sheet_url: str, worksheet_name: str = None) -> pd.DataFrame:
        """
        Extract data from Google Sheets
        
        Args:
            sheet_url: URL of the Google Sheet
            worksheet_name: Name of specific worksheet (optional)
            
        Returns:
            DataFrame containing the extracted data
        """
        try:
            # Open the spreadsheet
            sheet = self.gc.open_by_url(sheet_url)
            
            # Get specific worksheet or first one
            if worksheet_name:
                worksheet = sheet.worksheet(worksheet_name)
            else:
                worksheet = sheet.get_worksheet(0)
            
            # Get all records as list of dictionaries
            records = worksheet.get_all_records()
            
            # Convert to DataFrame
            df = pd.DataFrame(records)
            
            print(f"Successfully extracted {len(df)} rows from the sheet")
            return df
            
        except Exception as e:
            print(f"Error extracting data from Google Sheets: {e}")
            raise
    
    def prepare_spell_correction_data(self, df: pd.DataFrame, 
                                    text_column: str, 
                                    correct_column: str = None) -> List[Dict]:
        """
        Prepare data for T5 spell correction fine-tuning
        
        Args:
            df: DataFrame with brand/UX guidelines
            text_column: Column containing text with potential spelling errors
            correct_column: Column with corrected text (optional)
            
        Returns:
            List of training examples in T5 format
        """
        training_data = []
        
        for idx, row in df.iterrows():
            text = str(row[text_column]).strip()
            
            if correct_column and correct_column in df.columns:
                # Use provided corrections
                corrected_text = str(row[correct_column]).strip()
            else:
                # Use the original text as target (assuming it's already correct)
                corrected_text = text
            
            # Create T5 training example
            # Input format: "correct: [text with potential errors]"
            # Target format: "[corrected text]"
            example = {
                "input_text": f"correct: {text}",
                "target_text": corrected_text,
                "source": f"row_{idx}"
            }
            
            training_data.append(example)
        
        return training_data
    
    def create_synthetic_errors(self, text: str, error_rate: float = 0.1) -> str:
        """
        Create synthetic spelling errors for training data augmentation
        
        Args:
            text: Original correct text
            error_rate: Proportion of words to introduce errors in
            
        Returns:
            Text with synthetic spelling errors
        """
        words = text.split()
        error_patterns = [
            lambda w: w[:-1] if len(w) > 3 else w,  # Remove last character
            lambda w: w[0] + w[2:] if len(w) > 2 else w,  # Remove second character
            lambda w: w + w[-1] if len(w) > 1 else w,  # Duplicate last character
            lambda w: w[0] + w[1:].replace('e', 'a') if 'e' in w else w,  # Replace e with a
            lambda w: w.replace('i', 'y') if 'i' in w else w,  # Replace i with y
        ]
        
        import random
        
        for i, word in enumerate(words):
            if random.random() < error_rate and len(word) > 2:
                error_fn = random.choice(error_patterns)
                words[i] = error_fn(word)
        
        return ' '.join(words)
    
    def augment_training_data(self, training_data: List[Dict], 
                            augmentation_factor: int = 2) -> List[Dict]:
        """
        Augment training data by creating synthetic errors
        
        Args:
            training_data: Original training data
            augmentation_factor: How many synthetic examples to create per original
            
        Returns:
            Augmented training data
        """
        augmented_data = training_data.copy()
        
        for example in training_data:
            target_text = example["target_text"]
            
            for i in range(augmentation_factor):
                # Create text with synthetic errors
                error_text = self.create_synthetic_errors(target_text)
                
                augmented_example = {
                    "input_text": f"correct: {error_text}",
                    "target_text": target_text,
                    "source": f"{example['source']}_aug_{i}"
                }
                
                augmented_data.append(augmented_example)
        
        return augmented_data
    
    def save_training_data(self, training_data: List[Dict], output_path: str):
        """Save training data to JSON file"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(training_data, f, indent=2, ensure_ascii=False)
        
        print(f"Training data saved to {output_path}")
        print(f"Total examples: {len(training_data)}")

def main():
    """
    Example usage of GoogleSheetsExtractor
    """
    # Configuration
    SHEET_URL = "YOUR_GOOGLE_SHEET_URL_HERE"
    CREDENTIALS_PATH = "path/to/your/service-account-credentials.json"  # Optional
    WORKSHEET_NAME = None  # Or specify worksheet name
    TEXT_COLUMN = "text"  # Column containing your brand/UX guidelines text
    CORRECT_COLUMN = None  # Optional: column with corrected versions
    
    try:
        # Initialize extractor
        extractor = GoogleSheetsExtractor(credentials_path=CREDENTIALS_PATH)
        
        # Extract data from Google Sheets
        print("Extracting data from Google Sheets...")
        df = extractor.extract_data(SHEET_URL, WORKSHEET_NAME)
        
        print(f"Available columns: {list(df.columns)}")
        print(f"Sample data:\n{df.head()}")
        
        # Prepare training data
        print("\nPreparing training data for T5...")
        training_data = extractor.prepare_spell_correction_data(
            df, TEXT_COLUMN, CORRECT_COLUMN
        )
        
        # Augment data with synthetic errors
        print("Augmenting training data...")
        augmented_data = extractor.augment_training_data(training_data, augmentation_factor=3)
        
        # Save training data
        output_path = "training_data.json"
        extractor.save_training_data(augmented_data, output_path)
        
        print(f"\nTraining data preparation complete!")
        print(f"Original examples: {len(training_data)}")
        print(f"Augmented examples: {len(augmented_data)}")
        
    except Exception as e:
        print(f"Error in main execution: {e}")
        print("\nPlease ensure:")
        print("1. You have valid Google Sheets credentials")
        print("2. The sheet URL is correct and accessible")
        print("3. The specified columns exist in your sheet")

if __name__ == "__main__":
    main()