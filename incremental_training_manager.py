"""
Incremental Training Manager for T5 Spell Correction
Handles adding new data sources and efficient model updates
"""

import os
import json
import shutil
from datetime import datetime
from typing import List, Dict, Optional, Union
from pathlib import Path
import logging

# Core libraries
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer, TrainingArguments
from datasets import Dataset, concatenate_datasets

# Local imports
from pdf_data_extractor import MultiSourceDataManager, PDFDataExtractor
from t5_fine_tuner import T5SpellCorrectionTrainer, ModelConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataVersionManager:
    """Manage different versions of training data"""
    
    def __init__(self, base_dir: str = "./data_versions"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        
        self.current_version = self._get_latest_version()
        self.version_info_file = self.base_dir / "version_info.json"
        
        self._load_version_info()
    
    def _get_latest_version(self) -> str:
        """Get the latest data version"""
        existing_versions = [d.name for d in self.base_dir.iterdir() if d.is_dir() and d.name.startswith('v')]
        
        if not existing_versions:
            return "v1.0.0"
        
        # Sort versions
        versions = sorted(existing_versions, key=lambda x: [int(i) for i in x[1:].split('.')])
        return versions[-1]
    
    def _load_version_info(self):
        """Load version information"""
        if self.version_info_file.exists():
            with open(self.version_info_file, 'r') as f:
                self.version_info = json.load(f)
        else:
            self.version_info = {}
    
    def _save_version_info(self):
        """Save version information"""
        with open(self.version_info_file, 'w') as f:
            json.dump(self.version_info, f, indent=2)
    
    def create_new_version(self, training_data: List[Dict], 
                          sources_added: List[str], 
                          description: str = "") -> str:
        """Create a new data version"""
        # Increment version
        current_parts = [int(x) for x in self.current_version[1:].split('.')]
        current_parts[1] += 1  # Increment minor version
        new_version = f"v{'.'.join(map(str, current_parts))}"
        
        # Create version directory
        version_dir = self.base_dir / new_version
        version_dir.mkdir(exist_ok=True)
        
        # Save training data
        data_file = version_dir / "training_data.json"
        with open(data_file, 'w', encoding='utf-8') as f:
            json.dump(training_data, f, indent=2, ensure_ascii=False)
        
        # Save version metadata
        version_metadata = {
            "version": new_version,
            "created_at": datetime.now().isoformat(),
            "total_examples": len(training_data),
            "sources_added": sources_added,
            "description": description,
            "data_file": str(data_file)
        }
        
        metadata_file = version_dir / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(version_metadata, f, indent=2)
        
        # Update version info
        self.version_info[new_version] = version_metadata
        self._save_version_info()
        
        self.current_version = new_version
        logger.info(f"Created new data version: {new_version}")
        
        return new_version
    
    def get_version_data(self, version: str) -> List[Dict]:
        """Get training data for a specific version"""
        if version not in self.version_info:
            raise ValueError(f"Version {version} not found")
        
        data_file = self.version_info[version]["data_file"]
        with open(data_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def list_versions(self) -> List[Dict]:
        """List all available versions"""
        return list(self.version_info.values())

class IncrementalTrainingManager:
    """Manage incremental training with new data sources"""
    
    def __init__(self, base_model_path: str = "./t5_spell_finetuned"):
        self.base_model_path = base_model_path
        self.data_manager = MultiSourceDataManager()
        self.version_manager = DataVersionManager()
        
        # Training configuration
        self.config = ModelConfig(
            model_name=base_model_path if Path(base_model_path).exists() else "ai-forever/T5-large-spell",
            learning_rate=1e-5,  # Lower learning rate for incremental training
            batch_size=4,
            num_epochs=2,  # Fewer epochs for incremental updates
            output_dir=f"./incremental_models/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
    
    def add_new_data_source(self, source_path: str, source_name: str, 
                          source_type: str = "auto") -> str:
        """Add a new data source and create new version"""
        logger.info(f"Adding new data source: {source_name}")
        
        # Determine source type
        if source_type == "auto":
            if source_path.endswith('.pdf'):
                source_type = "pdf"
            elif source_path.endswith(('.txt', '.md')):
                source_type = "text"
            else:
                source_type = "text"  # Default
        
        # Process new data
        if source_type == "pdf":
            new_data = self.data_manager.add_pdf_source(source_path, source_name)
        elif source_type == "text":
            if os.path.exists(source_path):
                with open(source_path, 'r', encoding='utf-8') as f:
                    text_content = f.read()
            else:
                text_content = source_path  # Assume it's the actual text
            new_data = self.data_manager.add_text_source(text_content, source_name)
        else:
            raise ValueError(f"Unsupported source type: {source_type}")
        
        # Get existing data from latest version
        try:
            existing_data = self.version_manager.get_version_data(
                self.version_manager.current_version
            )
        except (ValueError, KeyError):
            existing_data = []
        
        # Combine with new data
        combined_data = existing_data + new_data
        
        # Create new version
        new_version = self.version_manager.create_new_version(
            combined_data,
            [source_name],
            f"Added {source_name} ({len(new_data)} examples)"
        )
        
        # Update main training file
        shutil.copy2(
            self.version_manager.base_dir / new_version / "training_data.json",
            "./training_data.json"
        )
        
        logger.info(f"New data version created: {new_version}")
        logger.info(f"Total examples: {len(combined_data)} (added {len(new_data)})")
        
        return new_version
    
    def incremental_train(self, new_data_ratio: float = 0.3) -> str:
        """Perform incremental training with new data"""
        logger.info("Starting incremental training...")
        
        # Load current training data
        with open("./training_data.json", 'r', encoding='utf-8') as f:
            all_data = json.load(f)
        
        # Separate new data (if we have source metadata)
        recent_sources = self._get_recent_sources()
        new_data = [ex for ex in all_data if ex.get("data_source") in recent_sources]
        old_data = [ex for ex in all_data if ex.get("data_source") not in recent_sources]
        
        # Create balanced training set
        if new_data and len(new_data) < len(all_data) * new_data_ratio:
            # If we have little new data, oversample it
            import random
            oversample_factor = max(1, int(len(old_data) * new_data_ratio / len(new_data)))
            training_data = old_data + new_data * oversample_factor
            random.shuffle(training_data)
        else:
            training_data = all_data
        
        # Update config for incremental training
        self.config.output_dir = f"./incremental_models/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Save balanced training data
        balanced_data_file = "balanced_training_data.json"
        with open(balanced_data_file, 'w', encoding='utf-8') as f:
            json.dump(training_data, f, indent=2, ensure_ascii=False)
        
        # Initialize trainer
        trainer = T5SpellCorrectionTrainer(self.config)
        
        # Train model
        test_results = trainer.train(balanced_data_file)
        
        # Update base model path for future incremental training
        self.base_model_path = self.config.output_dir
        
        logger.info(f"Incremental training completed: {self.config.output_dir}")
        
        return self.config.output_dir
    
    def _get_recent_sources(self, days_back: int = 7) -> List[str]:
        """Get data sources added in recent days"""
        cutoff_date = datetime.now().timestamp() - (days_back * 24 * 3600)
        recent_sources = []
        
        for version, info in self.version_manager.version_info.items():
            created_at = datetime.fromisoformat(info["created_at"]).timestamp()
            if created_at > cutoff_date:
                recent_sources.extend(info.get("sources_added", []))
        
        return recent_sources
    
    def validate_new_model(self, model_path: str, test_cases: List[str] = None) -> Dict:
        """Validate the newly trained model"""
        logger.info(f"Validating model: {model_path}")
        
        if test_cases is None:
            test_cases = [
                "Add money to your Save acount",  # Hugosave-specific
                "Recieve S$ 10 cash back",  # British spelling + currency
                "Your Net Wroth is availble",  # Terminology
                "Managee your Debit Card",  # Common typo
                "Sign-in to your accont"  # Style guide
            ]
        
        try:
            # Load model
            tokenizer = T5Tokenizer.from_pretrained(model_path)
            model = T5ForConditionalGeneration.from_pretrained(model_path)
            
            results = []
            for test_case in test_cases:
                # Tokenize input
                inputs = tokenizer(
                    f"correct: {test_case}",
                    max_length=512,
                    truncation=True,
                    padding=True,
                    return_tensors="pt"
                )
                
                # Generate correction
                with torch.no_grad():
                    outputs = model.generate(
                        inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        max_length=512,
                        num_beams=4,
                        early_stopping=True
                    )
                
                corrected = tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                results.append({
                    "input": test_case,
                    "output": corrected,
                    "improvement": test_case != corrected
                })
            
            validation_score = sum(1 for r in results if r["improvement"]) / len(results)
            
            validation_result = {
                "model_path": model_path,
                "validation_score": validation_score,
                "test_results": results,
                "timestamp": datetime.now().isoformat()
            }
            
            # Save validation results
            validation_file = Path(model_path) / "validation_results.json"
            with open(validation_file, 'w') as f:
                json.dump(validation_result, f, indent=2)
            
            logger.info(f"Validation completed. Score: {validation_score:.2%}")
            
            return validation_result
            
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return {"error": str(e)}
    
    def get_training_summary(self) -> Dict:
        """Get summary of all training iterations"""
        summary = {
            "data_versions": self.version_manager.list_versions(),
            "model_versions": [],
            "total_data_sources": len(self.data_manager.data_sources),
            "latest_version": self.version_manager.current_version
        }
        
        # Find model directories
        incremental_models_dir = Path("./incremental_models")
        if incremental_models_dir.exists():
            model_dirs = [d for d in incremental_models_dir.iterdir() if d.is_dir()]
            
            for model_dir in sorted(model_dirs):
                model_info = {
                    "path": str(model_dir),
                    "created": model_dir.stat().st_mtime,
                    "name": model_dir.name
                }
                
                # Check for validation results
                validation_file = model_dir / "validation_results.json"
                if validation_file.exists():
                    with open(validation_file, 'r') as f:
                        validation_data = json.load(f)
                        model_info["validation_score"] = validation_data.get("validation_score")
                
                summary["model_versions"].append(model_info)
        
        return summary

def main():
    """Example usage of incremental training manager"""
    logger.info("Incremental Training Manager Demo")
    
    # Initialize manager
    manager = IncrementalTrainingManager()
    
    # Process Hugosave brand guide as initial data
    from pdf_data_extractor import process_hugosave_brand_guide
    
    # Create initial version
    training_data, _ = process_hugosave_brand_guide()
    
    initial_version = manager.version_manager.create_new_version(
        training_data,
        ["hugosave_brand_guide"],
        "Initial Hugosave brand guide data"
    )
    
    # Copy to main training file
    shutil.copy2(
        manager.version_manager.base_dir / initial_version / "training_data.json",
        "./training_data.json"
    )
    
    logger.info(f"Initial data version created: {initial_version}")
    logger.info(f"Ready for incremental training!")
    
    # Example: Add new data source (would be a PDF file in practice)
    # new_version = manager.add_new_data_source("new_style_guide.pdf", "style_guide_v2")
    
    # Example: Perform incremental training
    # model_path = manager.incremental_train()
    
    # Example: Validate new model
    # validation_results = manager.validate_new_model(model_path)
    
    # Show training summary
    summary = manager.get_training_summary()
    logger.info(f"Training Summary: {json.dumps(summary, indent=2, default=str)}")

if __name__ == "__main__":
    main()