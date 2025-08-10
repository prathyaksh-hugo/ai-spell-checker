"""
T5 Large Spell Correction Fine-tuning Script
Fine-tunes ai-forever/T5-large-spell on custom brand/UX guidelines data
"""

import os
import json
import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datasets import Dataset, DatasetDict
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback
)
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Configuration for T5 fine-tuning"""
    model_name: str = "ai-forever/T5-large-spell"
    max_input_length: int = 512
    max_target_length: int = 512
    learning_rate: float = 5e-5
    batch_size: int = 8
    num_epochs: int = 3
    warmup_steps: int = 500
    save_steps: int = 1000
    eval_steps: int = 500
    logging_steps: int = 100
    gradient_accumulation_steps: int = 4
    fp16: bool = True
    output_dir: str = "./t5_spell_finetuned"
    wandb_project: str = "t5-spell-correction"

class T5DataProcessor:
    """Data processor for T5 spell correction fine-tuning"""
    
    def __init__(self, tokenizer: T5Tokenizer, config: ModelConfig):
        self.tokenizer = tokenizer
        self.config = config
    
    def load_training_data(self, data_path: str) -> List[Dict]:
        """Load training data from JSON file"""
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        logger.info(f"Loaded {len(data)} training examples")
        return data
    
    def create_datasets(self, training_data: List[Dict], 
                       test_size: float = 0.2, 
                       val_size: float = 0.1) -> DatasetDict:
        """
        Create train, validation, and test datasets
        
        Args:
            training_data: List of training examples
            test_size: Proportion of data for testing
            val_size: Proportion of remaining data for validation
            
        Returns:
            DatasetDict with train, validation, and test splits
        """
        # Split data
        train_data, test_data = train_test_split(
            training_data, test_size=test_size, random_state=42
        )
        
        train_data, val_data = train_test_split(
            train_data, test_size=val_size, random_state=42
        )
        
        logger.info(f"Dataset splits - Train: {len(train_data)}, "
                   f"Val: {len(val_data)}, Test: {len(test_data)}")
        
        # Create datasets
        datasets = DatasetDict({
            'train': Dataset.from_list(train_data),
            'validation': Dataset.from_list(val_data),
            'test': Dataset.from_list(test_data)
        })
        
        # Tokenize datasets
        tokenized_datasets = datasets.map(
            self._tokenize_function,
            batched=True,
            remove_columns=datasets['train'].column_names
        )
        
        return tokenized_datasets
    
    def _tokenize_function(self, examples):
        """Tokenize input and target texts"""
        # Tokenize inputs
        model_inputs = self.tokenizer(
            examples['input_text'],
            max_length=self.config.max_input_length,
            truncation=True,
            padding=False  # Will be handled by data collator
        )
        
        # Tokenize targets
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                examples['target_text'],
                max_length=self.config.max_target_length,
                truncation=True,
                padding=False
            )
        
        model_inputs['labels'] = labels['input_ids']
        return model_inputs

class T5SpellCorrectionTrainer:
    """Main trainer class for T5 spell correction fine-tuning"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.trainer = None
        self.data_processor = None
        
        # Initialize WandB if available
        self._wandb = None
        self.use_wandb = self._setup_wandb()
    
    def _setup_wandb(self) -> bool:
        """Setup Weights & Biases logging (optional)"""
        try:
            import wandb  # Local import to keep dependency optional
            wandb.init(
                project=self.config.wandb_project,
                config=self.config.__dict__
            )
            self._wandb = wandb
            return True
        except Exception as e:
            logger.warning(f"WandB disabled: {e}")
            self._wandb = None
            return False
    
    def load_model_and_tokenizer(self):
        """Load T5 model and tokenizer"""
        logger.info(f"Loading model: {self.config.model_name}")
        
        self.tokenizer = T5Tokenizer.from_pretrained(self.config.model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(self.config.model_name)
        
        # Initialize data processor
        self.data_processor = T5DataProcessor(self.tokenizer, self.config)
        
        logger.info(f"Model loaded successfully. Parameters: {self.model.num_parameters():,}")
    
    def prepare_training_arguments(self) -> TrainingArguments:
        """Prepare training arguments"""
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            warmup_steps=self.config.warmup_steps,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            eval_steps=self.config.eval_steps,
            evaluation_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            fp16=self.config.fp16,
            dataloader_pin_memory=True,
            remove_unused_columns=False,
            report_to="wandb" if self.use_wandb else "none",
            save_total_limit=2,
            prediction_loss_only=False,
        )
        
        return training_args
    
    def train(self, training_data_path: str):
        """
        Main training function
        
        Args:
            training_data_path: Path to training data JSON file
        """
        # Load model and tokenizer
        self.load_model_and_tokenizer()
        
        # Load and prepare data
        logger.info("Preparing training data...")
        training_data = self.data_processor.load_training_data(training_data_path)
        tokenized_datasets = self.data_processor.create_datasets(training_data)
        
        # Prepare training arguments
        training_args = self.prepare_training_arguments()
        
        # Create data collator
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.model,
            padding=True
        )
        
        # Initialize trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_datasets['train'],
            eval_dataset=tokenized_datasets['validation'],
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )
        
        # Start training
        logger.info("Starting training...")
        self.trainer.train()
        
        # Save the final model
        self.trainer.save_model()
        self.tokenizer.save_pretrained(self.config.output_dir)
        
        # Evaluate on test set
        logger.info("Evaluating on test set...")
        test_results = self.trainer.evaluate(tokenized_datasets['test'])
        
        logger.info(f"Test results: {test_results}")
        
        if self.use_wandb and self._wandb is not None:
            self._wandb.log({"test_loss": test_results.get("eval_loss")})
            self._wandb.finish()
        
        return test_results
    
    def predict(self, input_texts: List[str], batch_size: int = 8) -> List[str]:
        """
        Generate predictions for input texts
        
        Args:
            input_texts: List of texts to correct
            batch_size: Batch size for inference
            
        Returns:
            List of corrected texts
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not loaded. Call load_model_and_tokenizer() first.")
        
        predictions = []
        
        # Process in batches
        for i in tqdm(range(0, len(input_texts), batch_size), desc="Generating predictions"):
            batch_texts = input_texts[i:i + batch_size]
            
            # Tokenize batch
            inputs = self.tokenizer(
                batch_texts,
                max_length=self.config.max_input_length,
                truncation=True,
                padding=True,
                return_tensors="pt"
            )
            
            # Generate predictions
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_length=self.config.max_target_length,
                    num_beams=4,
                    early_stopping=True
                )
            
            # Decode predictions
            batch_predictions = self.tokenizer.batch_decode(
                outputs, skip_special_tokens=True
            )
            
            predictions.extend(batch_predictions)
        
        return predictions

class ModelEvaluator:
    """Evaluation utilities for the fine-tuned model"""
    
    def __init__(self, trainer: T5SpellCorrectionTrainer):
        self.trainer = trainer
    
    def calculate_bleu_score(self, predictions: List[str], references: List[str]) -> float:
        """Calculate BLEU score for predictions"""
        try:
            from sacrebleu import corpus_bleu
            bleu = corpus_bleu(predictions, [references])
            return bleu.score
        except ImportError:
            logger.warning("sacrebleu not installed. Install with: pip install sacrebleu")
            return 0.0
    
    def calculate_edit_distance(self, predictions: List[str], references: List[str]) -> float:
        """Calculate average edit distance"""
        try:
            from nltk.edit_distance import edit_distance
            distances = [edit_distance(pred, ref) for pred, ref in zip(predictions, references)]
            return np.mean(distances)
        except ImportError:
            logger.warning("nltk not installed. Install with: pip install nltk")
            return 0.0
    
    def evaluate_model(self, test_data: List[Dict]) -> Dict[str, float]:
        """
        Comprehensive model evaluation
        
        Args:
            test_data: Test dataset
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Extract input texts and references
        input_texts = [f"correct: {example['target_text']}" for example in test_data]
        references = [example['target_text'] for example in test_data]
        
        # Generate predictions
        predictions = self.trainer.predict(input_texts)
        
        # Calculate metrics
        metrics = {}
        
        # BLEU score
        bleu_score = self.calculate_bleu_score(predictions, references)
        metrics['bleu_score'] = bleu_score
        
        # Edit distance
        edit_dist = self.calculate_edit_distance(predictions, references)
        metrics['avg_edit_distance'] = edit_dist
        
        # Exact match accuracy
        exact_matches = sum(1 for pred, ref in zip(predictions, references) if pred.strip() == ref.strip())
        metrics['exact_match_accuracy'] = exact_matches / len(predictions)
        
        logger.info(f"Evaluation metrics: {metrics}")
        
        return metrics

def main():
    """Main function to run T5 fine-tuning"""
    # Configuration
    config = ModelConfig(
        model_name="ai-forever/T5-large-spell",
        max_input_length=512,
        max_target_length=512,
        learning_rate=5e-5,
        batch_size=4,  # Reduced for large model
        num_epochs=3,
        gradient_accumulation_steps=8,  # Increased to maintain effective batch size
        output_dir="./t5_spell_finetuned"
    )
    
    # Initialize trainer
    trainer = T5SpellCorrectionTrainer(config)
    
    # Training data path
    training_data_path = "training_data.json"
    
    if not os.path.exists(training_data_path):
        logger.error(f"Training data not found at {training_data_path}")
        logger.info("Please run google_sheets_extractor.py first to prepare training data")
        return
    
    try:
        # Train the model
        test_results = trainer.train(training_data_path)
        
        # Evaluate the model
        evaluator = ModelEvaluator(trainer)
        
        # Load test data for evaluation
        with open(training_data_path, 'r') as f:
            all_data = json.load(f)
        
        # Use a subset for evaluation (or load separate test data)
        test_data = all_data[-100:]  # Use last 100 examples for evaluation
        
        evaluation_metrics = evaluator.evaluate_model(test_data)
        
        logger.info("Training and evaluation completed successfully!")
        logger.info(f"Final evaluation metrics: {evaluation_metrics}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

if __name__ == "__main__":
    main()