#!/usr/bin/env python3
"""
Hugosave T5 Spell Correction Pipeline
Complete workflow from brand guide to deployed model
"""

import argparse
import sys
import os
import logging
from pathlib import Path
import json

# Local imports
from pdf_data_extractor import process_hugosave_brand_guide, MultiSourceDataManager
from incremental_training_manager import IncrementalTrainingManager
from t5_fine_tuner import T5SpellCorrectionTrainer, ModelConfig
from model_quantization import T5QuantizationPipeline, QuantizationConfig
from deployment_pipeline import ModelManager, PerformanceBenchmark

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HugosavePipeline:
    """Complete pipeline for Hugosave spell correction model"""
    
    def __init__(self):
        self.incremental_manager = IncrementalTrainingManager()
        
    def setup_initial_data(self):
        """Set up initial training data from Hugosave brand guide"""
        logger.info("🚀 Setting up initial Hugosave brand guide data...")
        
        # Process the brand guide
        training_data, data_manager = process_hugosave_brand_guide()
        
        # Create initial version
        initial_version = self.incremental_manager.version_manager.create_new_version(
            training_data,
            ["hugosave_brand_guide"],
            "Initial Hugosave brand guide with spell correction examples"
        )
        
        # Copy to main training file
        import shutil
        shutil.copy2(
            self.incremental_manager.version_manager.base_dir / initial_version / "training_data.json",
            "./training_data.json"
        )
        
        logger.info(f"✅ Initial data ready: {len(training_data)} training examples")
        logger.info(f"📊 Data includes:")
        
        # Count by error type
        error_types = {}
        for example in training_data:
            error_type = example.get("error_type", "unknown")
            error_types[error_type] = error_types.get(error_type, 0) + 1
        
        for error_type, count in error_types.items():
            logger.info(f"   - {error_type}: {count} examples")
        
        return initial_version, training_data
    
    def train_initial_model(self):
        """Train the initial T5 model"""
        logger.info("🏋️ Training initial T5 model on Hugosave brand guide...")
        
        # Configure for Hugosave-specific training
        config = ModelConfig(
            model_name="ai-forever/T5-large-spell",
            learning_rate=5e-5,
            batch_size=4,
            num_epochs=3,
            max_input_length=512,
            max_target_length=512,
            output_dir="./hugosave_t5_model",
            wandb_project="hugosave-spell-correction"
        )
        
        # Initialize trainer
        trainer = T5SpellCorrectionTrainer(config)
        
        # Train model
        test_results = trainer.train("./training_data.json")
        
        logger.info(f"✅ Initial model training completed: {config.output_dir}")
        
        return config.output_dir, test_results
    
    def validate_hugosave_model(self, model_path: str):
        """Validate model with Hugosave-specific test cases"""
        logger.info("🔍 Validating model with Hugosave-specific examples...")
        
        hugosave_test_cases = [
            # Brand terminology
            "Add mony to your Save acount",
            "Managee your Hugosave Debit Card",
            "Check your Net Wroth in the app",
            "Start investing with Wealthcarre",
            "Become a Hugohero todya",
            
            # UI/UX patterns
            "Recieve S$ 10 cash back",
            "Would you like to save your changess?",  # Should be: Save changes?
            "Add money to my Save account",  # Should be: your
            "Login to your account",  # Should be: Sign in
            "Message sent",  # Should be: Message has been sent
            
            # British English
            "Realise your savigns goals",
            "Centre your investmnet strategy",
            "Colour code your Pots",
            "Organisee your spending",
            
            # Common financial terms
            "Auto Top-upp your account",
            "Roundupp your purchases",
            "Investe as you spend",
            "View your Portfollio Composition",
            "Take the Investmnet Personality Quiz",
        ]
        
        validation_results = self.incremental_manager.validate_new_model(
            model_path, hugosave_test_cases
        )
        
        # Display results
        logger.info(f"📋 Validation Results:")
        logger.info(f"   Overall Score: {validation_results.get('validation_score', 0):.2%}")
        
        for i, result in enumerate(validation_results.get('test_results', [])[:5]):
            logger.info(f"   Example {i+1}:")
            logger.info(f"     Input:  {result['input']}")
            logger.info(f"     Output: {result['output']}")
            logger.info(f"     Fixed:  {'✅' if result['improvement'] else '❌'}")
        
        return validation_results
    
    def quantize_models(self, model_path: str):
        """Quantize the trained model"""
        logger.info("⚡ Quantizing models for production deployment...")
        
        config = QuantizationConfig(
            model_path=model_path,
            output_dir="./hugosave_quantized_models",
            calibration_samples=50,
            quantization_approaches=["dynamic_int8", "fp16"]  # Fast approaches for demo
        )
        
        quantizer = T5QuantizationPipeline(config)
        results = quantizer.run_quantization_pipeline()
        
        logger.info("✅ Quantization completed")
        return results
    
    def deploy_and_test(self):
        """Deploy models and run performance tests"""
        logger.info("🚀 Deploying models and running performance tests...")
        
        # Initialize model manager
        model_manager = ModelManager("./hugosave_quantized_models")
        
        # Discover models
        available_models = model_manager.discover_models()
        logger.info(f"📦 Available models: {available_models}")
        
        # Test with Hugosave examples
        test_texts = [
            "Add mony to your Save acount",
            "Managee your Debit Card",
            "Recieve cash back rewards",
            "Your Net Wroth is growing",
            "Sign-in to continue"
        ]
        
        # Benchmark models
        benchmark = PerformanceBenchmark(model_manager)
        results = benchmark.benchmark_models(test_texts, available_models[:3])
        
        # Save results
        benchmark.save_benchmark_results(results, "hugosave_benchmark_results.json")
        
        logger.info("✅ Deployment and testing completed")
        return results
    
    def add_new_data_source(self, source_path: str, source_name: str):
        """Add a new data source and retrain"""
        logger.info(f"📁 Adding new data source: {source_name}")
        
        # Add new data
        new_version = self.incremental_manager.add_new_data_source(
            source_path, source_name
        )
        
        # Incremental training
        model_path = self.incremental_manager.incremental_train()
        
        # Validate new model
        validation_results = self.validate_hugosave_model(model_path)
        
        logger.info(f"✅ New model ready: {model_path}")
        return model_path, validation_results
    
    def run_complete_pipeline(self):
        """Run the complete pipeline from start to finish"""
        logger.info("🎯 Starting complete Hugosave spell correction pipeline...")
        
        try:
            # Step 1: Setup initial data
            initial_version, training_data = self.setup_initial_data()
            
            # Step 2: Train initial model
            model_path, test_results = self.train_initial_model()
            
            # Step 3: Validate model
            validation_results = self.validate_hugosave_model(model_path)
            
            # Step 4: Quantize models
            quantization_results = self.quantize_models(model_path)
            
            # Step 5: Deploy and test
            deployment_results = self.deploy_and_test()
            
            # Summary
            logger.info("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
            logger.info("📊 Summary:")
            logger.info(f"   Data Version: {initial_version}")
            logger.info(f"   Training Examples: {len(training_data)}")
            logger.info(f"   Model Path: {model_path}")
            logger.info(f"   Validation Score: {validation_results.get('validation_score', 0):.2%}")
            logger.info(f"   Quantized Models: {len(quantization_results)} variants")
            
            # Save complete results
            complete_results = {
                "pipeline_completed": True,
                "data_version": initial_version,
                "training_examples": len(training_data),
                "model_path": model_path,
                "validation_results": validation_results,
                "quantization_results": quantization_results,
                "deployment_results": deployment_results,
                "timestamp": self.incremental_manager.version_manager.version_info[initial_version]["created_at"]
            }
            
            with open("hugosave_pipeline_results.json", 'w') as f:
                json.dump(complete_results, f, indent=2, default=str)
            
            logger.info("📁 Complete results saved to: hugosave_pipeline_results.json")
            
            return complete_results
            
        except Exception as e:
            logger.error(f"❌ Pipeline failed: {e}")
            raise

def main():
    parser = argparse.ArgumentParser(
        description="Hugosave T5 Spell Correction Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run complete pipeline
  python hugosave_pipeline.py --full

  # Run individual steps
  python hugosave_pipeline.py --setup-data
  python hugosave_pipeline.py --train
  python hugosave_pipeline.py --validate
  python hugosave_pipeline.py --quantize
  python hugosave_pipeline.py --deploy

  # Add new data source
  python hugosave_pipeline.py --add-data "new_guide.pdf" --source-name "style_guide_v2"
        """
    )
    
    # Pipeline steps
    parser.add_argument("--full", action="store_true", 
                       help="Run the complete pipeline")
    parser.add_argument("--setup-data", action="store_true",
                       help="Setup initial training data")
    parser.add_argument("--train", action="store_true",
                       help="Train the T5 model")
    parser.add_argument("--validate", action="store_true",
                       help="Validate the trained model")
    parser.add_argument("--quantize", action="store_true",
                       help="Quantize the model")
    parser.add_argument("--deploy", action="store_true",
                       help="Deploy and test models")
    
    # Data management
    parser.add_argument("--add-data", type=str,
                       help="Add new data source (file path or text)")
    parser.add_argument("--source-name", type=str,
                       help="Name for the new data source")
    
    # Configuration
    parser.add_argument("--model-path", type=str, default="./hugosave_t5_model",
                       help="Path to the trained model")
    
    args = parser.parse_args()
    
    if len(sys.argv) == 1:
        parser.print_help()
        return
    
    # Initialize pipeline
    pipeline = HugosavePipeline()
    
    try:
        if args.full:
            # Run complete pipeline
            results = pipeline.run_complete_pipeline()
            
        elif args.add_data and args.source_name:
            # Add new data and retrain
            model_path, validation = pipeline.add_new_data_source(
                args.add_data, args.source_name
            )
            logger.info(f"✅ New model trained: {model_path}")
            
        else:
            # Run individual steps
            if args.setup_data:
                version, data = pipeline.setup_initial_data()
                logger.info(f"✅ Data setup completed: {version}")
            
            if args.train:
                model_path, results = pipeline.train_initial_model()
                logger.info(f"✅ Training completed: {model_path}")
            
            if args.validate:
                validation = pipeline.validate_hugosave_model(args.model_path)
                logger.info(f"✅ Validation completed: {validation.get('validation_score', 0):.2%}")
            
            if args.quantize:
                results = pipeline.quantize_models(args.model_path)
                logger.info(f"✅ Quantization completed: {len(results)} variants")
            
            if args.deploy:
                results = pipeline.deploy_and_test()
                logger.info("✅ Deployment completed")
        
        logger.info("🎯 All requested operations completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()