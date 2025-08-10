#!/usr/bin/env python3
"""
Master Script for T5 Spell Correction Pipeline
Runs the complete pipeline from data preparation to deployment
"""

import argparse
import sys
import os
import subprocess
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_command(command, description):
    """Run a command and handle errors"""
    logger.info(f"Running: {description}")
    logger.info(f"Command: {command}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        logger.info(f"✅ {description} completed successfully")
        if result.stdout:
            logger.info(f"Output: {result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ {description} failed")
        logger.error(f"Error: {e.stderr}")
        return False

def check_dependencies():
    """Check if required dependencies are installed"""
    logger.info("Checking dependencies...")
    
    required_packages = [
        "torch", "transformers", "datasets", "accelerate",
        "pandas", "numpy", "scikit-learn", "tqdm"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        logger.error(f"Missing required packages: {missing_packages}")
        logger.info("Install them with: pip install -r requirements.txt")
        return False
    
    logger.info("✅ All required dependencies are installed")
    return True

def setup_environment():
    """Set up the environment and install dependencies"""
    logger.info("Setting up environment...")
    
    if not Path("requirements.txt").exists():
        logger.error("requirements.txt not found")
        return False
    
    return run_command(
        "pip install -r requirements.txt",
        "Installing dependencies"
    )

def prepare_data(use_google_sheets=False, sheet_url=None):
    """Prepare training data"""
    logger.info("Preparing training data...")
    
    if use_google_sheets and sheet_url:
        # Update the Google Sheets extractor with the provided URL
        logger.info(f"Extracting data from Google Sheets: {sheet_url}")
        return run_command(
            "python google_sheets_extractor.py",
            "Extracting data from Google Sheets"
        )
    else:
        # Check if training data already exists
        if Path("training_data.json").exists():
            logger.info("✅ Training data already exists")
            return True
        else:
            logger.info("Creating sample training data...")
            # Import and run the sample data creation
            try:
                from deployment_pipeline import create_sample_data
                create_sample_data()
                logger.info("✅ Sample training data created")
                return True
            except Exception as e:
                logger.error(f"Failed to create sample data: {e}")
                return False

def run_training():
    """Run the T5 fine-tuning"""
    logger.info("Starting T5 fine-tuning...")
    
    if not Path("training_data.json").exists():
        logger.error("Training data not found. Run data preparation first.")
        return False
    
    return run_command(
        "python t5_fine_tuner.py",
        "Fine-tuning T5 model"
    )

def run_quantization():
    """Run model quantization"""
    logger.info("Starting model quantization...")
    
    if not Path("t5_spell_finetuned").exists():
        logger.error("Fine-tuned model not found. Run training first.")
        return False
    
    return run_command(
        "python model_quantization.py",
        "Quantizing models"
    )

def run_deployment():
    """Run deployment pipeline"""
    logger.info("Starting deployment pipeline...")
    
    return run_command(
        "python deployment_pipeline.py",
        "Running deployment pipeline"
    )

def main():
    parser = argparse.ArgumentParser(
        description="T5 Spell Correction Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline with sample data
  python run_full_pipeline.py --full

  # Run with Google Sheets data
  python run_full_pipeline.py --full --google-sheets --sheet-url "YOUR_SHEET_URL"

  # Run individual steps
  python run_full_pipeline.py --setup
  python run_full_pipeline.py --data
  python run_full_pipeline.py --train
  python run_full_pipeline.py --quantize
  python run_full_pipeline.py --deploy

  # Skip training and use existing model
  python run_full_pipeline.py --quantize --deploy
        """
    )
    
    # Pipeline steps
    parser.add_argument("--full", action="store_true", 
                       help="Run the complete pipeline")
    parser.add_argument("--setup", action="store_true",
                       help="Setup environment and install dependencies")
    parser.add_argument("--data", action="store_true",
                       help="Prepare training data")
    parser.add_argument("--train", action="store_true",
                       help="Run T5 fine-tuning")
    parser.add_argument("--quantize", action="store_true",
                       help="Run model quantization")
    parser.add_argument("--deploy", action="store_true",
                       help="Run deployment pipeline")
    
    # Data source options
    parser.add_argument("--google-sheets", action="store_true",
                       help="Use Google Sheets as data source")
    parser.add_argument("--sheet-url", type=str,
                       help="Google Sheets URL")
    
    # Configuration options
    parser.add_argument("--skip-deps-check", action="store_true",
                       help="Skip dependency checking")
    parser.add_argument("--force", action="store_true",
                       help="Force overwrite existing files")
    
    args = parser.parse_args()
    
    # If no arguments provided, show help
    if len(sys.argv) == 1:
        parser.print_help()
        return
    
    # Track success of each step
    success = True
    
    # Check dependencies first (unless skipped)
    if not args.skip_deps_check:
        if not check_dependencies():
            logger.error("Dependency check failed. Use --setup to install dependencies.")
            sys.exit(1)
    
    # Determine which steps to run
    steps_to_run = []
    
    if args.full:
        steps_to_run = ["setup", "data", "train", "quantize", "deploy"]
    else:
        if args.setup:
            steps_to_run.append("setup")
        if args.data:
            steps_to_run.append("data")
        if args.train:
            steps_to_run.append("train")
        if args.quantize:
            steps_to_run.append("quantize")
        if args.deploy:
            steps_to_run.append("deploy")
    
    if not steps_to_run:
        logger.error("No steps specified. Use --help for options.")
        sys.exit(1)
    
    # Run the pipeline steps
    logger.info(f"Running pipeline steps: {' -> '.join(steps_to_run)}")
    
    for step in steps_to_run:
        logger.info(f"\n{'='*50}")
        logger.info(f"STEP: {step.upper()}")
        logger.info(f"{'='*50}")
        
        if step == "setup":
            success = setup_environment()
        elif step == "data":
            success = prepare_data(args.google_sheets, args.sheet_url)
        elif step == "train":
            success = run_training()
        elif step == "quantize":
            success = run_quantization()
        elif step == "deploy":
            success = run_deployment()
        
        if not success:
            logger.error(f"Step '{step}' failed. Stopping pipeline.")
            sys.exit(1)
    
    logger.info(f"\n{'='*50}")
    logger.info("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
    logger.info(f"{'='*50}")
    
    # Provide next steps
    if "deploy" in steps_to_run:
        logger.info("\n📚 Next Steps:")
        logger.info("1. Check the deployment benchmark results")
        logger.info("2. Test the API endpoints (if FastAPI is installed)")
        logger.info("3. Visit http://localhost:8000/docs for interactive API documentation")
        logger.info("4. Use the quantized models for production deployment")
    
    logger.info("\n📁 Generated Files:")
    files_to_check = [
        "training_data.json",
        "t5_spell_finetuned/",
        "quantized_models/",
        "benchmark_results.json"
    ]
    
    for file_path in files_to_check:
        if Path(file_path).exists():
            if Path(file_path).is_dir():
                logger.info(f"✅ {file_path} (directory)")
            else:
                size = Path(file_path).stat().st_size / (1024*1024)
                logger.info(f"✅ {file_path} ({size:.1f} MB)")
        else:
            logger.info(f"❌ {file_path} (not found)")

if __name__ == "__main__":
    main()