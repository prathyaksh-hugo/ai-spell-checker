

import os
import sys
import logging
from pathlib import Path

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

# NOTE: Avoid importing training pipeline at module import time to keep API startup light
HugosavePipeline = None  # type: ignore

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('hugosave_pipeline.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

# Expose FastAPI app for `uvicorn main:app --reload`
try:
    # Re-export the app defined in deployment_pipeline.py
    from deployment_pipeline import app as app  # type: ignore
except Exception as import_error:  # pragma: no cover - fallback path
    try:
        # Lightweight fallback so the process still starts and provides guidance
        from fastapi import FastAPI

        app = FastAPI(title="T5 Spell Correction API (fallback)")

        @app.get("/")
        async def _fallback_root():
            return {
                "message": "Fallback API running. Install dependencies to enable full API.",
                "detail": str(import_error),
                "hint": "pip install -r requirements.txt && uvicorn deployment_pipeline:app --reload",
            }
    except Exception:
        # If even FastAPI is unavailable, leave app undefined; uvicorn will surface a clear error
        app = None  # type: ignore

def check_dependencies():
    """Check if required packages are installed"""
    # Map pip names to their importable module names
    required_packages = {
        'torch': 'torch',
        'transformers': 'transformers',
        'datasets': 'datasets',
        'pandas': 'pandas',
        'numpy': 'numpy',
        'nltk': 'nltk',
        'scikit-learn': 'sklearn',
        'tqdm': 'tqdm',
    }

    missing = []
    for pip_name, import_name in required_packages.items():
        try:
            __import__(import_name)
        except ImportError:
            missing.append(pip_name)

    if missing:
        logger.error(f"Missing required packages: {missing}")
        logger.info("Please install them with: pip install -r requirements.txt")
        return False

    return True

def main():
    """Main function to run the Hugosave pipeline"""
    # Lazy import to avoid heavy dependencies during API-only runs
    global HugosavePipeline  # type: ignore
    if HugosavePipeline is None:
        from hugosave_pipeline import HugosavePipeline  # type: ignore
    
    print("=" * 60)
    print("🚀 HUGOSAVE T5 SPELL CORRECTION PIPELINE")
    print("=" * 60)
    print()
    
    # Check dependencies
    logger.info("Checking dependencies...")
    if not check_dependencies():
        sys.exit(1)
    
    # Initialize pipeline
    logger.info("Initializing Hugosave pipeline...")
    pipeline = HugosavePipeline()
    
    try:
        print("🎯 Starting complete pipeline with your Hugosave brand guide...")
        print()
        print("This will:")
        print("  1. ✅ Process your brand guide data")
        print("  2. 🏋️ Train T5 model on Hugosave terminology")
        print("  3. 🔍 Validate with brand-specific examples")
        print("  4. ⚡ Create optimized quantized models")
        print("  5. 🚀 Deploy and benchmark performance")
        print()
        
        # Ask user for confirmation
        response = input("Continue? (y/n): ").lower().strip()
        if response not in ['y', 'yes']:
            print("Pipeline cancelled.")
            return
        
        print("\n🎬 Starting pipeline...")
        
        # Run complete pipeline
        results = pipeline.run_complete_pipeline()
        
        print("\n" + "=" * 60)
        print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
        # Display results summary
        print(f"\n📊 RESULTS SUMMARY:")
        print(f"   📁 Training Examples: {results['training_examples']}")
        print(f"   🎯 Model Path: {results['model_path']}")
        print(f"   ✅ Validation Score: {results['validation_results'].get('validation_score', 0):.1%}")
        print(f"   ⚡ Quantized Models: {len(results['quantization_results'])} variants")
        
        print(f"\n📁 FILES CREATED:")
        files_to_check = [
            results['model_path'],
            "./hugosave_quantized_models",
            "./hugosave_pipeline_results.json",
            "./training_data.json"
        ]
        
        for file_path in files_to_check:
            if Path(file_path).exists():
                if Path(file_path).is_dir():
                    print(f"   ✅ {file_path}/ (directory)")
                else:
                    size_mb = Path(file_path).stat().st_size / (1024*1024)
                    print(f"   ✅ {file_path} ({size_mb:.1f} MB)")
        
        print(f"\n🚀 NEXT STEPS:")
        print(f"   1. Test your model: python deployment_pipeline.py")
        print(f"   2. API docs: http://localhost:8000/docs (after starting deployment)")
        print(f"   3. Add new data: python hugosave_pipeline.py --add-data 'new_file.pdf' --source-name 'update_name'")
        
        print(f"\n📈 PERFORMANCE HIGHLIGHTS:")
        if 'deployment_results' in results:
            for model_name, model_results in list(results['deployment_results'].items())[:3]:
                if isinstance(model_results, dict) and 'avg_inference_time_ms' in model_results:
                    print(f"   🔥 {model_name}: {model_results['avg_inference_time_ms']:.1f}ms avg inference")
        
        print(f"\n💾 Complete results saved to: hugosave_pipeline_results.json")
        print(f"📝 Logs saved to: hugosave_pipeline.log")
        
        print("\n🎯 Your Hugosave spell correction model is ready for production!")
        
    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
        print("\n⚠️  Pipeline interrupted. Partial results may be saved.")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        print(f"\n❌ Pipeline failed: {e}")
        print("\nCheck hugosave_pipeline.log for detailed error information.")
        sys.exit(1)

if __name__ == "__main__":
    main()
