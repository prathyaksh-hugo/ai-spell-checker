"""
Deployment Pipeline for Quantized T5 Spell Correction Models
Provides REST API endpoints, batch processing, and performance monitoring
"""

import os
import json
import time
import asyncio
from typing import List, Dict, Optional, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import logging

# FastAPI and web serving
try:
    from fastapi import FastAPI, HTTPException, BackgroundTasks
    from pydantic import BaseModel
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    logging.warning("FastAPI not available. Install with: pip install fastapi uvicorn")

# Core ML libraries
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
import numpy as np

# ONNX runtime
try:
    import onnxruntime as ort
    from optimum.onnxruntime import ORTModelForSeq2SeqLM
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic models for API
if FASTAPI_AVAILABLE:
    class SpellCorrectionRequest(BaseModel):
        text: str
        model_type: str = "original"  # original, dynamic_int8, fp16, onnx_quantized
        return_confidence: bool = False
    
    class BatchCorrectionRequest(BaseModel):
        texts: List[str]
        model_type: str = "original"
        return_confidence: bool = False
    
    class CorrectionResponse(BaseModel):
        original_text: str
        corrected_text: str
        model_used: str
        inference_time_ms: float
        confidence: Optional[float] = None
    
    class BatchCorrectionResponse(BaseModel):
        results: List[CorrectionResponse]
        total_texts: int
        avg_inference_time_ms: float
        model_used: str

@dataclass
class ModelInfo:
    """Information about a loaded model"""
    name: str
    path: str
    model_type: str
    size_mb: float
    load_time_s: float
    quantization_method: Optional[str] = None

class ModelManager:
    """Manages loading and inference for multiple quantized models"""
    
    def __init__(self, models_dir: str = "./quantized_models"):
        self.models_dir = models_dir
        self.loaded_models = {}
        self.model_info = {}
        self.tokenizers = {}
        
    def discover_models(self) -> List[str]:
        """Discover available model variants"""
        models = []
        models_path = Path(self.models_dir)
        
        if models_path.exists():
            for model_dir in models_path.iterdir():
                if model_dir.is_dir():
                    config_file = model_dir / "quantization_config.json"
                    if config_file.exists() or (model_dir / "config.json").exists():
                        models.append(model_dir.name)
        
        # Check for original fine-tuned model
        original_path = Path("./t5_spell_finetuned")
        if original_path.exists():
            models.insert(0, "original")
        
        logger.info(f"Discovered models: {models}")
        return models
    
    def load_model(self, model_name: str) -> bool:
        """Load a specific model variant"""
        if model_name in self.loaded_models:
            logger.info(f"Model {model_name} already loaded")
            return True
        
        start_time = time.time()
        
        try:
            if model_name == "original":
                model_path = "./t5_spell_finetuned"
            else:
                model_path = os.path.join(self.models_dir, model_name)
            
            if not os.path.exists(model_path):
                logger.error(f"Model path not found: {model_path}")
                return False
            
            # Load tokenizer
            tokenizer = T5Tokenizer.from_pretrained(model_path)
            self.tokenizers[model_name] = tokenizer
            
            # Load model based on type
            if model_name == "onnx_quantized" and ONNX_AVAILABLE:
                model = ORTModelForSeq2SeqLM.from_pretrained(model_path)
            else:
                model = T5ForConditionalGeneration.from_pretrained(model_path)
                
                # Apply specific loading configurations
                if model_name == "fp16":
                    model = model.half()
                elif "8bit" in model_name:
                    # Model was already quantized during saving
                    pass
            
            self.loaded_models[model_name] = model
            
            # Calculate model info
            load_time = time.time() - start_time
            model_size = self._calculate_model_size(model_path)
            
            # Load quantization info if available
            quantization_method = None
            config_path = os.path.join(model_path, "quantization_config.json")
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    quant_config = json.load(f)
                    quantization_method = quant_config.get("method")
            
            self.model_info[model_name] = ModelInfo(
                name=model_name,
                path=model_path,
                model_type=type(model).__name__,
                size_mb=model_size,
                load_time_s=load_time,
                quantization_method=quantization_method
            )
            
            logger.info(f"Successfully loaded {model_name} in {load_time:.2f}s")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            return False
    
    def _calculate_model_size(self, model_path: str) -> float:
        """Calculate model size in MB"""
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(model_path):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                total_size += os.path.getsize(filepath)
        return total_size / (1024 * 1024)  # Convert to MB
    
    def predict(self, text: str, model_name: str = "original", 
                return_confidence: bool = False) -> Dict:
        """Generate spell correction prediction"""
        if model_name not in self.loaded_models:
            if not self.load_model(model_name):
                raise ValueError(f"Failed to load model: {model_name}")
        
        model = self.loaded_models[model_name]
        tokenizer = self.tokenizers[model_name]
        
        # Prepare input
        input_text = f"correct: {text}" if not text.startswith("correct:") else text
        
        # Tokenize
        inputs = tokenizer(
            input_text,
            max_length=512,
            truncation=True,
            padding=True,
            return_tensors="pt"
        )
        
        # Measure inference time
        start_time = time.time()
        
        with torch.no_grad():
            outputs = model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=512,
                num_beams=4,
                early_stopping=True,
                return_dict_in_generate=True,
                output_scores=return_confidence
            )
        
        inference_time = (time.time() - start_time) * 1000  # Convert to ms
        
        # Decode prediction
        corrected_text = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
        
        result = {
            "original_text": text,
            "corrected_text": corrected_text,
            "model_used": model_name,
            "inference_time_ms": inference_time
        }
        
        # Calculate confidence if requested
        if return_confidence and hasattr(outputs, 'sequences_scores'):
            confidence = torch.exp(outputs.sequences_scores[0]).item()
            result["confidence"] = confidence
        
        return result
    
    def batch_predict(self, texts: List[str], model_name: str = "original",
                     return_confidence: bool = False) -> Dict:
        """Generate batch predictions"""
        results = []
        total_time = 0
        
        for text in texts:
            try:
                result = self.predict(text, model_name, return_confidence)
                results.append(result)
                total_time += result["inference_time_ms"]
            except Exception as e:
                logger.error(f"Batch prediction failed for text: {text[:50]}... Error: {e}")
                results.append({
                    "original_text": text,
                    "corrected_text": text,  # Return original on error
                    "model_used": model_name,
                    "inference_time_ms": 0,
                    "error": str(e)
                })
        
        return {
            "results": results,
            "total_texts": len(texts),
            "avg_inference_time_ms": total_time / len(texts) if texts else 0,
            "model_used": model_name
        }
    
    def get_model_info(self) -> Dict[str, Dict]:
        """Get information about all loaded models"""
        return {name: asdict(info) for name, info in self.model_info.items()}

class PerformanceBenchmark:
    """Performance benchmarking utilities"""
    
    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager
    
    def benchmark_models(self, test_texts: List[str], models: List[str] = None) -> Dict:
        """Benchmark multiple models on test texts"""
        if models is None:
            models = self.model_manager.discover_models()
        
        results = {}
        
        for model_name in models:
            logger.info(f"Benchmarking {model_name}...")
            
            try:
                # Warm up
                if test_texts:
                    self.model_manager.predict(test_texts[0], model_name)
                
                # Benchmark
                start_time = time.time()
                batch_result = self.model_manager.batch_predict(test_texts, model_name)
                total_time = time.time() - start_time
                
                # Calculate metrics
                successful_predictions = [r for r in batch_result["results"] if "error" not in r]
                accuracy = len(successful_predictions) / len(test_texts) if test_texts else 0
                
                results[model_name] = {
                    "total_time_s": total_time,
                    "avg_inference_time_ms": batch_result["avg_inference_time_ms"],
                    "throughput_texts_per_second": len(test_texts) / total_time if total_time > 0 else 0,
                    "accuracy": accuracy,
                    "successful_predictions": len(successful_predictions),
                    "total_predictions": len(test_texts),
                    "model_info": self.model_manager.model_info.get(model_name, {})
                }
                
            except Exception as e:
                logger.error(f"Benchmark failed for {model_name}: {e}")
                results[model_name] = {"error": str(e)}
        
        return results
    
    def save_benchmark_results(self, results: Dict, output_path: str = "benchmark_results.json"):
        """Save benchmark results to file"""
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Benchmark results saved to {output_path}")

# FastAPI application
if FASTAPI_AVAILABLE:
    app = FastAPI(title="T5 Spell Correction API", version="1.0.0")
    model_manager = ModelManager()
    
    @app.on_event("startup")
    async def startup_event():
        """Load models on startup"""
        logger.info("Starting up T5 Spell Correction API...")
        models = model_manager.discover_models()
        
        # Load original model by default
        if "original" in models:
            model_manager.load_model("original")
        
        logger.info("API startup complete")
    
    @app.get("/")
    async def root():
        return {"message": "T5 Spell Correction API", "version": "1.0.0"}
    
    @app.get("/models")
    async def get_models():
        """Get available models"""
        available_models = model_manager.discover_models()
        loaded_models = list(model_manager.loaded_models.keys())
        model_info = model_manager.get_model_info()
        
        return {
            "available_models": available_models,
            "loaded_models": loaded_models,
            "model_info": model_info
        }
    
    @app.post("/load_model/{model_name}")
    async def load_model(model_name: str):
        """Load a specific model"""
        success = model_manager.load_model(model_name)
        if success:
            return {"message": f"Model {model_name} loaded successfully"}
        else:
            raise HTTPException(status_code=400, detail=f"Failed to load model {model_name}")
    
    @app.post("/correct", response_model=CorrectionResponse)
    async def correct_text(request: SpellCorrectionRequest):
        """Correct spelling in a single text"""
        try:
            result = model_manager.predict(
                request.text,
                request.model_type,
                request.return_confidence
            )
            return CorrectionResponse(**result)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/correct_batch", response_model=BatchCorrectionResponse)
    async def correct_batch(request: BatchCorrectionRequest):
        """Correct spelling in multiple texts"""
        try:
            result = model_manager.batch_predict(
                request.texts,
                request.model_type,
                request.return_confidence
            )
            return BatchCorrectionResponse(**result)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/benchmark")
    async def benchmark_models(background_tasks: BackgroundTasks):
        """Run performance benchmark on all models"""
        # Sample test texts
        test_texts = [
            "This is a sampel text with some erors.",
            "Another exampl of incorect speling.",
            "Testing the spel corection system.",
            "Quality assuranc for our organiztion.",
            "User experence guidlines."
        ]
        
        benchmark = PerformanceBenchmark(model_manager)
        
        # Run benchmark in background
        def run_benchmark():
            results = benchmark.benchmark_models(test_texts)
            benchmark.save_benchmark_results(results, "api_benchmark_results.json")
        
        background_tasks.add_task(run_benchmark)
        
        return {"message": "Benchmark started", "test_texts": len(test_texts)}

def create_sample_data():
    """Create sample training data for testing"""
    sample_data = [
        {
            "input_text": "correct: This is a sampel text with some erors.",
            "target_text": "This is a sample text with some errors.",
            "source": "sample_1"
        },
        {
            "input_text": "correct: Another exampl of incorect speling.",
            "target_text": "Another example of incorrect spelling.",
            "source": "sample_2"
        },
        {
            "input_text": "correct: Testing the spel corection system.",
            "target_text": "Testing the spell correction system.",
            "source": "sample_3"
        },
        {
            "input_text": "correct: Quality assuranc for our organiztion.",
            "target_text": "Quality assurance for our organization.",
            "source": "sample_4"
        },
        {
            "input_text": "correct: User experence guidlines are importnt.",
            "target_text": "User experience guidelines are important.",
            "source": "sample_5"
        }
    ] * 20  # Repeat to create more samples
    
    with open("training_data.json", 'w') as f:
        json.dump(sample_data, f, indent=2)
    
    logger.info(f"Created sample training data with {len(sample_data)} examples")

def main():
    """Main deployment function"""
    logger.info("Starting T5 Spell Correction Deployment Pipeline")
    
    # Create sample data if not exists
    if not os.path.exists("training_data.json"):
        create_sample_data()
    
    # Initialize model manager
    model_manager = ModelManager()
    
    # Discover and load models
    available_models = model_manager.discover_models()
    logger.info(f"Available models: {available_models}")
    
    if not available_models:
        logger.warning("No models found. Please run training and quantization first.")
        return
    
    # Load original model
    if "original" in available_models:
        model_manager.load_model("original")
    
    # Run benchmark
    test_texts = [
        "This is a sampel text with some erors.",
        "Another exampl of incorect speling.",
        "Testing the spel corection system.",
        "Quality assuranc for our organiztion.",
        "User experence guidlines are importnt."
    ]
    
    benchmark = PerformanceBenchmark(model_manager)
    results = benchmark.benchmark_models(test_texts, available_models[:3])  # Limit for demo
    benchmark.save_benchmark_results(results)
    
    # Print results
    logger.info("\n" + "="*50)
    logger.info("DEPLOYMENT BENCHMARK RESULTS")
    logger.info("="*50)
    
    for model_name, result in results.items():
        if "error" not in result:
            logger.info(f"{model_name}:")
            logger.info(f"  - Avg inference time: {result['avg_inference_time_ms']:.2f}ms")
            logger.info(f"  - Throughput: {result['throughput_texts_per_second']:.2f} texts/sec")
            logger.info(f"  - Model size: {result['model_info'].get('size_mb', 'Unknown'):.1f}MB")
        else:
            logger.info(f"{model_name}: Error - {result['error']}")
    
    logger.info("="*50)
    
    # Start API server if FastAPI is available
    if FASTAPI_AVAILABLE:
        logger.info("Starting FastAPI server...")
        logger.info("API will be available at: http://localhost:8000")
        logger.info("API docs at: http://localhost:8000/docs")
        
        uvicorn.run(app, host="0.0.0.0", port=8000)
    else:
        logger.info("FastAPI not available. Install with: pip install fastapi uvicorn")

if __name__ == "__main__":
    main()