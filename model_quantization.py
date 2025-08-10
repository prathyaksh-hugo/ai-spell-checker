"""
Model Quantization and Calibration Pipeline
Supports various quantization techniques for T5 spell correction model
"""

import os
import torch
import json
import numpy as np
from typing import List, Dict, Optional, Union, Tuple
import logging
from pathlib import Path
from dataclasses import dataclass

# Core imports
from transformers import T5ForConditionalGeneration, T5Tokenizer
from datasets import Dataset

# Quantization imports
import torch.quantization as torch_quant
from torch.quantization import QuantStub, DeQuantStub
from torch.quantization.quantize_fx import prepare_fx, convert_fx

# ONNX and optimization imports
try:
    import onnx
    import onnxruntime as ort
    from optimum.onnxruntime import ORTModelForSeq2SeqLM, ORTQuantizer
    from optimum.onnxruntime.configuration import OptimizationConfig, QuantizationConfig
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    logging.warning("ONNX dependencies not available. Install with: pip install optimum[onnxruntime]")

# BitsAndBytes for 8-bit quantization
try:
    import bitsandbytes as bnb
    BITSANDBYTES_AVAILABLE = True
except ImportError:
    BITSANDBYTES_AVAILABLE = False
    logging.warning("BitsAndBytes not available. Install with: pip install bitsandbytes")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class QuantizationConfig:
    """Configuration for model quantization"""
    model_path: str = "./t5_spell_finetuned"
    output_dir: str = "./quantized_models"
    calibration_samples: int = 100
    quantization_approaches: List[str] = None
    
    def __post_init__(self):
        if self.quantization_approaches is None:
            self.quantization_approaches = ["dynamic_int8", "static_int8", "fp16", "onnx_int8"]

class CalibrationDataset:
    """Dataset for model calibration during quantization"""
    
    def __init__(self, calibration_data: List[str], tokenizer: T5Tokenizer, max_length: int = 512):
        self.data = calibration_data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text = self.data[idx]
        inputs = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        return {
            "input_ids": inputs["input_ids"].squeeze(),
            "attention_mask": inputs["attention_mask"].squeeze()
        }

class T5QuantizationPipeline:
    """Main quantization pipeline for T5 models"""
    
    def __init__(self, config: QuantizationConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.calibration_dataset = None
        
        # Create output directory
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
    
    def load_model(self):
        """Load the fine-tuned T5 model and tokenizer"""
        logger.info(f"Loading model from {self.config.model_path}")
        
        self.tokenizer = T5Tokenizer.from_pretrained(self.config.model_path)
        self.model = T5ForConditionalGeneration.from_pretrained(self.config.model_path)
        
        logger.info(f"Model loaded. Parameters: {self.model.num_parameters():,}")
    
    def prepare_calibration_data(self, training_data_path: str = "training_data.json"):
        """Prepare calibration dataset from training data"""
        try:
            with open(training_data_path, 'r', encoding='utf-8') as f:
                training_data = json.load(f)
            
            # Extract input texts for calibration
            calibration_texts = [
                example["input_text"] 
                for example in training_data[:self.config.calibration_samples]
            ]
            
            self.calibration_dataset = CalibrationDataset(
                calibration_texts, self.tokenizer
            )
            
            logger.info(f"Prepared calibration dataset with {len(calibration_texts)} samples")
            
        except FileNotFoundError:
            logger.warning(f"Training data not found at {training_data_path}")
            # Create dummy calibration data
            dummy_texts = [
                "correct: This is a sample text for calibration.",
                "correct: Another example sentence for model calibration.",
                "correct: Testing the quantization pipeline with this text."
            ] * (self.config.calibration_samples // 3 + 1)
            
            self.calibration_dataset = CalibrationDataset(
                dummy_texts[:self.config.calibration_samples], self.tokenizer
            )
            
            logger.info("Using dummy calibration data")
    
    def dynamic_quantization_int8(self) -> torch.nn.Module:
        """Apply dynamic int8 quantization"""
        logger.info("Applying dynamic int8 quantization...")
        
        quantized_model = torch.quantization.quantize_dynamic(
            self.model,
            {torch.nn.Linear},  # Quantize linear layers
            dtype=torch.qint8
        )
        
        # Save quantized model
        output_path = os.path.join(self.config.output_dir, "dynamic_int8")
        Path(output_path).mkdir(parents=True, exist_ok=True)
        
        torch.save(quantized_model.state_dict(), os.path.join(output_path, "pytorch_model.bin"))
        self.tokenizer.save_pretrained(output_path)
        
        # Save config
        with open(os.path.join(output_path, "quantization_config.json"), 'w') as f:
            json.dump({"method": "dynamic_int8", "dtype": "qint8"}, f, indent=2)
        
        logger.info(f"Dynamic int8 quantized model saved to {output_path}")
        return quantized_model
    
    def static_quantization_int8(self) -> torch.nn.Module:
        """Apply static int8 quantization with calibration"""
        logger.info("Applying static int8 quantization...")
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Configure quantization
        self.model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        
        # Prepare model for quantization
        prepared_model = torch.quantization.prepare(self.model, inplace=False)
        
        # Calibration phase
        logger.info("Running calibration...")
        with torch.no_grad():
            for i, batch in enumerate(self.calibration_dataset):
                if i >= self.config.calibration_samples:
                    break
                
                inputs = {
                    "input_ids": batch["input_ids"].unsqueeze(0),
                    "attention_mask": batch["attention_mask"].unsqueeze(0)
                }
                
                try:
                    _ = prepared_model(**inputs)
                except Exception as e:
                    logger.warning(f"Calibration step {i} failed: {e}")
                    continue
        
        # Convert to quantized model
        quantized_model = torch.quantization.convert(prepared_model, inplace=False)
        
        # Save quantized model
        output_path = os.path.join(self.config.output_dir, "static_int8")
        Path(output_path).mkdir(parents=True, exist_ok=True)
        
        torch.save(quantized_model.state_dict(), os.path.join(output_path, "pytorch_model.bin"))
        self.tokenizer.save_pretrained(output_path)
        
        # Save config
        with open(os.path.join(output_path, "quantization_config.json"), 'w') as f:
            json.dump({"method": "static_int8", "dtype": "qint8", "calibration_samples": self.config.calibration_samples}, f, indent=2)
        
        logger.info(f"Static int8 quantized model saved to {output_path}")
        return quantized_model
    
    def fp16_quantization(self) -> torch.nn.Module:
        """Apply FP16 quantization"""
        logger.info("Applying FP16 quantization...")
        
        # Convert model to half precision
        fp16_model = self.model.half()
        
        # Save quantized model
        output_path = os.path.join(self.config.output_dir, "fp16")
        Path(output_path).mkdir(parents=True, exist_ok=True)
        
        fp16_model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)
        
        # Save config
        with open(os.path.join(output_path, "quantization_config.json"), 'w') as f:
            json.dump({"method": "fp16", "dtype": "float16"}, f, indent=2)
        
        logger.info(f"FP16 quantized model saved to {output_path}")
        return fp16_model
    
    def onnx_quantization(self) -> Optional[str]:
        """Apply ONNX quantization"""
        if not ONNX_AVAILABLE:
            logger.error("ONNX dependencies not available")
            return None
        
        logger.info("Applying ONNX quantization...")
        
        try:
            # Export to ONNX first
            onnx_output_path = os.path.join(self.config.output_dir, "onnx")
            Path(onnx_output_path).mkdir(parents=True, exist_ok=True)
            
            # Convert to ONNX using Optimum
            onnx_model = ORTModelForSeq2SeqLM.from_pretrained(
                self.config.model_path,
                export=True
            )
            
            # Save ONNX model
            onnx_model.save_pretrained(onnx_output_path)
            self.tokenizer.save_pretrained(onnx_output_path)
            
            # Apply quantization
            quantized_onnx_path = os.path.join(self.config.output_dir, "onnx_quantized")
            Path(quantized_onnx_path).mkdir(parents=True, exist_ok=True)
            
            # Configure quantization
            qconfig = QuantizationConfig(
                is_static=True,
                format="QDQ",  # Quantize-Dequantize format
                mode="IntegerOps",
                activations_dtype="QInt8",
                weights_dtype="QInt8",
                per_channel=True,
                reduce_range=True,
            )
            
            # Create quantizer
            quantizer = ORTQuantizer.from_pretrained(onnx_output_path)
            
            # Prepare calibration data for ONNX
            def calibration_data_reader():
                for i, batch in enumerate(self.calibration_dataset):
                    if i >= self.config.calibration_samples:
                        break
                    yield {
                        "input_ids": batch["input_ids"].unsqueeze(0).numpy(),
                        "attention_mask": batch["attention_mask"].unsqueeze(0).numpy()
                    }
            
            # Apply quantization
            quantizer.quantize(
                save_dir=quantized_onnx_path,
                quantization_config=qconfig,
                calibration_tensors_range=calibration_data_reader(),
            )
            
            # Save config
            with open(os.path.join(quantized_onnx_path, "quantization_config.json"), 'w') as f:
                json.dump({
                    "method": "onnx_quantization",
                    "format": "QDQ",
                    "dtype": "int8",
                    "calibration_samples": self.config.calibration_samples
                }, f, indent=2)
            
            logger.info(f"ONNX quantized model saved to {quantized_onnx_path}")
            return quantized_onnx_path
            
        except Exception as e:
            logger.error(f"ONNX quantization failed: {e}")
            return None
    
    def bitsandbytes_quantization(self) -> Optional[torch.nn.Module]:
        """Apply 8-bit quantization using BitsAndBytes"""
        if not BITSANDBYTES_AVAILABLE:
            logger.error("BitsAndBytes not available")
            return None
        
        logger.info("Applying BitsAndBytes 8-bit quantization...")
        
        try:
            # Load model with 8-bit quantization
            from transformers import BitsAndBytesConfig
            
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                llm_int8_enable_fp32_cpu_offload=False,
            )
            
            # Load quantized model
            quantized_model = T5ForConditionalGeneration.from_pretrained(
                self.config.model_path,
                quantization_config=bnb_config,
                device_map="auto"
            )
            
            # Save quantized model
            output_path = os.path.join(self.config.output_dir, "bitsandbytes_8bit")
            Path(output_path).mkdir(parents=True, exist_ok=True)
            
            quantized_model.save_pretrained(output_path)
            self.tokenizer.save_pretrained(output_path)
            
            # Save config
            with open(os.path.join(output_path, "quantization_config.json"), 'w') as f:
                json.dump({
                    "method": "bitsandbytes_8bit",
                    "load_in_8bit": True,
                    "threshold": 6.0
                }, f, indent=2)
            
            logger.info(f"BitsAndBytes 8-bit quantized model saved to {output_path}")
            return quantized_model
            
        except Exception as e:
            logger.error(f"BitsAndBytes quantization failed: {e}")
            return None
    
    def benchmark_model(self, model: torch.nn.Module, model_name: str, num_samples: int = 10):
        """Benchmark quantized model performance"""
        logger.info(f"Benchmarking {model_name}...")
        
        # Prepare test inputs
        test_texts = [f"correct: Sample text {i} for benchmarking." for i in range(num_samples)]
        
        model.eval()
        inference_times = []
        
        with torch.no_grad():
            for text in test_texts:
                inputs = self.tokenizer(
                    text,
                    max_length=512,
                    truncation=True,
                    padding=True,
                    return_tensors="pt"
                )
                
                # Measure inference time
                start_time = torch.cuda.Event(enable_timing=True)
                end_time = torch.cuda.Event(enable_timing=True)
                
                start_time.record()
                
                try:
                    outputs = model.generate(
                        inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        max_length=512,
                        num_beams=2,
                        early_stopping=True
                    )
                    
                    end_time.record()
                    torch.cuda.synchronize()
                    
                    inference_time = start_time.elapsed_time(end_time)
                    inference_times.append(inference_time)
                    
                except Exception as e:
                    logger.warning(f"Inference failed for {model_name}: {e}")
                    continue
        
        if inference_times:
            avg_time = np.mean(inference_times)
            std_time = np.std(inference_times)
            
            logger.info(f"{model_name} - Average inference time: {avg_time:.2f}ms ± {std_time:.2f}ms")
            
            return {
                "model_name": model_name,
                "avg_inference_time": avg_time,
                "std_inference_time": std_time,
                "num_samples": len(inference_times)
            }
        else:
            logger.error(f"No successful inferences for {model_name}")
            return None
    
    def run_quantization_pipeline(self):
        """Run the complete quantization pipeline"""
        logger.info("Starting quantization pipeline...")
        
        # Load model and prepare calibration data
        self.load_model()
        self.prepare_calibration_data()
        
        results = {}
        
        # Benchmark original model
        original_benchmark = self.benchmark_model(self.model, "original")
        if original_benchmark:
            results["original"] = original_benchmark
        
        # Apply different quantization methods
        for method in self.config.quantization_approaches:
            logger.info(f"\n--- Applying {method} quantization ---")
            
            try:
                if method == "dynamic_int8":
                    quantized_model = self.dynamic_quantization_int8()
                    benchmark = self.benchmark_model(quantized_model, method)
                    
                elif method == "static_int8":
                    quantized_model = self.static_quantization_int8()
                    benchmark = self.benchmark_model(quantized_model, method)
                    
                elif method == "fp16":
                    quantized_model = self.fp16_quantization()
                    benchmark = self.benchmark_model(quantized_model, method)
                    
                elif method == "onnx_int8":
                    onnx_path = self.onnx_quantization()
                    if onnx_path:
                        # Benchmark ONNX model separately if needed
                        benchmark = {"model_name": method, "status": "completed"}
                    else:
                        benchmark = {"model_name": method, "status": "failed"}
                        
                elif method == "bitsandbytes_8bit":
                    quantized_model = self.bitsandbytes_quantization()
                    if quantized_model:
                        benchmark = self.benchmark_model(quantized_model, method)
                    else:
                        benchmark = {"model_name": method, "status": "failed"}
                
                else:
                    logger.warning(f"Unknown quantization method: {method}")
                    continue
                
                if benchmark:
                    results[method] = benchmark
                    
            except Exception as e:
                logger.error(f"Quantization method {method} failed: {e}")
                results[method] = {"model_name": method, "status": "failed", "error": str(e)}
        
        # Save benchmark results
        results_path = os.path.join(self.config.output_dir, "benchmark_results.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Quantization pipeline completed. Results saved to {results_path}")
        
        # Print summary
        self.print_results_summary(results)
        
        return results
    
    def print_results_summary(self, results: Dict):
        """Print a summary of quantization results"""
        logger.info("\n" + "="*50)
        logger.info("QUANTIZATION RESULTS SUMMARY")
        logger.info("="*50)
        
        for method, result in results.items():
            if "avg_inference_time" in result:
                logger.info(f"{method}: {result['avg_inference_time']:.2f}ms avg inference time")
            else:
                status = result.get("status", "unknown")
                logger.info(f"{method}: {status}")
        
        logger.info("="*50)

def main():
    """Main function to run quantization pipeline"""
    # Configuration
    config = QuantizationConfig(
        model_path="./t5_spell_finetuned",
        output_dir="./quantized_models",
        calibration_samples=50,
        quantization_approaches=["dynamic_int8", "fp16", "onnx_int8"]  # Reduced for faster execution
    )
    
    # Initialize quantization pipeline
    quantizer = T5QuantizationPipeline(config)
    
    # Run quantization
    try:
        results = quantizer.run_quantization_pipeline()
        logger.info("Quantization pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"Quantization pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main()