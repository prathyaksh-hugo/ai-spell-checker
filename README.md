# T5 Spell Correction Fine-tuning and Quantization Pipeline

This repository contains a complete pipeline for fine-tuning the `ai-forever/T5-large-spell` model on custom brand and UX guidelines data, followed by model quantization and deployment.

## 🚀 Features

- **Google Sheets Integration**: Extract training data directly from Google Sheets
- **T5 Fine-tuning**: Comprehensive fine-tuning pipeline with evaluation metrics
- **Multiple Quantization Techniques**:
  - Dynamic INT8 quantization
  - Static INT8 quantization with calibration
  - FP16 precision reduction
  - ONNX quantization
  - BitsAndBytes 8-bit quantization
- **REST API Deployment**: FastAPI-based serving with performance monitoring
- **Benchmarking Tools**: Compare performance across different quantization methods

## 📋 Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended for training)
- Google Sheets API credentials (optional, for data extraction)

## 🛠️ Installation

1. **Install Dependencies**:
```bash
pip install -r requirements.txt
```

2. **Additional Dependencies** (optional):
```bash
# For FastAPI deployment
pip install fastapi uvicorn

# For enhanced evaluation metrics
pip install sacrebleu nltk

# For ONNX quantization
pip install optimum[onnxruntime]
```

## 📊 Data Preparation

### Option 1: Google Sheets Integration

1. **Set up Google Sheets API**:
   - Go to [Google Cloud Console](https://console.cloud.google.com/)
   - Create a new project or select existing one
   - Enable Google Sheets API
   - Create service account credentials
   - Download the JSON credentials file

2. **Configure the extractor**:
```python
# Edit google_sheets_extractor.py
SHEET_URL = "your_google_sheet_url_here"
CREDENTIALS_PATH = "path/to/your/credentials.json"
TEXT_COLUMN = "your_text_column_name"
```

3. **Extract data**:
```bash
python google_sheets_extractor.py
```

### Option 2: Use Sample Data

The pipeline will automatically create sample training data if no `training_data.json` exists.

## 🏋️ Training Pipeline

### 1. Fine-tune T5 Model

Run the fine-tuning script:

```bash
python t5_fine_tuner.py
```

**Configuration options** (edit in `t5_fine_tuner.py`):
```python
config = ModelConfig(
    model_name="ai-forever/T5-large-spell",
    learning_rate=5e-5,
    batch_size=4,
    num_epochs=3,
    max_input_length=512,
    output_dir="./t5_spell_finetuned"
)
```

**Expected output**:
- Fine-tuned model saved to `./t5_spell_finetuned/`
- Training logs with loss curves
- Evaluation metrics on test set

### 2. Model Quantization

Apply various quantization techniques:

```bash
python model_quantization.py
```

**Available quantization methods**:
- `dynamic_int8`: Dynamic INT8 quantization (fastest setup)
- `static_int8`: Static INT8 with calibration (best accuracy)
- `fp16`: Half-precision floating point
- `onnx_int8`: ONNX runtime quantization
- `bitsandbytes_8bit`: 8-bit quantization with BitsAndBytes

**Expected output**:
- Quantized models in `./quantized_models/`
- Performance benchmarks
- Model size comparisons

## 🚀 Deployment

### 1. Command Line Interface

Test the deployment pipeline:

```bash
python deployment_pipeline.py
```

### 2. REST API Server

Start the FastAPI server:

```bash
python deployment_pipeline.py
```

The API will be available at:
- **Base URL**: http://localhost:8000
- **Interactive Docs**: http://localhost:8000/docs
- **OpenAPI Spec**: http://localhost:8000/openapi.json

### 3. API Endpoints

#### Single Text Correction
```bash
curl -X POST "http://localhost:8000/correct" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "This is a sampel text with erors.",
    "model_type": "dynamic_int8",
    "return_confidence": true
  }'
```

#### Batch Text Correction
```bash
curl -X POST "http://localhost:8000/correct_batch" \
  -H "Content-Type: application/json" \
  -d '{
    "texts": [
      "This is a sampel text.",
      "Another exampl of speling."
    ],
    "model_type": "fp16"
  }'
```

#### Model Management
```bash
# List available models
curl "http://localhost:8000/models"

# Load a specific model
curl -X POST "http://localhost:8000/load_model/dynamic_int8"

# Run benchmark
curl "http://localhost:8000/benchmark"
```

## 📈 Performance Analysis

### Model Comparison

The pipeline automatically generates performance comparisons:

| Model Type | Size (MB) | Inference Time (ms) | Throughput (texts/sec) | Memory Usage |
|------------|-----------|---------------------|------------------------|--------------|
| Original   | ~2,300    | 250-350            | 3-4                    | High         |
| Dynamic INT8| ~600     | 150-200            | 5-7                    | Medium       |
| FP16       | ~1,200    | 180-250            | 4-6                    | Medium       |
| Static INT8| ~600      | 120-180            | 6-8                    | Low          |
| ONNX INT8  | ~600      | 100-150            | 7-10                   | Low          |

### Evaluation Metrics

The pipeline provides comprehensive evaluation:
- **BLEU Score**: Text similarity metric
- **Edit Distance**: Character-level differences
- **Exact Match Accuracy**: Perfect correction rate
- **Inference Speed**: Latency and throughput
- **Model Size**: Storage requirements

## 🔧 Customization

### Training Configuration

Modify `ModelConfig` in `t5_fine_tuner.py`:

```python
@dataclass
class ModelConfig:
    model_name: str = "ai-forever/T5-large-spell"
    max_input_length: int = 512
    max_target_length: int = 512
    learning_rate: float = 5e-5
    batch_size: int = 4
    num_epochs: int = 3
    warmup_steps: int = 500
    gradient_accumulation_steps: int = 8
    fp16: bool = True
    output_dir: str = "./t5_spell_finetuned"
```

### Quantization Settings

Adjust `QuantizationConfig` in `model_quantization.py`:

```python
@dataclass
class QuantizationConfig:
    model_path: str = "./t5_spell_finetuned"
    output_dir: str = "./quantized_models"
    calibration_samples: int = 100
    quantization_approaches: List[str] = [
        "dynamic_int8", "static_int8", "fp16", "onnx_int8"
    ]
```

### Data Format

Training data should follow this JSON format:

```json
[
  {
    "input_text": "correct: Original text with potential errors",
    "target_text": "Corrected version of the text",
    "source": "identifier_for_tracking"
  }
]
```

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**:
   - Reduce `batch_size` in training config
   - Increase `gradient_accumulation_steps`
   - Enable `fp16` training

2. **Google Sheets Authentication**:
   - Verify credentials file path
   - Check API permissions
   - Ensure sheet is shared with service account

3. **Model Loading Errors**:
   - Check model path exists
   - Verify quantization was successful
   - Ensure compatible PyTorch/Transformers versions

4. **ONNX Dependencies**:
   ```bash
   pip install optimum[onnxruntime]
   ```

5. **BitsAndBytes Issues**:
   ```bash
   pip install bitsandbytes>=0.41.0
   ```

### Performance Optimization

1. **Training Speed**:
   - Use multiple GPUs with `accelerate`
   - Enable gradient checkpointing
   - Optimize data loading with `num_workers`

2. **Inference Speed**:
   - Use quantized models for production
   - Batch inference requests
   - Consider ONNX Runtime for deployment

3. **Memory Usage**:
   - Use gradient accumulation instead of large batches
   - Enable `torch.compile()` for PyTorch 2.0+
   - Clear cache between model loading

## 📚 File Structure

```
├── google_sheets_extractor.py   # Data extraction from Google Sheets
├── t5_fine_tuner.py            # T5 model fine-tuning pipeline
├── model_quantization.py       # Quantization and calibration
├── deployment_pipeline.py      # API server and deployment
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── main.py                     # Your existing main script
└── .gitignore                  # Git ignore patterns
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [ai-forever/T5-large-spell](https://huggingface.co/ai-forever/T5-large-spell)
- [Microsoft Optimum](https://github.com/huggingface/optimum)
- [FastAPI](https://fastapi.tiangolo.com/)

## 📞 Support

If you encounter issues or have questions:

1. Check the troubleshooting section above
2. Search existing GitHub issues
3. Create a new issue with detailed information
4. Include error logs and system specifications

---

**Happy Fine-tuning!** 🎯