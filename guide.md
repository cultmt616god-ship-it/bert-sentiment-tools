### Step-by-Step Guide: Running DistilBERT for Sentiment Analysis on a 4GB GPU

This guide gets you up and running with the `distilbert-base-uncased-finetuned-sst-2-english` model from Hugging Face. It's lightweight (fits easily in 4GB VRAM), accurate (~91–99% on benchmarks), and handles full-text context for sentiment (positive/negative labels with confidence scores). We'll use Python 3.10+ and the Transformers library.

#### 1. **Prerequisites**
   - **Hardware**: NVIDIA GPU with ≥4GB VRAM (e.g., GTX 1650, RTX 3050). No GPU? It falls back to CPU but slower.
   - **Software**:
     - Python 3.10+ installed.
     - NVIDIA CUDA 11.8+ (for GPU acceleration). Download from [NVIDIA's site](https://developer.nvidia.com/cuda-downloads) if needed.
     - Verify CUDA: Run `nvidia-smi` in terminal—it should show your GPU and CUDA version.
   - **Internet**: For downloading the model (~268MB) on first run.

#### 2. **Set Up Your Environment**
   Create a virtual environment (recommended) and install dependencies:
   ```bash
   # Create and activate venv (on Windows: python -m venv env; env\Scripts\activate)
   python -m venv sentiment_env
   source sentiment_env/bin/activate  # macOS/Linux

   # Install libraries (PyTorch with CUDA support; adjust for your CUDA version if needed)
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118  # For CUDA 11.8
   pip install transformers accelerate  # Core libs; accelerate optimizes for GPU
   pip install sentencepiece  # Optional, for tokenizer efficiency
   ```

   - **Test Installation**: Run this in Python:
     ```python
     import torch
     print(f"PyTorch version: {torch.__version__}")
     print(f"CUDA available: {torch.cuda.is_available()}")
     print(f"GPU name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
     ```
     Expected: CUDA=True and your GPU listed.

#### 3. **Load the Model**
   Use Hugging Face's `pipeline` for simplicity—it handles loading, tokenization, and inference automatically.

   Create a file `sentiment_analyzer.py`:
   ```python
   from transformers import pipeline
   import torch

   # Load the model (downloads on first run; uses GPU if available)
   device = 0 if torch.cuda.is_available() else -1  # 0 = GPU, -1 = CPU
   classifier = pipeline(
       "sentiment-analysis",
       model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
       device=device  # Moves model to GPU
   )

   print("Model loaded successfully!")
   ```

   Run it: `python sentiment_analyzer.py`. It should print "Model loaded successfully!" and use ~0.5–1GB VRAM (check with `nvidia-smi`).

#### 4. **Run Inference (Analyze Text)**
   Add this to your `sentiment_analyzer.py` for testing:
   ```python
   # Example texts
   texts = [
       "I love this product! It's amazing.",
       "The battery dies too fast, not worth it.",
       "This movie wasn't bad at all—pretty good actually.",
       "Meh, average experience."
   ]

   # Analyze (batch for efficiency; handles up to ~32 texts on 4GB)
   results = classifier(texts, batch_size=8)  # Adjust batch_size based on VRAM

   # Print results
   for text, result in zip(texts, results):
       label = result['label']
       score = result['score']
       print(f"Text: '{text}' → {label} (confidence: {score:.2f})")
   ```

   Sample Output:
   ```
   Text: 'I love this product! It's amazing.' → POSITIVE (confidence: 0.99)
   Text: 'The battery dies too fast, not worth it.' → NEGATIVE (confidence: 0.99)
   Text: 'This movie wasn't bad at all—pretty good actually.' → POSITIVE (confidence: 0.98)
   Text: 'Meh, average experience.' → NEGATIVE (confidence: 0.85)
   ```

   - **Speed**: ~50–200 ms per text on a 4GB GPU (faster with larger batches).
   - **Power**: ~50–100W during runs—efficient for local use.

#### 5. **Advanced Tips**
   - **Quantization for Even Less VRAM**: Reduce to ~0.3GB with 8-bit ints:
     ```python
     from transformers import BitsAndBytesConfig
     quantization_config = BitsAndBytesConfig(load_in_8bit=True)
     classifier = pipeline(
         "sentiment-analysis",
         model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
         device=device,
         quantization_config=quantization_config
     )
     ```
     Install extra: `pip install bitsandbytes`.

   - **Process a File (e.g., CSV of Reviews)**:
     ```python
     import pandas as pd
     df = pd.read_csv('reviews.csv')  # Assume column 'text'
     df['sentiment'] = [r['label'] for r in classifier(df['text'].tolist())]
     df.to_csv('analyzed_reviews.csv', index=False)
     ```

   - **Fine-Tuning (If Needed)**: For custom data, use Hugging Face's `Trainer` API. Start with their [tutorial](https://huggingface.co/docs/transformers/tasks/sequence_classification). It fits on 4GB with small datasets (<10k samples).

   - **Troubleshooting**:
     - Out of Memory? Lower `batch_size` to 4 or use CPU (`device=-1`).
     - No CUDA? Install CPU-only PyTorch: `pip install torch --index-url https://download.pytorch.org/whl/cpu`.
     - Errors? Check logs—common fix: `export CUDA_LAUNCH_BLOCKING=1` for debugging.

This setup is production-ready for apps, scripts, or prototypes. For the ultra-light BERT-Mini alternative, swap the model name to `"Varnikasiva/sentiment-classification-bert-mini"`. If you hit snags or need tweaks (e.g., multi-label emotions), drop more details!