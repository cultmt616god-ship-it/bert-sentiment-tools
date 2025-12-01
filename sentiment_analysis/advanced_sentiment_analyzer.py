from transformers import pipeline
import torch

# Quantization for Even Less VRAM: Reduce to ~0.3GB with 8-bit ints
device = 0 if torch.cuda.is_available() else -1  # 0 = GPU, -1 = CPU

# For the pipeline, we need to pass the quantization config differently
classifier = pipeline(
    "sentiment-analysis",
    model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
    device=device
)

print("Model loaded successfully!")

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