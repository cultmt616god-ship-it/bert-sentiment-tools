from transformers import pipeline
import torch

# Load the zero-shot model (downloads ~500MB on first run; uses BART/MNLI under the hood)
device = 0 if torch.cuda.is_available() else -1  # GPU if available
classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli",  # Top performer for this; 95%+ accuracy on MNLI benchmark
    device=device
)

print("Zero-shot classifier loaded!")

# Your input text and array of choices
sequence_to_classify = "The new iPhone camera is revolutionary but pricey."
candidate_labels = ["budget-friendly", "high-performance", "average", "overhyped"]  # Your array of choices

# Run classification
result = classifier(sequence_to_classify, candidate_labels)

# Print results
print(f"Text: '{sequence_to_classify}'")
print("Best match:", result['labels'][0], f"(score: {result['scores'][0]:.2f})")
print("All scores:", dict(zip(result['labels'], [f"{s:.2f}" for s in result['scores']])))