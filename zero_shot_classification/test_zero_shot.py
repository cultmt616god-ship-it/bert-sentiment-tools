from transformers import pipeline
import torch

# Load the zero-shot model
device = 0 if torch.cuda.is_available() else -1  # GPU if available
classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli",
    device=device
)

# Simple test
text = "This is a great product!"
labels = ["positive", "negative", "neutral"]

result = classifier(text, labels)

print("Zero-shot Classifier - Simple Test")
print("=" * 40)
print(f"Text: '{text}'")
print(f"Labels: {labels}")
print("Best match:", result['labels'][0], f"(score: {result['scores'][0]:.2f})")
print("\nAll scores:")
for label, score in zip(result['labels'], result['scores']):
    print(f"  {label}: {score:.2f}")