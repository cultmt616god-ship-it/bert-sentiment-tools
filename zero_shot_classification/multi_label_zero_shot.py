from transformers import pipeline
import torch

# Load the zero-shot model
device = 0 if torch.cuda.is_available() else -1  # GPU if available
classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli",
    device=device
)

print("Zero-shot Classifier - Multi-label Mode")
print("Model loaded successfully!")
print("=" * 50)

# Example with multi-label classification
text = "This smartphone has an excellent camera and great battery life, but the price is too high."
candidate_labels = ["quality", "price", "performance", "design", "usability"]

print(f"Text: '{text}'")
print(f"Labels: {candidate_labels}")
print("\nSingle label mode (default):")
result_single = classifier(text, candidate_labels)
print("Best match:", result_single['labels'][0], f"(score: {result_single['scores'][0]:.2f})")

print("\nMulti-label mode:")
result_multi = classifier(text, candidate_labels, multi_label=True)
print("All scores above 0.1 threshold:")
for label, score in zip(result_multi['labels'], result_multi['scores']):
    if score >= 0.1:
        print(f"  {label}: {score:.2f}")

print("\nMulti-label mode is useful when text can match multiple categories simultaneously.")