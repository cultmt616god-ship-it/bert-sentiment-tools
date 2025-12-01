from transformers import pipeline
import torch

# Load the model (downloads on first run; uses GPU if available)
device = 0 if torch.cuda.is_available() else -1  # 0 = GPU, -1 = CPU
classifier = pipeline(
    "sentiment-analysis",
    model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
    device=device  # Moves model to GPU
)

print("DistilBERT Sentiment Analyzer - Demo")
print("Model loaded successfully!")
print("=" * 40)

# Test texts
test_texts = [
    "This is a great tutorial!",
    "I'm not sure I like this approach.",
    "The weather is okay today.",
    "Python programming is fun and powerful!",
    "I hate waiting in long lines."
]

print("Analyzing sample texts:")
for text in test_texts:
    result = classifier(text)[0]
    label = result['label']
    score = result['score']
    print(f"Text: '{text}'")
    print(f"Sentiment: {label} (confidence: {score:.2f})")
    print("-" * 40)

print("Demo complete!")