from transformers import pipeline
import torch

# Load the zero-shot model
device = 0 if torch.cuda.is_available() else -1  # GPU if available
classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli",
    device=device
)

print("Zero-shot Classifier - Demo")
print("Model loaded successfully!")
print("=" * 50)

# Example 1: Sentiment analysis with granular labels
print("\n1. Granular Sentiment Analysis:")
text1 = "This movie was absolutely fantastic! Best film I've seen all year."
labels1 = ["very positive", "positive", "neutral", "negative", "very negative"]
result1 = classifier(text1, labels1)
print(f"Text: '{text1}'")
print("Best match:", result1['labels'][0], f"(score: {result1['scores'][0]:.2f})")

# Example 2: Topic classification
print("\n2. Topic Classification:")
text2 = "The new smartphone features an advanced neural processor and improved battery life."
labels2 = ["technology", "sports", "politics", "entertainment", "business"]
result2 = classifier(text2, labels2)
print(f"Text: '{text2}'")
print("Best match:", result2['labels'][0], f"(score: {result2['scores'][0]:.2f})")

# Example 3: Intent detection
print("\n3. Intent Detection:")
text3 = "I'd like to book a flight from New York to London for next Friday."
labels3 = ["book_flight", "check_weather", "order_food", "schedule_meeting", "search_info"]
result3 = classifier(text3, labels3)
print(f"Text: '{text3}'")
print("Best match:", result3['labels'][0], f"(score: {result3['scores'][0]:.2f})")

# Example 4: Product review classification
print("\n4. Product Review Classification:")
text4 = "The battery life is terrible and the screen is too dim to use outdoors."
labels4 = ["quality", "price", "usability", "design", "performance"]
result4 = classifier(text4, labels4)
print(f"Text: '{text4}'")
print("Best match:", result4['labels'][0], f"(score: {result4['scores'][0]:.2f})")

print("\nDemo complete!")