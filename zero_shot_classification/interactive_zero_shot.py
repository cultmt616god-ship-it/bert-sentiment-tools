from transformers import pipeline
import torch

# Load the zero-shot model (downloads ~500MB on first run; uses BART/MNLI under the hood)
device = 0 if torch.cuda.is_available() else -1  # GPU if available
classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli",  # Top performer for this; 95%+ accuracy on MNLI benchmark
    device=device
)

print("Zero-shot Classifier - Interactive Mode")
print("Model loaded successfully!")
print("=" * 50)

while True:
    # Get input text from user
    text = input("\nEnter text to classify (or 'quit' to exit): ")
    if text.lower() == 'quit':
        break
    
    if text.strip() == '':
        continue
    
    # Get candidate labels from user
    labels_input = input("Enter candidate labels separated by commas: ")
    if labels_input.strip() == '':
        continue
    
    candidate_labels = [label.strip() for label in labels_input.split(',')]
    
    # Run classification
    try:
        result = classifier(text, candidate_labels)
        
        # Print results
        print(f"\nText: '{text}'")
        print("Best match:", result['labels'][0], f"(score: {result['scores'][0]:.2f})")
        print("All scores:")
        for label, score in zip(result['labels'], result['scores']):
            print(f"  {label}: {score:.2f}")
    except Exception as e:
        print(f"Error occurred: {e}")

print("\nGoodbye!")