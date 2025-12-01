from transformers import pipeline
import torch

# Load the model (downloads on first run; uses GPU if available)
device = 0 if torch.cuda.is_available() else -1  # 0 = GPU, -1 = CPU
classifier = pipeline(
    "sentiment-analysis",
    model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
    device=device  # Moves model to GPU
)

print("DistilBERT Sentiment Analyzer")
print("Model loaded successfully!")
print("Enter text to analyze sentiment (type 'quit' to exit):")

while True:
    user_input = input("\nEnter text: ")
    if user_input.lower() == 'quit':
        break
    
    if user_input.strip() == '':
        continue
        
    # Analyze the text
    result = classifier(user_input)[0]
    label = result['label']
    score = result['score']
    print(f"Sentiment: {label} (confidence: {score:.2f})")

print("Goodbye!")