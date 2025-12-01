from transformers import pipeline
import torch

# Load the sentiment analysis model
device = 0 if torch.cuda.is_available() else -1
classifier = pipeline(
    "sentiment-analysis",
    model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
    device=device
)

def interactive_sentiment_analyzer():
    """Interactive sentiment analyzer for multi-line text input"""
    
    print("Interactive Bulk Sentiment Analyzer")
    print("Enter multiple lines of text (one per line)")
    print("Press Enter twice to finish input")
    print("=" * 50)
    
    lines = []
    while True:
        line = input(f"Line {len(lines) + 1}: ")
        if line == "" and lines and lines[-1] == "":
            # Two consecutive empty lines - finish input
            lines.pop()  # Remove the last empty line
            break
        elif line == "" and not lines:
            # First line is empty - continue
            continue
        else:
            lines.append(line)
    
    if not lines:
        print("No text entered.")
        return
    
    print(f"\nAnalyzing {len(lines)} lines...")
    print("=" * 50)
    
    results = []
    
    for i, line in enumerate(lines, 1):
        try:
            # Analyze sentiment for this line
            result = classifier(line)[0]
            label = result['label']
            score = result['score']
            
            print(f"{i:2d}. {line[:70]}{'...' if len(line) > 70 else ''}")
            print(f"    Sentiment: {label} (confidence: {score:.2f})")
            
            results.append({
                'line_number': i,
                'text': line,
                'sentiment': label,
                'confidence': score
            })
            
        except Exception as e:
            print(f"{i:2d}. {line[:70]}{'...' if len(line) > 70 else ''}")
            print(f"    Error: {str(e)}")
            
            results.append({
                'line_number': i,
                'text': line,
                'sentiment': 'Error',
                'confidence': 0.0,
                'error': str(e)
            })
    
    # Provide summary
    summarize_results(results)

def summarize_results(results):
    """Provide a summary of sentiment analysis results"""
    
    # Filter out error results
    valid_results = [r for r in results if r['sentiment'] != 'Error']
    
    if not valid_results:
        print("\nNo valid results to summarize.")
        return
    
    positive_count = sum(1 for r in valid_results if r['sentiment'] == 'POSITIVE')
    negative_count = sum(1 for r in valid_results if r['sentiment'] == 'NEGATIVE')
    total_count = len(valid_results)
    
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"Total lines analyzed: {len(results)}")
    print(f"Successful analyses: {total_count}")
    print(f"Errors: {len(results) - total_count}")
    print(f"Positive sentiment: {positive_count} ({positive_count/total_count*100:.1f}%)")
    print(f"Negative sentiment: {negative_count} ({negative_count/total_count*100:.1f}%)")
    
    # Show most positive and most negative
    if valid_results:
        most_positive = max(valid_results, key=lambda x: x['confidence'] if x['sentiment'] == 'POSITIVE' else -1)
        most_negative = max(valid_results, key=lambda x: x['confidence'] if x['sentiment'] == 'NEGATIVE' else -1)
        
        print(f"\nMost positive: \"{most_positive['text'][:50]}...\" ({most_positive['confidence']:.2f})")
        print(f"Most negative: \"{most_negative['text'][:50]}...\" ({most_negative['confidence']:.2f})")

if __name__ == "__main__":
    interactive_sentiment_analyzer()