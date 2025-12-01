from transformers import pipeline
import torch

# Load the sentiment analysis model
device = 0 if torch.cuda.is_available() else -1
classifier = pipeline(
    "sentiment-analysis",
    model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
    device=device
)

def analyze_sentiment_lines(text_block):
    """Analyze sentiment for each line in a text block"""
    
    # Split text into lines
    lines = text_block.strip().split('\n')
    
    print("Sentiment Analysis for Multiple Lines")
    print("=" * 50)
    
    results = []
    
    for i, line in enumerate(lines, 1):
        if line.strip():  # Skip empty lines
            try:
                # Analyze sentiment for this line
                result = classifier(line)[0]
                label = result['label']
                score = result['score']
                
                print(f"{i:2d}. {line}")
                print(f"    Sentiment: {label} (confidence: {score:.2f})")
                
                results.append({
                    'line_number': i,
                    'text': line,
                    'sentiment': label,
                    'confidence': score
                })
                
            except Exception as e:
                print(f"{i:2d}. {line}")
                print(f"    Error: {str(e)}")
                
                results.append({
                    'line_number': i,
                    'text': line,
                    'sentiment': 'Error',
                    'confidence': 0.0,
                    'error': str(e)
                })
    
    return results

def summarize_results(results):
    """Provide a summary of sentiment analysis results"""
    
    positive_count = sum(1 for r in results if r['sentiment'] == 'POSITIVE')
    negative_count = sum(1 for r in results if r['sentiment'] == 'NEGATIVE')
    total_count = len(results)
    
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"Total lines analyzed: {total_count}")
    print(f"Positive sentiment: {positive_count} ({positive_count/total_count*100:.1f}%)")
    print(f"Negative sentiment: {negative_count} ({negative_count/total_count*100:.1f}%)")
    
    # Show most positive and most negative
    if results:
        most_positive = max(results, key=lambda x: x['confidence'] if x['sentiment'] == 'POSITIVE' else -1)
        most_negative = max(results, key=lambda x: x['confidence'] if x['sentiment'] == 'NEGATIVE' else -1)
        
        print(f"\nMost positive: \"{most_positive['text'][:50]}...\" ({most_positive['confidence']:.2f})")
        print(f"Most negative: \"{most_negative['text'][:50]}...\" ({most_negative['confidence']:.2f})")

# Sample text data
sample_text = """Here is What to Know Beyond Why AT&T Inc. (T) is a Trending Stock
T-Mobile Launches Most Aggressive Black Friday Wireless Promotion
AT&T to Release Fourth-Quarter 2025 Earnings on Jan. 28
Copper Thieves Are Wreaking Havoc Across America
T-Mobile Launches Most Aggressive Black Friday Wireless Promotion
Verizon (VZ) Up 4.9% Since Last Earnings Report: Can It Continue?
Why AI Telecom Stock IQSTEL Inc. (IQST) Bets on Cybersecurity as FCC Eliminates Cybersecurity Requirements for Telecom
T-Mobile shares holiday offer customers won't want to pass up
Dallas Is Solving Its Oversupply Of Offices By Converting Them Into Housing, And Public Funds Can Help It Cross The Finish Line: 'It's An Opportunity'
Can AT&T Benefit From EchoStar's Mid-Band Spectrum Deployment?"""

if __name__ == "__main__":
    # Analyze sentiment for each line
    results = analyze_sentiment_lines(sample_text)
    
    # Provide summary
    summarize_results(results)