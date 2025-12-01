from transformers import pipeline
import torch
import csv
import sys

# Load the sentiment analysis model
device = 0 if torch.cuda.is_available() else -1
classifier = pipeline(
    "sentiment-analysis",
    model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
    device=device
)

def analyze_sentiment_from_file(input_file, output_file=None):
    """Analyze sentiment for each line in a text file"""
    
    try:
        # Read lines from input file
        with open(input_file, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found.")
        return []
    except Exception as e:
        print(f"Error reading file: {str(e)}")
        return []
    
    print(f"Sentiment Analysis for {len(lines)} Lines")
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
    
    # Save to CSV if output file specified
    if output_file:
        save_results_to_csv(results, output_file)
    
    return results

def analyze_sentiment_from_text(text_block, output_file=None):
    """Analyze sentiment for each line in a text block"""
    
    # Split text into lines
    lines = text_block.strip().split('\n')
    
    print(f"Sentiment Analysis for {len(lines)} Lines")
    print("=" * 50)
    
    results = []
    
    for i, line in enumerate(lines, 1):
        if line.strip():  # Skip empty lines
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
    
    # Save to CSV if output file specified
    if output_file:
        save_results_to_csv(results, output_file)
    
    return results

def save_results_to_csv(results, output_file):
    """Save results to a CSV file"""
    try:
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['line_number', 'sentiment', 'confidence', 'text']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for result in results:
                # Only write the main fields to CSV
                writer.writerow({
                    'line_number': result['line_number'],
                    'sentiment': result['sentiment'],
                    'confidence': result['confidence'],
                    'text': result['text']
                })
        
        print(f"\nResults saved to {output_file}")
    except Exception as e:
        print(f"Error saving to CSV: {str(e)}")

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
    # Check if a file argument was provided
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else None
        results = analyze_sentiment_from_file(input_file, output_file)
    else:
        # Use sample text
        print("No input file provided. Using sample text.")
        results = analyze_sentiment_from_text(sample_text, "sentiment_results.csv")
    
    # Provide summary
    summarize_results(results)