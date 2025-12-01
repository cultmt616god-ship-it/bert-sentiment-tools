#!/usr/bin/env python3
"""
Simple script to analyze sentiment of lines in any text file
Usage: python analyze_file.py <input_file> [output_file]
"""

import sys
from transformers import pipeline
import torch
import csv

def analyze_sentiment_file(input_file, output_file=None):
    """Analyze sentiment for each line in a text file"""
    
    # Load the sentiment analysis model
    device = 0 if torch.cuda.is_available() else -1
    classifier = pipeline(
        "sentiment-analysis",
        model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
        device=device
    )
    
    try:
        # Read lines from input file
        with open(input_file, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found.")
        return
    except Exception as e:
        print(f"Error reading file: {str(e)}")
        return
    
    print(f"Analyzing sentiment for {len(lines)} lines from {input_file}")
    print("=" * 60)
    
    results = []
    
    for i, line in enumerate(lines, 1):
        try:
            # Analyze sentiment for this line
            result = classifier(line)[0]
            label = result['label']
            score = result['score']
            
            print(f"{i:3d}. {label} ({score:.2f}) - {line[:60]}{'...' if len(line) > 60 else ''}")
            
            results.append({
                'line_number': i,
                'text': line,
                'sentiment': label,
                'confidence': score
            })
            
        except Exception as e:
            print(f"{i:3d}. ERROR - {line[:60]}{'...' if len(line) > 60 else ''}")
            print(f"     Error: {str(e)}")
            
            results.append({
                'line_number': i,
                'text': line,
                'sentiment': 'Error',
                'confidence': 0.0,
                'error': str(e)
            })
    
    # Save to CSV if output file specified
    if output_file:
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
    
    # Provide summary
    valid_results = [r for r in results if r['sentiment'] != 'Error']
    if valid_results:
        positive_count = sum(1 for r in valid_results if r['sentiment'] == 'POSITIVE')
        negative_count = sum(1 for r in valid_results if r['sentiment'] == 'NEGATIVE')
        total_count = len(valid_results)
        
        print(f"\nSUMMARY:")
        print(f"  Total lines: {len(results)}")
        print(f"  Successful analyses: {total_count}")
        print(f"  Positive sentiment: {positive_count} ({positive_count/total_count*100:.1f}%)")
        print(f"  Negative sentiment: {negative_count} ({negative_count/total_count*100:.1f}%)")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_file.py <input_file> [output_file]")
        print("Example: python analyze_file.py sample_news.txt results.csv")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    analyze_sentiment_file(input_file, output_file)