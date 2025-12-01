import requests
from bs4 import BeautifulSoup
import time
import random
from transformers import pipeline
import torch
import csv

# Load the zero-shot classifier
device = 0 if torch.cuda.is_available() else -1
classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli",
    device=device
)

# Define website categories
website_categories = [
    "news", "entertainment", "shopping", "social_media", 
    "educational", "blog", "government", "business",
    "technology", "sports", "health", "travel"
]

# Headers to mimic a real browser
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
    'Accept-Encoding': 'gzip, deflate',
    'Connection': 'keep-alive',
    'Upgrade-Insecure-Requests': '1',
}

def scrape_title(url):
    """Scrape the title from a given URL"""
    try:
        # Add random delay to be respectful to servers
        time.sleep(random.uniform(1, 3))
        
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        title_tag = soup.find('title')
        
        if title_tag:
            return title_tag.get_text().strip()
        else:
            return "No title found"
    except Exception as e:
        return f"Error scraping title: {str(e)}"

def classify_website(title):
    """Classify website type based on its title"""
    try:
        result = classifier(title, website_categories)
        return {
            'best_match': result['labels'][0],
            'confidence': result['scores'][0],
            'all_scores': dict(zip(result['labels'], result['scores']))
        }
    except Exception as e:
        return f"Error classifying website: {str(e)}"

def read_urls_from_file(filename):
    """Read URLs from a text file (one URL per line)"""
    try:
        with open(filename, 'r') as f:
            urls = [line.strip() for line in f if line.strip()]
        return urls
    except FileNotFoundError:
        print(f"File {filename} not found.")
        return []

def save_results_to_csv(results, filename):
    """Save results to a CSV file"""
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['url', 'title', 'category', 'confidence']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for result in results:
            if isinstance(result['classification'], dict):
                writer.writerow({
                    'url': result['url'],
                    'title': result['title'],
                    'category': result['classification']['best_match'],
                    'confidence': result['classification']['confidence']
                })
            else:
                writer.writerow({
                    'url': result['url'],
                    'title': result['title'],
                    'category': 'Error',
                    'confidence': 0.0
                })

def process_urls(urls, output_file=None):
    """Process a list of URLs"""
    results = []
    
    print("Website Classifier - Processing URLs")
    print("=" * 50)
    
    for i, url in enumerate(urls, 1):
        print(f"\n{i}. Processing: {url}")
        
        # Scrape title
        title = scrape_title(url)
        print(f"   Title: {title}")
        
        # Classify website type
        classification = classify_website(title)
        if isinstance(classification, dict):
            print(f"   Category: {classification['best_match']} (confidence: {classification['confidence']:.2f})")
        else:
            print(f"   {classification}")
        
        results.append({
            'url': url,
            'title': title,
            'classification': classification
        })
    
    # Save to CSV if output file specified
    if output_file:
        save_results_to_csv(results, output_file)
        print(f"\nResults saved to {output_file}")
    
    return results

if __name__ == "__main__":
    # Example usage
    print("Website Classifier - Batch Mode")
    print("1. Create a file named 'urls.txt' with one URL per line")
    print("2. Run this script to process all URLs")
    print("=" * 50)
    
    # Read URLs from file
    urls = read_urls_from_file('urls.txt')
    
    if urls:
        # Process URLs and save results to CSV
        results = process_urls(urls, 'classified_websites.csv')
        
        # Print summary
        print("\n" + "=" * 50)
        print("SUMMARY")
        print("=" * 50)
        for result in results:
            if isinstance(result['classification'], dict):
                print(f"{result['url']} -> {result['classification']['best_match']} ({result['classification']['confidence']:.2f})")
            else:
                print(f"{result['url']} -> Error in classification")
    else:
        print("\nNo URLs found in urls.txt. Creating a sample file...")
        sample_urls = [
            "https://www.bbc.com/news",
            "https://www.amazon.com",
            "https://www.github.com",
            "https://www.reddit.com",
            "https://www.coursera.org"
        ]
        
        with open('urls.txt', 'w') as f:
            for url in sample_urls:
                f.write(url + '\n')
        
        print("Created sample urls.txt file. Run this script again to process these URLs.")