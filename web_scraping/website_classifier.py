import requests
from bs4 import BeautifulSoup
import time
import random
from transformers import pipeline
import torch

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

def process_urls(urls):
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
    
    return results

# Example URLs to test
test_urls = [
    "https://www.bbc.com/news",
    "https://www.amazon.com",
    "https://www.github.com",
    "https://www.reddit.com",
    "https://www.coursera.org"
]

if __name__ == "__main__":
    # Process the test URLs
    results = process_urls(test_urls)
    
    # Print summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    for result in results:
        if isinstance(result['classification'], dict):
            print(f"{result['url']} -> {result['classification']['best_match']} ({result['classification']['confidence']:.2f})")
        else:
            print(f"{result['url']} -> Error in classification")