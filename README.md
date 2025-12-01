# DistilBERT and Zero-Shot Classification Toolkit

This repository contains various Python scripts for natural language processing tasks, organized into three main categories:

## Directory Structure

### 1. Sentiment Analysis ([sentiment_analysis/](file:///home/lsia/Projects/bert/sentiment_analysis/))
Tools for performing sentiment analysis using DistilBERT:
- Basic sentiment analyzer with predefined examples
- Interactive sentiment analyzer for custom text input
- Advanced sentiment analyzer with quantization options
- Demo scripts with various sample texts

### 2. Zero-Shot Classification ([zero_shot_classification/](file:///home/lsia/Projects/bert/zero_shot_classification/))
Tools for classifying text into custom categories without training:
- Basic zero-shot classifier with predefined examples
- Interactive zero-shot classifier for custom inputs
- Demo scripts showing various use cases (sentiment, topic, intent classification)
- Multi-label classification examples
- Guide documentation ([zero-shot.md](file:///home/lsia/Projects/bert/zero_shot_classification/zero-shot.md))

### 3. Web Scraping and Classification ([web_scraping/](file:///home/lsia/Projects/bert/web_scraping/))
Tools for scraping website titles and classifying website types:
- Basic website classifier for predefined URLs
- Interactive website classifier for custom URL input
- Batch processor that reads URLs from files and exports to CSV
- Test scripts and sample data files

## Setup

All scripts require the virtual environment to be activated:
```bash
source sentiment_env/bin/activate
```

## Requirements

- Python 3.10+
- NVIDIA GPU with ≥4GB VRAM (optional, falls back to CPU)
- CUDA 11.8+ (for GPU acceleration)
