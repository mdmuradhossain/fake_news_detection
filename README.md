# Fake News Detection Model

This project aims to build and train machine learning models for detecting fake news in Bangla and English, using
datasets from news sources and Facebook posts.

## Features

- Supports Bangla and English datasets
- Handles Facebook post datasets with custom label mapping
- Includes data scraping, preprocessing, training, and evaluation scripts
- Outputs model performance metrics

## Project Structure

- `src/extract/`: Data scraping and extraction scripts
- `src/model/`: Model training and evaluation scripts
- `data/`: Raw and processed datasets
- `README.md`: Project documentation

## Setup

1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Place your datasets in the `data/` directory.

## Usage

1. **Scrape Bangla News:**
    - Run the Bangla news scraper to collect and save news articles:
      ```bash
      python src/extract/bangla_news_scraper.py
      ```

2. **Train Model:**
    - Run the training script to train and evaluate models:
      ```bash
      python src/model/training_model.py
      python src/model/unified_training_model.py
      ```

## Datasets

- **Bangla News:** Contains `text` and `label` (0: fake, 1: real)
- **English News:** Contains `text` and `label`
- **Facebook Posts (Bangla/English):** Contains `text` and `label` (binary)

## Notes

- Ensure all datasets have at least two samples per class for stratified splitting.
- Labels must be binary (0/1) for training.

