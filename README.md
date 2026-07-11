# SMS Spam Classifier

A machine learning project that classifies SMS messages as spam or ham (not spam) using a Naive Bayes classifier built with scikit-learn.

## Overview

This project takes the classic SMS Spam Collection dataset, cleans and processes the text, converts it into numerical features using the Bag of Words model, and trains a Multinomial Naive Bayes classifier to detect spam messages.

## Results

- **Accuracy:** 98.7%
- **Spam precision:** 1.00 (no false alarms — real messages are never wrongly flagged as spam)
- **Spam recall:** 0.91 (catches 91% of actual spam messages)

## How it works

1. **Data loading** — Loads the SMS Spam Collection dataset (tab-separated).
2. **Text cleaning** — Lowercases text and removes punctuation.
3. **Vectorization** — Converts cleaned text into numerical features using `CountVectorizer` (Bag of Words).
4. **Model training** — Trains a Multinomial Naive Bayes classifier on 80% of the data.
5. **Evaluation** — Tests on the remaining 20%, reporting accuracy, precision, and recall.
6. **Interactive testing** — Lets you type any message and see if it's predicted as spam or ham.

## Tech stack

- Python
- pandas
- scikit-learn

## How to run

1. Clone this repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run: `python main.py`
4. Type any message when prompted to see if it's classified as spam or ham

## Dataset

[SMS Spam Collection Dataset](https://archive.ics.uci.edu/dataset/228/sms+spam+collection) — a public dataset of 5,574 SMS messages labeled as spam or ham.

## Future improvements

- Integrate with Gmail API for automatic classification of incoming emails
- Try TF-IDF vectorization for potentially better performance
- Add a web interface using Flask
