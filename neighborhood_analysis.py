# Neighborhood Sentiment Analysis
# This script performs sentiment analysis on a corpus of newspaper articles to analyze 
# the sentiment across neighborhoods. It includes functions for processing text data, 
# extracting sentiment scores, and generating statistics.

import os
import pickle
import bs4
from bs4 import BeautifulSoup as bs
import lxml
import numpy
import pandas as pd
import glob
import tensorflow as tf
from tensorflow import keras
import h5py
import json
import spacy
import tarfile
import zipfile
import re
from collections import Counter
import nltk
from datetime import datetime
import sys
import time
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from statistics import stdev
import matplotlib.pyplot as plt

# List of neighborhoods to analyze
nhoods = ["Midtown", "Enlgewood"]

# ------------------------------------------------------------
# Function: get_doc_sent_vader
# Description: Analyzes the sentiment of text using VADER.
# Input:
#   - text (str): The text to analyze.
# Output:
#   - list: A list of sentiment scores for each sentence with strong sentiment.
# ------------------------------------------------------------
def get_doc_sent_vader(text):
    sents = nltk.tokenize.sent_tokenize(text)
    analyzer = SentimentIntensityAnalyzer()
    sent_scores = []
    for sentence in sents:
        polarities = analyzer.polarity_scores(sentence)
        score = polarities['compound']
        if score > 0.2 or score < -0.2:
            sent_scores.append(score)
    return sent_scores

# ------------------------------------------------------------
# Function: get_score
# Description: Computes the sentiment scores for a given article CSV.
# Input:
#   - filename (str): The name of the CSV file with articles.
# Output:
#   - tuple: A list of scores, final score, date-wise score dictionary, and article count.
# ------------------------------------------------------------
def get_score(filename):
    count = 0
    all_scores = []
    ndf = pd.read_csv(filename)
    date_dict = {}
    
    for index, row in ndf.iterrows():
        if index < 1: 
            continue
        text = row['text']
        date = row['date']
        year = date.split('-')[0]
        
        if type(text) != str or len(text.split()) not in range(10, 6001):
            continue
        
        sent_scores = get_doc_sent_vader(text)
        if not sent_scores:
            count += 1
            continue

        doc_score = sum(sent_scores) / len(sent_scores) if sent_scores else 0
        date_dict.setdefault(year, []).append(doc_score)
        all_scores.append(doc_score)

    final_score = sum(all_scores) / len(all_scores) if all_scores else 0

    with open(f'Sentiment_Scores_Final/{filename[5:-4]}.pkl', 'wb') as file:
        pickle.dump(all_scores, file)

    with open(f'Date_Scores_Final/{filename[5:-4]}.pkl', 'wb') as file:
        pickle.dump(date_dict, file)

    return all_scores, final_score, date_dict, count

# ------------------------------------------------------------
# Function: get_nhood_scores
# Description: Fetches sentiment scores for a specific neighborhood.
# Input:
#   - nhood (str): The neighborhood data filename.
# Output:
#   - int: The number of articles processed.
# ------------------------------------------------------------
def get_nhood_scores(nhood):
    a_scores, a_score, a_dates, a_count = get_score(nhood)
    return a_count

total = sum(get_nhood_scores(f'data/{nhood}.csv') for nhood in nhoods)
print(total)

# ------------------------------------------------------------
# Function: get_date_scores
# Description: Loads date-based sentiment scores from pickle files.
# Output:
#   - dict: A dictionary with date-based sentiment scores.
# ------------------------------------------------------------
def get_date_scores():
    date_scores = {}
    for file in glob.glob('Date_Scores_Final/*.pkl'):
        nhood_dict = pickle.load(open(file, "rb"))
        name = file.replace('Date_Scores_Final/', "").replace('2', '').replace(".pkl", "").strip()
        date_scores[name] = nhood_dict
    return date_scores

# ------------------------------------------------------------
# Function: average_years
# Description: Computes the average sentiment score per year.
# Input:
#   - date_dict (dict): Dictionary with yearly sentiment scores.
# Output:
#   - dict: Dictionary with average scores per year.
# ------------------------------------------------------------
def average_years(date_dict):
    return {key: sum(scores) / len(scores) for key, scores in date_dict.items()}

# ------------------------------------------------------------
# Function: average_years_all
# Description: Computes average sentiment scores for all neighborhoods.
# Input:
#   - nhoods (dict): Neighborhoods with their respective sentiment scores.
# Output:
#   - dict: Neighborhoods with yearly average scores.
# ------------------------------------------------------------
def average_years_all(nhoods):
    return {n: average_years(nhoods[n]) for n in nhoods}

# ------------------------------------------------------------
# Function: get_stats
# Description: Calculates statistics (article counts) for each neighborhood and year.
# Input:
#   - nhoods (dict): Neighborhood sentiment scores.
# Output:
#   - dict: Neighborhood statistics with article counts per year.
# ------------------------------------------------------------
def get_stats(nhoods):
    nhood_to_stats = {}
    for n in nhoods:
        nhood_to_stats[n] = {d: len(scores) for d, scores in nhoods[n].items() if scores}
    return nhood_to_stats

years_std = get_stats(nhoods)
years_std_df = pd.DataFrame.from_dict(years_std).transpose()
years_std_df = years_std_df.reindex(sorted(years_std_df.columns), axis=1)
years_std_df.to_csv('yearly_counts.csv')

# ------------------------------------------------------------
# Function: graph_years
# Description: Plots sentiment trends across years.
# Input:
#   - date_dict (dict): Year-wise sentiment scores.
# ------------------------------------------------------------
def graph_years(date_dict):
    date_pairs = sorted(date_dict.items())
    X, Y = map(list, zip(*date_pairs))
    Y = [(y / 5) - 1 for y in Y]

    if True:  # Five-year averages
        new_X, new_Y, sumY, count = [], [], 0, 0
        for i in range(len(X) - 1):
            count += 1
            sumY += Y[i]
            if count == 5:
                label = f'{X[i - 4]}-{X[i]}'
                new_X.append(label)
                new_Y.append(sumY / 5)
                sumY, count = 0, 0
        X, Y = new_X, new_Y

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(X, Y)
    plt.xticks(rotation=45)
    plt.show()
