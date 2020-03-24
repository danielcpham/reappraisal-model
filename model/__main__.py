import argparse
import logging
import os
import sys

import en_core_web_sm
import numpy as np
import pandas as pd
import spacy
from textblob import TextBlob
from tqdm import tqdm


from data_process import reappStrategyFactory, data_partition
from reappraisal import (Model, SentimentWrapper, extrapolate_data,
                         normalize_sentiment)

# Parse the arguments passed into the command line.
parser = argparse.ArgumentParser()
parser.add_argument(
    "-f", "--farAway", help="reappraise using far-away distancing", action='store_true')
parser.add_argument(
    "-o", "--objective", help="reappraise using objective distancing", action="store_true")
args = parser.parse_args()
if not (args.farAway or args.objective):
    parser.exit(
        "Error: Please specify either far-away distancing (-f) or objective distancing (-o)")

# Load english language model.
nlp = spacy.load("en_core_web_sm")

# Set logger.
logger = logging.getLogger("MAIN")
formatter = logging.Formatter(
    '[%(name)s-%(levelname)s]: %(asctime)-15s: %(message)s')
# Set logging level of stdout
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler(sys.stdout)
ch.setFormatter(formatter)
ch.setLevel(logging.INFO)
logger.addHandler(ch)


def main():
    # Read training data
    cwd = os.getcwd()
    print(cwd)
    data = pd.DataFrame(
        columns=[['response', 'score_spatiotemp', 'score_obj']])
    for filename in os.listdir(cwd + "/input/training"):
        data = pd.concat([data, extrapolate_data(
            cwd + "/input/training/" + filename).dropna()], axis=0)
        logger.info(f'Added {filename} to training.')
    # Remove invalid data
    data = data.dropna()
    data.reset_index(drop=True, inplace=True)

    # Partition the Data manually.
    data_train, data_test = data_partition(data, frac=0.9)

    # Run train -> test on objective distancing
    if args.objective:
        correl_obj, _ = run(data_train[['response', 'score_obj']], data_test[[
            'response', 'score_obj']], nlp, reappStrategyFactory('obj'))

        logger.info(f"Correlation for objective distancing: {correl_obj}")
    if args.farAway:
        # Run train -> test on far away distancing
        correl_spatiotemp, _ = run(data_train[['response', 'score_spatiotemp']], data_test[[
            'response', 'score_spatiotemp']], nlp, reappStrategyFactory('spatiotemp'))

        logger.info(
            f"Correlation for Far Away distancing: {correl_spatiotemp}")


def run(data_train, data_test, nlp, reappStrategy):

    # Create reappraisal model and fit training data
    model = Model(nlp, reappStrategy)
    model.fit(data_train)

    logger.info(f"Testing {len(data_test)} responses.")
    # Create a new column for observed scores.
    data_test.insert(2, 'observed', [np.nan] * len(data_test))
    with tqdm(total=len(data_test)) as pbar:
        for index, response, score, _ in data_test.itertuples():
            # Using the trained model, predict the score of the response, and return the sentence
            # as a list of (word, score) tuples.
            # Scored Sentence is also returned, can be used for debugging
            _, score = model.predict(response)
            data_test.at[index, 'observed'] = score
            pbar.update(1)

    correl = data_test['observed'].corr(
        data_test[f"score_{reappStrategy.name}"])
    data_test.to_csv(
        f"output/data_test_results_{reappStrategy.name}.csv", index_label="Serial")
    return correl, data_test


main()
