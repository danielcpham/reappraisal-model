import argparse
import logging
import os
import sys


import numpy as np
import pandas as pd
import spacy
import en_core_web_sm
from textblob import TextBlob
from tqdm import tqdm

import data_process
from data_process import reappStrategyFactory, data_partition
from reappraisal import (Model, SentimentWrapper, extrapolate_data,
                         normalize_sentiment)

# Parse the arguments passed into the command line.


parser = argparse.ArgumentParser()
parser.add_argument(
    "-f", "--farAway", help="reappraise using far-away distancing", action='store_true')
parser.add_argument(
    "-o", "--objective", help="reappraise using objective distancing", action="store_true")
parser.add_argument(
    "-d", "--dev", help="Used to print debug statements", action='store_true')
args = parser.parse_args()
if not (args.farAway or args.objective):
    parser.exit(
        "Error: Please specify either far-away distancing (-f) or objective distancing (-o)")


# Load english language model.
nlp = en_core_web_sm.load()

formatter = logging.Formatter(
    '[%(name)s-%(levelname)s]: %(asctime)-15s: %(message)s')
ch = logging.StreamHandler(sys.stdout)
ch.setFormatter(formatter)
if args.dev:
    ch.setLevel(logging.DEBUG)
else:
    ch.setLevel(logging.INFO)
# Set logger.
logger = logging.getLogger("MAIN")
logger.setLevel(logging.DEBUG)
# Set logging level of stdout
logger.addHandler(ch)



def main():
    # Read training data
    cwd = os.getcwd()
    data = pd.DataFrame(
        columns=[['response', 'score_spatiotemp', 'score_obj']])
    for filename in os.listdir(cwd + "/input/training"):
        data = pd.concat([data, extrapolate_data(
            cwd + "/input/training/" + filename).dropna()], axis=0)
        logger.info(f'Added {filename} to training.')
    # Remove invalid data
    if args.dev:
        import pdb
        import faulthandler
        faulthandler.enable()
        # Shrink dataset to check
        data = data.sample(100)
        logger.debug("Smaller Dataset used.")
    data = data.dropna()
    data.reset_index(drop=True, inplace=True)

    # Partition the Data manually.
    data_train, data_test = data_partition(data, frac=0.9)
    data_train.to_csv("output/data_train_fixed.csv")
    data_test.to_csv("output/data_test_fixed.csv")

    with pd.ExcelWriter("output/results.xlsx") as writer:
        # Run train -> test on objective distancing
        if args.objective:
            logger.info("Running algorithm on objective distancing.")
            correl_obj, test_res_obj = run(data_train[['response', 'score_obj']], data_test[['response', 'score_obj']],
                                        nlp, reappStrategyFactory('obj'))
            logger.info(f"Correlation for objective distancing: {correl_obj}")
            print(test_res_obj)
            test_res_obj.to_excel(writer, sheet_name='objective', index=False)
        if args.farAway:
            # Run train -> test on far away distancing
            logger.info("Running algorithm on far away distancing.")
            correl_spatiotemp, test_res_st = run(data_train[['response', 'score_spatiotemp']], data_test[[
                'response', 'score_spatiotemp']], nlp, reappStrategyFactory('spatiotemp'))
            logger.info(
                f"Correlation for Far Away distancing: {correl_spatiotemp}")
            test_res_st.to_excel(writer, sheet_name='far away',index=False)


def run(data_train: pd.DataFrame, data_test: pd.DataFrame, nlp, reappStrategy):

    # format test data
    data_test['observed'] = [np.nan] * len(data_test)

    # Create reappraisal model and fit training data
    model = Model(nlp, reappStrategy)
    model.fit(data_train)

    logger.info(f"Testing {len(data_test)} responses.")
    # Create a new column for observed scores.
    with tqdm(total=len(data_test)) as pbar:
        for index, response, score, _ in data_test.itertuples():
            # Using the trained model, predict the score of the response, and return the sentence
            # as a list of (word, score) tuples.
            # Scored Sentence is also returned, can be used for debugging
            _, score = model.predict(response)
            data_test.at[index, 'observed'] = score
            pbar.update(1)
    # with open("fuck.csv", "w+") as csv:
    #     data_test.to_csv(csv, index_label="serial")
    # data_test.to_csv(index_label="serial")
    logger.debug("Hello!")
    # if reappStrategy.name == data_process.farAwayFlag:
    #     correl = data_test['observed'].corr(data_test['score_spatiotemp'])
    # elif reappStrategy.name == data_process.objectiveFlag:
    #     correl = data_test['observed'].corr(data_test['score_obj'])
    correl = 0
    data_test.index.name = 'serial'
    data_test.reset_index(inplace=True)
    return correl, data_test


main()
