import argparse
import logging
import os
import sys
import pdb
import json
import pprint

import numpy as np
import pandas as pd
import spacy
import en_core_web_sm
from textblob import TextBlob
from tqdm import tqdm
import pickle

import data_process
from data_process import reappStrategyFactory, data_partition, SentimentWrapper
from reappraisal import (Model, extrapolate_data,
                         normalize_sentiment, ModelMetadata)

# Parse the arguments passed into the command line.
parser = argparse.ArgumentParser()
parser.add_argument(
    '-e', "--eval", help="Specify a path of a csv or excel to be evaluated. Add if you're not testing for correlation of testing data and just want results.")
parser.add_argument(
    "-f", "--farAway", help="Reappraise using far-away distancing", action='store_true')
parser.add_argument(
    "-o", "--objective", help="Reappraise using objective distancing", action="store_true")
parser.add_argument(
    '-s', '--saveModel', help='Specify a location to save the model once it has been trained')
parser.add_argument('-l', '--load-model',
                    help='Specify a previously trained model to be loaded.')
parser.add_argument(
    "-d", "--dev", help="Used to print debug statements", action='store_true')
args = parser.parse_args()
if not (args.farAway or args.objective):
    parser.exit(
        "Error: Please specify either far-away distancing (-f) or objective distancing (-o)")

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


# Load english language model.
nlp = en_core_web_sm.load()

# load wordnet
try:
    from nltk import wordnet
except ModuleNotFoundError:
    nltk.download("wordnet")
    from nltk import wordnet

def main():
    # Read training data
    cwd = os.getcwd()
    # Generate dataframe for test data
    # If eval is an argument, just train the model and run it on the specified data
    data_train = pd.read_csv('eval/data_train_example.csv')
    if args.eval:
        try: 
            data_test = pd.read_csv(args.eval)
        except:
            data_test = pd.read_excel(args.eval)
    else:
        data_test = pd.read_csv('eval/data_test_example.csv')

    if not os.path.isdir("output"):
        os.makedirs("output")

    with pd.ExcelWriter("output/results.xlsx") as writer:
        # Run train -> test on objective distancing
        if args.objective:
            logger.info("Running algorithm on OBJECTIVE distancing.")
            correl_obj, test_res_obj = run(
                data_train, data_test, nlp, reappStrategyFactory('obj'))
            logger.info(f"Correlation for objective distancing: {correl_obj}")
            test_res_obj.to_excel(writer, sheet_name='objective', index=False)
        if args.farAway:
            # Run train -> test on far away distancing
            logger.info("Running algorithm on FAR AWAY distancing.")
            correl_spatiotemp, test_res_st = run(
                data_train, data_test, nlp, reappStrategyFactory('spatiotemp'))
            if not args.eval:
                logger.info(
                f"Correlation for Far Away distancing: {correl_spatiotemp}")
            test_res_st.to_excel(writer, sheet_name='far away', index=False)
            logger.info("Please see output/results.xlsx for the result of this computation.")


def run(data_train: pd.DataFrame, data_test: pd.DataFrame, nlp, reappStrategy):
    # Prompt user to submit column names for sentences and scores
    print(f"Training Data Columns: {list(data_train.columns)}")
    response_col = input(
        "Please enter the name of the column containing the sentences: ")
    score_col = input(
        "Please enter the name of the column containing the scores to be trained against: ")

    # Create reappraisal model and fit training data
    model = Model(nlp, reappStrategy)
    model.fit(data_train[response_col], data_train[score_col])


    # Prompt user to submit column names for sentences.
    print(f"Test Data Columns: {list(data_test.columns)}")
    response_col_test = input(
        "Please enter the name of the column containing the sentences to be evaluated: ")
    if not args.eval:
        score_col_test = input(
            "Please enter the name of the column containing the scores to test the model against: ")

    # format test data
    logger.info(f"Testing {len(data_test)} responses.")
    data_test['response'] = data_test[response_col_test]
    tqdm.pandas()  # Starts progress bar for prediction.
    data_test['observed'] = data_test.progress_apply(
        lambda row: model.predict(row.response)[1], axis=1)
    if not args.eval:
        correl = data_test['observed'].corr(data_test[score_col_test])
    else:
        correl = np.nan
    data_test.index.name = 'serial'

    metadata = model.export_metadata()
    with open("output/metadata.txt", "w+") as file:
        file.write(f"[{reappStrategy.name} Word Bank]\n")
        file.write(pprint.pformat(metadata.wordtag_scores, indent=4) + "\n")
        file.write(f"[{reappStrategy.name} Weights]\n")
        file.write(pprint.pformat(metadata.weights));

    return correl, data_test


main()
