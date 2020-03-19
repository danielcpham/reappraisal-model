# import threading
import logging
import os
import sys
from datetime import datetime

import en_core_web_sm
import numpy as np
import pandas as pd
import spacy
from spacy.tokens import Doc, Token
from textblob import TextBlob
from tqdm import tqdm


from data_process import reappStrategyFactory
from reappraisal import (Model, SentimentWrapper, extrapolate_data,
                         normalize_sentiment)

import argparse


def main():
    strat = None
    verbose = False
    test = False

    parser = argparse.ArgumentParser()
    
    parser.parse_args()

    for arg in sys.argv[1:]:
        if arg == "-v":
            verbose = True
        elif arg == "-f":
            strat = "f"
        elif arg == "-o":
            strat = "o"
        elif arg == '-t':
            test = True
    if not strat:
        strat = input(
            "Press f to initiate a far-away model. Press o to initiate an objectivity model.")

    logger = logging.getLogger("MAIN")
    formatter = logging.Formatter(
        '[%(name)s-%(levelname)s]: %(asctime)-15s: %(message)s')
    # Set logging level of stdout
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    ch.setLevel(logging.INFO)
    logger.addHandler(ch)

    # Read training data
    cwd = os.getcwd()
    data = pd.DataFrame(
        columns=['Text Response', "Objectivity Score", "Far Away Score"])
    for filename in os.listdir(cwd + "/input/training"):
        data = pd.concat([data, extrapolate_data(
            cwd + "/input/training/" + filename).dropna()], axis=0)
    # Remove invalid data
    data = data.dropna()
    # Create the reappraisal strategy and drop the other column
    if strat == 'f':
        data = data.drop('Objectivity Score', axis='columns')
        reappStrategy = reappStrategyFactory('spatiotemp')
        logger.info("Initializing Far Away Model.")
    if strat == 'o':
        data = data.drop('Far Away Score', axis='columns')
        reappStrategy = reappStrategyFactory('obj')
        logger.info("Initializing Objectivity Model.")
        Doc.set_extension(
            "sentiment", getter=lambda doc: TextBlob(doc.text).sentiment)
    # Rename the columns in the data.
    data.columns = ['response', 'score']

    # Check that the spacy language model is loaded; if not, load it.
    nlp = None

    # Bootstrapping
    correls = []

    # TODO: create a blank model that can be referred to.
    for i in range(10):
        # Split data into training data and testing data.
        data_train = data.sample(frac=0.90)
        data_test = data[~data.apply(tuple, 1).isin(
            data_train.apply(tuple, 1))]
        data_test.columns = data.columns
        data_train.reset_index(drop=True, inplace=True)
        data_test.reset_index(drop=True, inplace=True)

        logger.info(data_train['response'])
        logger.info(data_train['score'])

        # Create linguistic model and fit training data
        model = Model(data_train, nlp, reappStrategy, strat)
        model.fit(data_train)

        if test:
            logger.info(f"Testing {len(data_test)} responses.")
            # Create a new column for observed scores.
            data_test.insert(2, 'observed', [np.nan] * len(data_test))
            with tqdm(total=len(data_test)) as pbar:
                for index, response, score, _ in data_test.itertuples():
                    # Using the trained model, predict the score of the response, and return the sentence
                    # as a list of (word, score) tuples.
                    scored_sentence, score = model.predict(response)
                    data_test.at[index, 'observed'] = score
                    pbar.update(1)
            # Test Data Analysis:
            correl = data_test['observed'].corr(data_test['score'])
            logger.info(f"Correlation for run {i}: {correl}")
            correls.append(correl)
        else:
            pass
            # ftypes = [
            # ('Excel files', '*.xlsx'),
            # ('CSV files', '*.csv'),  # semicolon trick
            # # ('Text files', '*.txt'),
            # ('All files', '*'),
            # ]

            # print("Enter a file to evaluate: ")
            # Tk().withdraw()
            # eval_filename = askopenfilename(filetypes = ftypes)
            # # Evaluate by pasting a table
            # if ".xlsx" in eval_filename:
            #     df = pd.read_excel(filename)
            # elif ".csv" in eval_filename:
            #     df = pd.read_csv(filename)
            # # elif ".txt" in eval_filename:
            # #     # Evaluate by pasting text
            # #     nlp
            # #     pass
            # else:
            #     raise("Invalid file type!")
            #     exit()
        # logger.warning("Evaluation not currently implemented.")
    correls = pd.Series(correls)
    print(correls.describe())


main()

# test = TextBlob('The movie yesterday was amazing!')
# print(test.sentiment)
# print(normalize_sentiment(SentimentWrapper(test.sentiment.polarity, test.sentiment.subjectivity)))
