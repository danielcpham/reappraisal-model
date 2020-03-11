from reappraisal import Model, extrapolate_data, normalize_sentiment, SentimentWrapper
from data_process import reappStrategyFactory
from textblob import TextBlob
import pandas as pd
import numpy as np
import spacy
from spacy.tokens import Doc, Token
from tqdm import tqdm

import os
import sys
# import threading
import logging 
from datetime import datetime

# from tkinter import Tk
# from tkinter.filedialog import askopenfilename






def main():
    strat = None
    verbose = False
    test = False
    cwd = os.getcwd()

   
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
        strat = input("Press f to initiate a far-away model. Press o to initiate an objectivity model.")
    
    
    logger = logging.getLogger("ROOT")
    formatter = logging.Formatter('[%(name)s-%(levelname)s]: %(asctime)-15s: %(message)s')
    
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    ch.setLevel(logging.INFO)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(ch)



    starttime = datetime.now().strftime('%y%m%d_%H%M%S')
    





    
    # Read training data
    data = pd.DataFrame(columns = ['Text Response', "Objectivity Score", "Far Away Score"])
    for filename in os.listdir(cwd + "/input/training"):
        data = pd.concat([data, extrapolate_data(cwd + "/input/training/" + filename).dropna()], axis = 0)
    data = data.dropna()
   # Drop the column we don't need
    if strat == 'f':
        data = data.drop('Objectivity Score', axis='columns')
        reappStrategy = reappStrategyFactory('spatiotemp')
        logger.info("Initializing Far Away Model.")
    if strat == 'o':
        data = data.drop('Far Away Score', axis='columns')
        reappStrategy = reappStrategyFactory('obj')
        logger.info("Initializing Objectivity Model.")

        Doc.set_extension("sentiment", getter = lambda doc: TextBlob(doc.text).sentiment)
    data.columns = ['response', 'score']
    data = data.dropna()

    try: 
        nlp = spacy.load('en_core_web_sm')
    except OSError:
        os.system('python -m spacy download en_core_web_sm')
        nlp = spacy.load('en_core_web_sm')


     #bootstrap the data
    # For loop looks useless because it does not implement bootstrap check yet
    correls = []
    for i in range(10):
        # TODO: split data into test and training data 
        # Randomly sample 85% of the data (without replacement) to use as the training data
        # Subtract the training data from thce original data 
        # to get the testing data 
        data2 = data
        data_train = data2.sample(frac = 0.80)
        data_test = data2[~data2.apply(tuple,1).isin(data_train.apply(tuple,1))]
        data_test.columns = data.columns
        # print(data_test)
        # pdb.set_trace()
        data_train.reset_index(drop=True, inplace=True)
        data_test.reset_index(drop=True, inplace=True)
    
    
        # # Create linguistic model and fit training data 
        model = Model(data_train, nlp, reappStrategy, strat)
        model.fit()

    
        
        if test:
            logger.info(f"Testing {len(data_test)} responses.")
            data_test.insert(2, 'observed', [np.nan] * len(data_test))
            # print(data_test)
            with tqdm(total=len(data_test)) as pbar:
                for index,response, score, _ in data_test.itertuples():
                    # logger.info(f"Reading response {index}")
                    scored_sentence, score = model.predict(response)
                    data_test.at[index, 'observed'] = score
                    pbar.update(1)
            # Save testing data
            # test_data.to_csv(f"{cwd}/output/{starttime}-{strat}_test.csv")
            # logger.info("Testing Complete")
            # logger.debug(test_data)

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