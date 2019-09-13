from reappraisal import Model, extrapolate_data, normalize_sentiment, SentimentWrapper
from textblob import TextBlob
import pandas as pd
import spacy

import os
import sys
# import threading
import logging 
from datetime import datetime

from tkinter import Tk
from tkinter.filedialog import askopenfilename






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
    fh = logging.FileHandler(f'output/{starttime}-{strat}.log')
    fh.setFormatter(formatter)
    logger.addHandler(fh)


    if verbose: 
        fhv = logging.FileHandler(f'output/verbose/{starttime}-{strat}.log')
        fhv.setLevel(logging.DEBUG)
        fhv.setFormatter(formatter)
        logger.addHandler(fhv)
    logger.info('Testing Notes: Objectivity Trained, Sentiment: Word and Sentence, Objectivity')





    if strat == 'f':
        logger.info("Far Away Analysis Initialized")
    if strat == 'o':
        logger.info("Objective Analysis Initialized")

    # Read training data
    data = pd.DataFrame(columns = ['Text Response', "Objectivity Score", "Far Away Score"])
    # for filename in os.listdir(cwd + "/input/training"):
    #     data = pd.concat([data, extrapolate_data(cwd + "/input/training/" + filename)], axis = 0)

    data = pd.concat([data, extrapolate_data(cwd + "/input/training/training_data.xlsx")], axis = 0)



    
    # Create linguistic model and fit training data 
    model = Model(data, starttime, strat, verbose)
    model.fit()

    if test:
        # Specify Test Data 
        print("Enter a file for testing (.xlsx or .csv):")
        root = Tk()
        root.withdraw()
        root.wm_attributes('-topmost', True)
        # test_filename = askopenfilename()
        test_filename = ".\\input\\test\\test_data.xlsx"


        # Reading Test Data
        test_data = pd.DataFrame(columns = ['Text Response', "Objectivity Score", "Far Away Score"])
        test_data = pd.concat([test_data, extrapolate_data(test_filename)], axis = 0)
        logger.info(f"{len(test_data)} responses added for testing")
        if strat == 'f':
            ## Far Away Labeling
            test_data.columns = ["Text Response", "Observed Score", "Expected Score"]
        elif strat == 'o':
            ## Objective Labeling 
            test_data.columns = ["Text Response", "Expected Score", "Observed Score"]
        test_data['Observed Score'].values[:] = float('-inf')

        # Predict Scores from Test Data 
        for index, row in test_data.iterrows():
            response = row['Text Response']
            scored_sentence, score = model.predict(response)
            test_data.iloc[index, test_data.columns.get_loc('Observed Score')] = score

        # Save testing data
        test_data.to_csv(f"{cwd}/output/{starttime}-{strat}_test.csv")
        logger.info("Testing Complete")
        # logger.debug(test_data)

        # Test Data Analysis:
        logger.info(f"Correlation: {test_data['Observed Score'].corr(test_data['Expected Score'])}")
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
        
        
main()

# test = TextBlob('The movie yesterday was amazing!')
# print(test.sentiment)
# print(normalize_sentiment(SentimentWrapper(test.sentiment.polarity, test.sentiment.subjectivity)))