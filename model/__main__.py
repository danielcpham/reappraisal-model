import reappraisal

import pandas as pd
import spacy

import os
import sys
import threading
import logging

from tkinter import Tk
from tkinter.filedialog import askopenfilename


FORMAT = '%(asctime)-15s: %(message)s'

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
        elif arg == "-t":
            test = True 
        else:
            raise(Exception(f"Invalid Argument: {arg}"))
    if verbose:
        logging.basicConfig(level=logging.DEBUG, format=FORMAT)
        logging.debug("Verbose Logging Enabled")
    else:
        logging.basicConfig(level=logging.INFO, format=FORMAT)
    if strat == 'f':
        logging.info("Far Away Analysis Initialized")
    if strat == 'o':
        logging.info("Objective Analysis Initialized")

    # Read training data
    data = pd.DataFrame(columns = ['Text Response', "Objectivity Score", "Far Away Score"])
    for filename in os.listdir(cwd + "/input/training"):
        data = pd.concat([data, reappraisal.extrapolate_data(cwd + "/input/training/" + filename)], axis = 0)
    if not strat:
        strat = input("Press f to initiate a far-away model. Press o to initiate an objectivity model.  ")
    # Create linguistic model and fit training data 
    model = reappraisal.Model(data, strat, verbose)
    model.fit()

    if test:
        # Specify Test Data 
        # print("Enter a file for testing (.xlsx or .csv):")
        # root = Tk()
        # root.withdraw()
        # root.wm_attributes('-topmost', True)
        # test_filename = askopenfilename()
        test_filename = "C:\\Users\\palad\\Desktop\\reappraisal-model-spacy\\input\\test\\Proofread_LDH_Data_Readable_Test.xlsx"

        # Reading Test Data
        test_data = pd.DataFrame(columns = ['Text Response', "Objectivity Score", "Far Away Score"])
        test_data = pd.concat([test_data, reappraisal.extrapolate_data(test_filename)], axis = 0)
        logging.info(f"{len(test_data)} responses added for testing")
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
        test_data.to_csv(f"{cwd}/output/{strat}_test.csv")
        logging.info("Testing Complete")
        logging.debug(test_data)

        # Test Data Analysis:
        logging.info(f"Correlation: {test_data['Observed Score'].corr(test_data['Expected Score'])}")
    else:
        ftypes = [
        ('Excel files', '*.xlsx'), 
        ('CSV files', '*.csv'),  # semicolon trick
        # ('Text files', '*.txt'), 
        ('All files', '*'), 
        ]


        print("Enter a file to evaluate: ")
        Tk().withdraw()
        eval_filename = askopenfilename(filetypes = ftypes)
        # Evaluate by pasting a table 
        if ".xlsx" in eval_filename:
            df = pd.read_excel(filename)
        elif ".csv" in eval_filename:
            df = pd.read_csv(filename)
        # elif ".txt" in eval_filename:
        #     # Evaluate by pasting text
        #     nlp
        #     pass
        else:
            raise("Invalid file type!")
            exit()
        
        
main()
